#!/usr/bin/env python3
"""Figure 9 Model A: controlled selected BEV queries, inside actual UniAD E2E."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, random, sys, time, types
from collections import defaultdict
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'projects/configs').is_dir()), None)
if ROOT is None: raise RuntimeError('Place script under the UniAD repository')
V3 = ROOT/'tools/test/roi_cost_v3'
if not (V3/'profile_roi_cost_v3_thor.py').is_file(): raise RuntimeError('Existing roi_cost_v3 code required')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(V3))
import os
os.chdir(ROOT)
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from profile_roi_cost_v3_thor import import_project_modules, seed_everything, _reset_temporal_state, _get_first_meta

EXPECTED={'bevformer_selective.py':'776fca863ae1bdfe7fa63aaabc3d4961b1b6ee66',
          'bev_change_analysis.py':'7ac2544ad62a7933aedd5d3256dc869d14b71492'}

def load_ae(root):
    root=Path(root)/'projects/mmdet3d_plugin/utils'
    sources={}
    for name,expected in EXPECTED.items():
        raw=(root/name).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        # A Git checkout can convert line endings. Check the repository blob separately.
        if actual!=expected:
            import subprocess
            actual=subprocess.check_output(['git','-C',str(root),'hash-object','--path',str(root/name),str(root/name)],text=True).strip()
        if actual!=expected: raise RuntimeError(f'AE source differs: {name} {actual}; expected {expected}')
        sources[name]=raw.decode('utf-8-sig')
    warp=types.ModuleType('_figure9_warp');warp.__file__=str(root/'bev_change_analysis.py');sys.modules[warp.__name__]=warp
    exec(compile(sources['bev_change_analysis.py'],str(root/'bev_change_analysis.py'),'exec'),warp.__dict__)
    src=sources['bevformer_selective.py'].replace('\r\n','\n')
    old='from projects.mmdet3d_plugin.utils.bev_change_analysis import ('
    if src.count(old)!=1: raise RuntimeError('Unexpected AE import')
    src=src.replace(old,'from _figure9_warp import (')
    if src.count('shift_ref_2d = ref_2d\n')!=2: raise RuntimeError('Unexpected AE reference-point code')
    # Match Thor v2: avoid in-place mutation of the unshifted current references.
    src=src.replace('shift_ref_2d = ref_2d\n','shift_ref_2d = ref_2d.clone()\n')
    ae=types.ModuleType('_figure9_selective');ae.__file__=str(root/'bevformer_selective.py');sys.modules[ae.__name__]=ae
    exec(compile(src,str(root/'bevformer_selective.py'),'exec'),ae.__dict__)
    return ae,warp.warp_prev_bev_to_current

def events(): return (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
def ms(pair): return float(pair[0].elapsed_time(pair[1]))

def install(model,ae,warp):
    head=model.pts_bbox_head; enc=head.transformer.encoder
    n=head.bev_h*head.bev_w
    if (head.bev_h,head.bev_w)!=(200,200) or len(enc.layers)!=6: raise RuntimeError('Expected 200x200 BEV, six encoder layers')
    for layer in enc.layers:
        if tuple(layer.operation_order)!=('self_attn','norm','cross_attn','norm','ffn','norm'): raise RuntimeError('Unexpected encoder layer ordering')
        if not layer.attentions[0].batch_first: raise RuntimeError('Expected batch_first TSA')
    ae.patch_bevformer_selective(model) # installs AE internal methods; native transformer entry remains in use
    state={'mode':'warm','cache':None,'pose':None,'record':None,'mask':None,'indices':None,'ffn':[],'sca':[],'tsa':[]}
    orig_head=head.get_bev_features; orig_point=enc.point_sampling
    def point(self,*a,**kw):
        r=orig_point(*a,**kw)
        state['mask']=r[1].detach() # no host transfer or sync here
        return r
    enc.point_sampling=types.MethodType(point,enc)
    def hook(key):
        def record(mod,inputs):
            state[key].append(int(inputs[0].shape[1]))
        return record
    def instrument_selective_tsa(tsa):
        original=tsa.forward_selective
        def wrapped(self,query,*a,**kw):
            state['tsa'].append(int(query.shape[1]))
            return original(query,*a,**kw)
        tsa.forward_selective=types.MethodType(wrapped,tsa)
    for layer in enc.layers:
        layer.attentions[0].register_forward_pre_hook(hook('tsa'))
        instrument_selective_tsa(layer.attentions[0])
        layer.ffns[0].register_forward_pre_hook(hook('ffn'))
        layer.attentions[1].register_forward_pre_hook(hook('sca'))
    def core(self,*a,**kw):
        state['core_calls']+=1
        ev=events();ev[0].record()
        if state['mode']=='warm':
            result,cache=self._forward_full_with_cache(*a,**kw)
        else:
            if state['cache'] is None or state['warped'] is None: raise RuntimeError('No actual previous-frame cache')
            result,cache=self.forward_selective(*a,update_mask=state['update_mask'],
                prev_bev_warped=state['warped'],prev_layer_bevs_warped=state['warped_layers'],**kw)
        ev[1].record();state['core_event']=ev
        state['next_cache']=cache
        return result
    enc.forward=types.MethodType(core,enc)
    def get(self,mlvl_feats,img_metas,prev_bev=None):
        if state['record'] is not None: raise RuntimeError('Multiple head calls in one E2E forward')
        if mlvl_feats[0].shape[:2]!=(1,6): raise RuntimeError('Expected one sample and six cameras')
        state['core_calls']=0;state['ffn']=[];state['sca']=[];state['tsa']=[];state['mask']=None
        ev=events();t=time.perf_counter();ev[0].record()
        current_pose=(model.l2g_r_mat,model.l2g_t)
        if any(x is None for x in current_pose): raise RuntimeError('Missing LiDAR pose in inference path')
        warp_ev=None
        if state['mode']!='warm':
            if prev_bev is None or state['pose'] is None or len(state['cache'] or [])!=len(enc.layers): raise RuntimeError('Missing previous BEV or layer cache')
            if len(state['indices'])>n: raise RuntimeError('Too many query indices')
            mask=torch.zeros(n,device=mlvl_feats[0].device,dtype=torch.bool)
            mask[torch.as_tensor(state['indices'],device=mask.device,dtype=torch.long)]=True
            state['update_mask']=mask
            warp_ev=events();warp_ev[0].record()
            prev_r,prev_t=state['pose'];curr_r,curr_t=current_pose
            def warp_one(x): return warp(x,prev_r,prev_t,curr_r,curr_t,self.bev_h,self.bev_w,self.pc_range,padding_mode='border')[0]
            # Use the actual preceding full-frame layer outputs; no fake or random BEV.
            state['warped']=warp_one(state['cache'][-1]);state['warped_layers']=[warp_one(x) for x in state['cache']]
            warp_ev[1].record()
        # Thor's own v2 get_bev_features / transformer prepare the real FPN values,
        # position encoding and ego shift. The original transformer may mutate prev_bev.
        result=orig_head(mlvl_feats=mlvl_feats,img_metas=img_metas,prev_bev=prev_bev.clone() if prev_bev is not None else None)
        ev[1].record();host=(time.perf_counter()-t)*1000
        if state['core_calls']!=1 or len(state['next_cache'])!=len(enc.layers): raise RuntimeError('Encoder core/cache incomplete')
        expected=n if state['mode']=='warm' else len(state['indices'])
        expected_counts=[expected]*len(enc.layers) if expected else []
        if any(state[k]!=expected_counts for k in ('tsa','sca','ffn')):
            raise RuntimeError(f'Actual query execution mismatch: expected {expected_counts}, TSA={state["tsa"]}, FFN={state["ffn"]}, SCA={state["sca"]}')
        bev,pos=result
        if tuple(bev.shape)!=(1,n,head.embed_dims): raise RuntimeError(f'Unexpected BEV output {tuple(bev.shape)}')
        state['record']=dict(total=ev,core=state['core_event'],warp=warp_ev,host=host,n=expected,
            layer_count=len(state['ffn']),fpn_shapes=[list(x.shape) for x in mlvl_feats])
        state['cache']=state.pop('next_cache');state['pose']=current_pose
        for k in ('warped','warped_layers','update_mask'):state[k]=None
        return result
    head.get_bev_features=types.MethodType(get,head)
    return state

def tile_order(mask,seed):
    # Warm-up mask from real point_sampling, outside measured forward.
    visible=mask.detach().any(-1).any(0)[0].reshape(200,200).cpu().numpy()
    valid=visible.reshape(20,10,20,10).any(axis=(1,3)).reshape(-1)
    choices=np.flatnonzero(valid)
    if not len(choices): raise RuntimeError('No visible warm-up BEV tile')
    rng=np.random.default_rng(seed);first=int(rng.choice(choices));order=rng.permutation(400).tolist();order.remove(first)
    return [first]+order

def indices_for(tiles,q):
    if not 0<=q<=40000 or q%100:raise ValueError('query count must be a multiple of 100')
    return sorted((t//20*10+y)*200+t%20*10+x for t in tiles[:q//100] for y in range(10) for x in range(10))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('checkpoint');p.add_argument('--ae-root',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--scenes',type=int,default=9);p.add_argument('--repeats',type=int,default=2);p.add_argument('--seed',type=int,default=20261126)
    p.add_argument('--counts',default='0,100,200,400,800,2000,4000,6000,8000,12000,16000,20000,24000,28000,32000,36000,38000,39600,40000')
    p.add_argument('--no-tf32',dest='tf32',action='store_false',default=True);args=p.parse_args()
    counts=list(dict.fromkeys(int(v) for v in args.counts.split(',')))
    if not counts or any(v<0 or v>40000 or v%100 for v in counts) or args.repeats<1 or args.scenes<1:raise ValueError('invalid counts/repeats/scenes')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    ae,warp=load_ae(args.ae_root);seed_everything(args.seed);torch.backends.cuda.matmul.allow_tf32=args.tf32;torch.backends.cudnn.allow_tf32=args.tf32
    cfg=Config.fromfile(args.config);torch.backends.cudnn.benchmark=bool(cfg.get('cudnn_benchmark',False))
    if cfg.get('fp16',None) is not None:raise ValueError('FP16 config not supported')
    import_project_modules(cfg);cfg.model.pretrained=None;cfg.model.train_cfg=None;cfg.data.test.test_mode=True
    if int(cfg.data.test.pop('samples_per_gpu',1))!=1:raise ValueError('samples_per_gpu must be 1')
    dataset=build_dataset(cfg.data.test);groups=defaultdict(list)
    for i,info in enumerate(dataset.data_infos):groups[str(info['scene_token'])].append(i)
    eligible=sorted(s for s,ids in groups.items() if len(ids)>=2);rng=random.Random(args.seed);rng.shuffle(eligible)
    if args.scenes>len(eligible):raise ValueError('not enough scenes')
    pairs=[]
    for s in eligible[:args.scenes]:
        ids=sorted(groups[s],key=lambda i:dataset.data_infos[i]['timestamp']);j=rng.randrange(1,len(ids));pairs.append((s,ids[j-1],ids[j],rng.randrange(1,2**30)))
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'encoder_frame_raw.csv').exists():raise FileExistsError('Use a fresh output directory')
    manifest=dict(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,tf32=args.tf32,counts=counts,repeats=args.repeats,seed=args.seed,pairs=pairs,ae_blobs=EXPECTED,config=str(Path(args.config).resolve()),checkpoint=str(Path(args.checkpoint).resolve()),backbone='original_full_image',selection='nested random 10x10 tiles, exact counts, border warp, no forced out-of-bounds updates')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    base=build_model(cfg.model,test_cfg=cfg.get('test_cfg'));ckpt=load_checkpoint(base,args.checkpoint,map_location='cpu');base.CLASSES=ckpt.get('meta',{}).get('CLASSES',dataset.CLASSES);base.eval().cuda()
    state=install(base,ae,warp);model=MMDataParallel(base,device_ids=[torch.cuda.current_device()])
    def run(idx,seed):
        seed_everything(seed);data=collate([dataset[idx]],samples_per_gpu=1);meta=_get_first_meta(data);sample=str(meta.get('sample_idx',idx));scene=str(meta.get('scene_token',''))
        state['record']=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        with torch.no_grad():result=model(return_loss=False,rescale=True,**data)
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
        if state['record'] is None:raise RuntimeError('Encoder was not called inside E2E')
        return sample,scene,elapsed,torch.cuda.max_memory_allocated()/1048576
    writer=None
    with (out/'encoder_frame_raw.csv').open('w',newline='',encoding='utf-8') as f:
        for si,(scene,widx,idx,seed) in enumerate(pairs):
            for rep in range(args.repeats):
                order=counts.copy();random.Random(seed+rep).shuffle(order)
                for oi,q in enumerate(order):
                    _reset_temporal_state(base);state.update(mode='warm',cache=None,pose=None,indices=None,warped=None,warped_layers=None)
                    run(widx,seed)
                    tiles=tile_order(state['mask'],seed+rep);state.update(mode='measure',indices=indices_for(tiles,q))
                    sample,actual_scene,e2e,mem=run(idx,seed+9999)
                    if actual_scene and actual_scene!=scene:raise RuntimeError('Scene mismatch')
                    r=state['record'];n=r['n']
                    if n!=q:raise RuntimeError('Query count mismatch')
                    visible=state['mask'].any(-1)[:,:,state['indices']].sum(-1).reshape(-1).cpu().tolist() if q else [0]*6
                    if q and not any(visible):raise RuntimeError('All selected queries invisible; cannot fit normal SCA path')
                    row=dict(scene_token=scene,dataset_index=idx,warmup_index=widx,sample_idx=sample,repeat=rep,order=oi,seed=seed+rep,requested_queries=q,actual_queries=n,active_ratio=n/40000,execution_path='zero_selective' if q==0 else ('full_cached' if q==40000 else 'selective'),encoder_ms=ms(r['total']),encoder_core_ms=ms(r['core']),warp_ms=ms(r['warp']) if r['warp'] else 0,encoder_host_ms=r['host'],e2e_forward_ms=e2e,executed_layers=r['layer_count'],tsa_queries_per_layer=json.dumps(state['tsa']),sca_queries_per_layer=json.dumps(state['sca']),ffn_queries_per_layer=json.dumps(state['ffn']),visible_queries_per_camera=json.dumps(visible),peak_allocated_mib=mem,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,fpn_shapes=json.dumps(r['fpn_shapes']))
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);f.flush();print(f'[ENC-A] scene={si+1}/{len(pairs)} rep={rep} q={q:5d} enc={row["encoder_ms"]:.3f} core={row["encoder_core_ms"]:.3f} warp={row["warp_ms"]:.3f} e2e={e2e:.3f}',flush=True)
    (out/'COMPLETE').write_text('ok\n');print('[ENC-A] COMPLETE',out,flush=True)
if __name__=='__main__':main()
