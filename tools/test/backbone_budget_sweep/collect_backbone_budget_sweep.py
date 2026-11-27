#!/usr/bin/env python3
"""Measured 0..100% Backbone/FPN latency sweep inside real UniAD E2E on Thor."""
from __future__ import annotations
import argparse,csv,hashlib,importlib,json,math,os,random,subprocess,sys,time,types
from pathlib import Path
from collections import Counter,defaultdict
from statistics import median
from launch_census_utils import BUDGETS,MODES,LAYOUTS,raw_boxes,plan_for_budget,decide,coverage

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'projects/configs').is_dir()),None)
if ROOT is None:raise RuntimeError('Install under UniAD/tools/test/backbone_budget_sweep')
for p in (ROOT,ROOT/'tools/test',ROOT/'tools/test/cudagraph'):sys.path.insert(0,str(p))
os.chdir(ROOT)
import cv2
cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel,collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from roi_tile_dispatch_runtime import build_fixed_tile_geometry,build_execution_plan,summarize_execution_geometry
from shape_census_utils import describe_tree,signature_id
from profile_roi_tile_dispatch_thor import (_make_padded_camera_image,_make_crop_batch,
    _init_full_feature_canvases,_scatter_crop_features,normalize_backbone_outputs)
STRIDES=(8,16,32,64)


def _parse_budget_list(text, allowed=BUDGETS):
    vals=sorted(set(float(x.strip()) for x in text.split(',') if x.strip()))
    if not vals or any(not any(abs(v-a)<1e-9 for a in allowed) for v in vals):
        raise ValueError(f"budgets must be selected from {allowed}")
    return vals


def _parse_sweep_budget_list(text):
    vals=sorted(set(float(x.strip()) for x in text.split(',') if x.strip()))
    if not vals or any(v < 0 or v > 100 or abs(v / 5.0 - round(v / 5.0)) > 1e-8 for v in vals):
        raise ValueError('sweep budgets must be multiples of 5 in [0,100]')
    return vals


def _parse_mode_budgets(text):
    out={}
    for item in text.split(';'):
        if not item.strip():
            continue
        if '=' not in item:
            raise ValueError('mode budgets must look like roi_batch=2.5,5;adjacency_merge=10,20')
        mode,values=item.split('=',1)
        mode=mode.strip()
        if mode not in MODES or mode in out:
            raise ValueError(f'invalid or duplicate mode in --mode-budgets: {mode}')
        out[mode]=_parse_budget_list(values)
    if not out:
        raise ValueError('--mode-budgets must specify at least one mode')
    return {mode:out.get(mode,[]) for mode in MODES}


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('checkpoint');p.add_argument('--out-dir',required=True)
    p.add_argument('--stage',choices=('pilot','census','sweep'),required=True)
    p.add_argument('--sweep-budgets',default=','.join(str(x) for x in range(0,101,5)),help='0..100 in 5%% increments')
    p.add_argument('--pilot-report')
    p.add_argument('--force-budgets',help='legacy exploratory override, comma-separated; applies to both modes')
    p.add_argument('--mode-budgets',help='per-mode census tiers, e.g. roi_batch=2.5,5;adjacency_merge=10,20')
    p.add_argument('--pilot-modes',default='roi_batch,adjacency_merge',help='comma-separated modes to test in pilot')
    p.add_argument('--pilot-budgets',help='explicit pilot tiers, e.g. 2.5,5,10,20; disables automatic upper extension')
    p.add_argument('--scenes',type=int,default=None,help='pilot default 6; census default 12; 0=all')
    p.add_argument('--frames-per-scene',type=int,default=None,help='pilot default 1; census default 4; 0=all after warmup')
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--seed',type=int,default=20261127)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--flat',type=float,default=.15)
    p.add_argument('--rise',type=float,default=.25);p.add_argument('--min-area-gain',type=float,default=.25)
    p.add_argument('--max-pilot-budget',type=float,default=100.0)
    p.add_argument('--budget-axis',choices=('packed','raw'),default='packed',help='packed = actual B*H*W cap; raw = nominal ROI union')
    p.add_argument('--nvtx',action='store_true');p.add_argument('--no-tf32',action='store_true')
    a=p.parse_args()
    if a.scenes is None:a.scenes=6 if a.stage in ('pilot','sweep') else 12
    if a.frames_per_scene is None:a.frames_per_scene=1 if a.stage in ('pilot','sweep') else 4
    if min(a.scenes,a.frames_per_scene)<0 or min(a.warmup,a.repeats)<1:p.error('invalid counts')
    if not 0<a.flat<a.rise or a.min_area_gain<=0:p.error('invalid knee thresholds')
    try:
        a.pilot_modes=[x.strip() for x in a.pilot_modes.split(',') if x.strip()]
        if not a.pilot_modes or len(set(a.pilot_modes))!=len(a.pilot_modes) or not set(a.pilot_modes)<=set(MODES):
            raise ValueError(f'--pilot-modes must be selected from {MODES}')
        a.pilot_budgets=_parse_budget_list(a.pilot_budgets) if a.pilot_budgets else None
        a.force_budgets=_parse_budget_list(a.force_budgets) if a.force_budgets else None
        a.mode_budgets=_parse_mode_budgets(a.mode_budgets) if a.mode_budgets else None
        a.sweep_budgets=_parse_sweep_budget_list(a.sweep_budgets) if a.stage=='sweep' else None
        if a.max_pilot_budget not in BUDGETS:raise ValueError(f'--max-pilot-budget must be selected from {BUDGETS}')
    except Exception as exc:
        p.error(str(exc))
    if a.stage=='pilot' and (a.mode_budgets or a.force_budgets):p.error('census-only budget override used in pilot')
    if a.stage=='sweep' and (a.pilot_report or a.force_budgets or a.mode_budgets or a.pilot_budgets):p.error('sweep uses --sweep-budgets only')
    if a.stage=='census' and (a.pilot_budgets or a.mode_budgets is None and not (a.pilot_report or a.force_budgets)):
        p.error('census needs --pilot-report, --force-budgets, or --mode-budgets')
    if a.stage=='census' and a.mode_budgets and (a.pilot_report or a.force_budgets):
        p.error('use --mode-budgets by itself; do not combine with other census selectors')
    if a.stage=='pilot' and a.pilot_budgets and len(a.pilot_budgets)<2:
        p.error('--pilot-budgets needs at least two tiers')
    return a


def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def command(xs):
    try:return subprocess.check_output(xs,text=True,stderr=subprocess.STDOUT,timeout=10).strip()
    except Exception as e:return str(e)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(s):
    random.seed(s);np.random.seed(s%(2**32-1));torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def reset(m):
    m.prev_frame_info=dict(prev_bev=None,scene_token=None,prev_pos=0,prev_angle=0)
    for k in ('prev_bev','scene_token','timestamp','l2g_r_mat','l2g_t','test_track_instances'):
        if hasattr(m,k):setattr(m,k,None)
def meta0(d):
    x=d.get('img_metas')
    while True:
        if hasattr(x,'data'):x=x.data
        elif isinstance(x,(list,tuple)) and x:x=x[0]
        else:return x if isinstance(x,dict) else {}
def stable_seed(*v):return int(hashlib.sha256('|'.join(map(str,v)).encode()).hexdigest()[:8],16)

class Writer:
    def __init__(self,p):self.f=open(p,'w',newline='',encoding='utf8');self.w=None
    def write(self,rows):
        for r in rows:
            if self.w is None:self.w=csv.DictWriter(self.f,fieldnames=list(r));self.w.writeheader()
            self.w.writerow(r)
        self.f.flush()
    def close(self):self.f.close()


def cases(ds,a):
    groups=defaultdict(list)
    for i,d in enumerate(ds.data_infos):groups[str(d['scene_token'])].append(i)
    for v in groups.values():v.sort(key=lambda i:ds.data_infos[i]['timestamp'])
    scenes=sorted(k for k,v in groups.items() if len(v)>a.warmup)
    random.Random(a.seed).shuffle(scenes)
    if a.scenes>len(scenes):raise ValueError('not enough eligible scenes')
    selected=scenes[:a.scenes] if a.scenes else scenes
    out=[]
    for j,s in enumerate(selected):
        v=groups[s];n=min(a.frames_per_scene,len(v)-a.warmup) if a.frames_per_scene else len(v)-a.warmup
        # Full mode starts at scene's first frame; sampled mode varies scene position.
        start=a.warmup if not a.frames_per_scene else random.Random(stable_seed(a.seed,s)).randint(a.warmup,len(v)-n)
        out.append(dict(scene=s,warm=v[start-a.warmup:start],measure=v[start:start+n],layout=LAYOUTS[j%len(LAYOUTS)]))
    if not out:raise ValueError('empty scene list')
    return out


def install(m,a):
    state=dict(active=False,rows=[],mode=None,budget=None,roi_seed=None,layout=None)
    orig=m.extract_img_feat
    from functools import lru_cache
    cached_plan=lru_cache(maxsize=4096)(plan_for_budget)
    def extract(self,img,len_queue=None):
        if not state['active']:return orig(img,len_queue)
        if len_queue is not None or img.ndim!=5 or tuple(img.shape[:3])!=(1,6,3):raise RuntimeError('unexpected img tensor')
        _,nc,c,h,w=img.shape
        if state['budget']==0:
            # Explicit 0% reference: no ResNet/FPN work. Return correctly shaped
            # zero FPN features so the remaining UniAD E2E forward still runs.
            if not getattr(self,'with_img_neck',False) or not hasattr(self.img_neck,'out_channels'):
                raise RuntimeError('0% bypass requires FPN img_neck.out_channels')
            out_channels=int(self.img_neck.out_channels)
            canvas=[img.new_zeros((1,nc,out_channels,int(math.ceil(h/s)),int(math.ceil(w/s)))) for s in STRIDES]
            sig=dict(backbone_bypassed=True,fpn_output=describe_tree(canvas),
                     execution_context=dict(tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                         tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),cudnn_benchmark=bool(torch.backends.cudnn.benchmark)))
            sid=signature_id(sig)
            rows=[dict(camera=cam,image_h=h,image_w=w,batch_size=0,crop_h=0,crop_w=0,
                raw_roi_union_pixels=0,crop_pixels=0,padded_pixels=0,unique_tiles=0,raw_roi_count=0,
                adjacency_edges=0,raw_boxes_json='[]',exec_boxes_json='[]',signature_id=sid,signature=sig,
                cnn_host_ms=0.0,events=None,backbone_skipped=1) for cam in range(nc)]
            state['rows'].extend(rows)
            return canvas
        flat=img.reshape(nc,c,h,w)
        if self.use_grid_mask:flat=self.grid_mask(flat)
        canvas=None;rows=[]
        for cam in range(nc):
            boxes,raw_area,g,plan,summary=cached_plan(h,w,state['budget'],stable_seed(state['roi_seed'],cam),state['layout'],state['mode'],a.budget_axis)
            exec_boxes=plan['exec_boxes']
            if a.budget_axis=='packed' and summary['padded_batch_area_px']>h*w*state['budget']/100:raise RuntimeError('packed budget violated')
            src=_make_padded_camera_image(flat[cam],g['pad_h'],g['pad_w'])
            x,pack=_make_crop_batch(src,exec_boxes)
            if tuple(x.shape)!=(len(exec_boxes),3,summary['max_batch_crop_h'],summary['max_batch_crop_w']):raise RuntimeError('batch shape mismatch')
            if pack['padded_batch_area_px']!=summary['padded_batch_area_px']:raise RuntimeError('packed area mismatch')
            if x.dtype!=torch.float32 or not x.is_cuda or not x.is_contiguous():raise RuntimeError('expected CUDA contiguous FP32')
            # Warmup frames use the same path and geometry. Timing only in measured pilot frames.
            timed=state.get('timed',False)
            if a.nvtx:torch.cuda.nvtx.range_push(f"CNN/{state['mode']}/b{state['budget']}/cam{cam}/{tuple(x.shape)}")
            if timed:e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record();t=time.perf_counter()
            bb=self.img_backbone(x);bb_list=normalize_backbone_outputs(bb);fpn=self.img_neck(bb_list)
            if timed:host=(time.perf_counter()-t)*1000;e1.record()
            if a.nvtx:torch.cuda.nvtx.range_pop()
            feats=normalize_backbone_outputs(fpn)
            if len(feats)!=4 or any(tuple(f.shape[-2:])!=(x.shape[-2]//s,x.shape[-1]//s) for f,s in zip(feats,STRIDES)):
                raise RuntimeError('FPN stride mismatch')
            context=dict(tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                         cudnn_benchmark=bool(torch.backends.cudnn.benchmark),model_training=bool(self.training),autocast_cuda=bool(torch.is_autocast_enabled()))
            sig=dict(backbone_input=describe_tree((x,)),resnet_output=describe_tree(bb),fpn_input=describe_tree((bb_list,)),fpn_output=describe_tree(fpn),execution_context=context)
            if canvas is None:canvas=_init_full_feature_canvases(feats,1,nc,h,w)
            _scatter_crop_features(canvas,feats,exec_boxes,cam,h,w)
            rows.append(dict(camera=cam,image_h=h,image_w=w,batch_size=int(x.shape[0]),crop_h=int(x.shape[2]),crop_w=int(x.shape[3]),
                raw_roi_union_pixels=raw_area,crop_pixels=summary['exec_crop_area_sum_px'],padded_pixels=pack['padded_batch_area_px'],
                unique_tiles=summary['unique_tile_count'],raw_roi_count=len(boxes),adjacency_edges=summary['adjacency_edges'],
                raw_boxes_json=json.dumps(boxes),exec_boxes_json=json.dumps(exec_boxes),signature_id=signature_id(sig),signature=sig,
                cnn_host_ms=host if timed else None,events=(e0,e1) if timed else None,backbone_skipped=0))
        if len(rows)!=6 or canvas is None:raise RuntimeError('expected six camera calls')
        state['rows'].extend(rows)
        return canvas
    m.extract_img_feat=types.MethodType(extract,m)
    return state


def main():
    a=args();out=Path(a.out_dir).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'use fresh output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('run on Thor CUDA')
    seed(a.seed);torch.backends.cuda.matmul.allow_tf32=not a.no_tf32;torch.backends.cudnn.allow_tf32=not a.no_tf32
    cfg=Config.fromfile(a.config)
    if cfg.get('fp16') is not None:raise RuntimeError('no FP16/AMP')
    torch.backends.cudnn.benchmark=bool(cfg.get('cudnn_benchmark',False))
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin') and cfg.get('plugin_dir'):importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/','.'))
    cfg.model.pretrained=None;cfg.model.train_cfg=None;tc=cfg.data.test;tc.test_mode=True
    if int(tc.pop('samples_per_gpu',1))!=1:raise RuntimeError('BS=1 required')
    ds=build_dataset(tc);m=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    ck=load_checkpoint(m,a.checkpoint,map_location='cpu');m.CLASSES=ck.get('meta',{}).get('CLASSES',ds.CLASSES)
    del ck;m.eval().cuda();state=install(m,a);net=MMDataParallel(m,device_ids=[torch.cuda.current_device()])
    manifest=cases(ds,a);dump(out/'scene_manifest.json',manifest)
    sources=[Path(__file__),Path(__file__).with_name('launch_census_utils.py'),ROOT/'tools/test/roi_tile_dispatch_runtime.py',ROOT/'tools/test/profile_roi_tile_dispatch_thor.py',ROOT/'tools/test/cudagraph/shape_census_utils.py']
    cp=Path(a.checkpoint).resolve()
    env=dict(config=str(Path(a.config).resolve()),config_sha=digest(a.config),checkpoint=str(cp),checkpoint_size=cp.stat().st_size,checkpoint_mtime_ns=cp.stat().st_mtime_ns,
             git=command(['git','rev-parse','HEAD']),source_hashes={p.name:digest(p) for p in sources},gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
             cudnn=torch.backends.cudnn.version(),tf32=not a.no_tf32,cudnn_benchmark=bool(torch.backends.cudnn.benchmark),power=command(['nvpmodel','-q']),
             budget_axis=a.budget_axis,budget_definition=('per-camera hard cap on padded CNN input B*H*W relative to H_img*W_img' if a.budget_axis=='packed' else 'fixed nominal raw ROI union percent, quantized to 64px cells'),geometry='synthetic_nested_image_grid_v2',
             planner='dense_fixed_tile_v1',seed=a.seed,dataset_length=len(ds))
    dump(out/'environment.json',env);dump(out/'args.json',vars(a))
    try:(out/'resolved_config.py').write_text(cfg.pretty_text)
    except Exception as exc:(out/'config_dump_error.txt').write_text(str(exc))
    framewriter=Writer(out/'frame_summary.csv');camwriter=Writer(out/'camera_shape_trace.csv')
    frame_rows=[];signatures={};by_scope=defaultdict(Counter);by_frame=defaultdict(list)
    def forward(idx,mode,budget,roi_seed,layout,measure,case_id,rep):
        # Fresh dataset object each time: UniAD mutates CAN bus metadata.
        seed(stable_seed(a.seed,idx,rep));batch=collate([ds[idx]],samples_per_gpu=1);meta=meta0(batch)
        token=str(meta.get('sample_idx',idx));scene=str(meta.get('scene_token',''))
        if not scene or scene!=str(ds.data_infos[idx]['scene_token']):raise RuntimeError('missing or mismatched scene_token')
        state.update(active=True,rows=[],mode=mode,budget=budget,roi_seed=roi_seed,layout=layout,timed=bool(measure and a.stage in ('pilot','sweep')))
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():result=net(return_loss=False,rescale=True,**batch)
        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000;del result,batch
        rows=state['rows'];state['active']=False
        if len(rows)!=6:raise RuntimeError(f'expected six actual ROI calls, got {len(rows)}')
        if not measure:return
        common=dict(case_id=case_id,dataset_index=idx,scene_token=scene,sample_token=token,repeat=rep,mode=mode,budget_pct=budget,layout=layout,roi_seed=roi_seed)
        actual=sum(r['padded_pixels'] for r in rows);den=sum(r['image_h']*r['image_w'] for r in rows)
        t=[];host=[];camera_rows=[]
        for r in rows:
            sig=r.pop('signature');sid=r['signature_id'];ev=r.pop('events')
            ms=float(ev[0].elapsed_time(ev[1])) if ev else (0.0 if budget==0 else None)
            if ev:t.append(ms);host.append(r['cnn_host_ms'])
            elif budget==0:t.append(0.0);host.append(0.0)
            else:raise RuntimeError('missing CUDA timing events for nonzero Backbone budget')
            if sid in signatures and signatures[sid]!=sig:raise RuntimeError('signature hash collision')
            signatures[sid]=sig
            by_scope[budget,mode][sid]+=1
            by_frame[budget,mode].append((case_id,rep,idx,sid))
            camera_rows.append(dict(**common,**r,cnn_stream_ms=ms,raw_area_pct=100*r['raw_roi_union_pixels']/(r['image_h']*r['image_w']),padded_area_pct=100*r['padded_pixels']/(r['image_h']*r['image_w'])))
        f=dict(**common,calls=6,raw_area_pct=100*sum(r['raw_roi_union_pixels'] for r in rows)/den,padded_pixels=actual,padded_area_pct=100*actual/den,
               budget_fill_ratio=(actual/(den*budget/100) if budget>0 else 0.0),
               cnn_stream_ms=sum(t),cnn_host_ms=sum(host),e2e_wall_ms=wall,
               backbone_skipped=int(budget==0),signature_ids=json.dumps([r['signature_id'] for r in rows]))
        frame_rows.append(f);framewriter.write([f]);camwriter.write(camera_rows)
        print(f'[{a.stage}] {case_id} {mode} b={budget} raw={f["raw_area_pct"]:.2f}% packed={f["padded_area_pct"]:.2f}% CNN={f["cnn_stream_ms"]} E2E={wall:.1f}ms',flush=True)
    def run_pilot(bs,modes):
        for case in manifest:
            for idx in case['measure']:
                cid=f'{case["scene"]}:{idx}';rs=stable_seed(a.seed,cid)
                # Same immediate warm history and geometry for every paired budget/mode/repeat.
                seq=case['warm']+case['measure'][:case['measure'].index(idx)]
                for rep in range(a.repeats):
                    arms=[(b,mo) for b in bs for mo in modes];random.Random(stable_seed(rs,rep)).shuffle(arms)
                    for b,mo in arms:
                        reset(m)
                        for wi in seq[-a.warmup:]:forward(wi,mo,b,rs,case['layout'],False,cid,rep)
                        forward(idx,mo,b,rs,case['layout'],True,cid,rep)
    def run_sweep(bs,modes):
        # Same scene/frame/repeat paired across every budget and both modes.
        # Randomized arm order limits monotonic thermal/order bias.
        for case in manifest:
            for idx in case['measure']:
                cid=f'{case["scene"]}:{idx}';rs=stable_seed(a.seed,cid)
                seq=case['warm']+case['measure'][:case['measure'].index(idx)]
                for rep in range(a.repeats):
                    arms=[(b,mo) for b in bs for mo in modes]
                    random.Random(stable_seed(rs,rep)).shuffle(arms)
                    for b,mo in arms:
                        reset(m)
                        for wi in seq[-a.warmup:]:forward(wi,mo,b,rs,case['layout'],False,cid,rep)
                        forward(idx,mo,b,rs,case['layout'],True,cid,rep)

    selection={};report=None
    try:
        if a.stage=='sweep':
            sweep_budgets=a.sweep_budgets
            run_sweep(sweep_budgets,list(MODES))
            expected=sum(len(c['measure']) for c in manifest)*a.repeats*len(sweep_budgets)*len(MODES)
            camera_calls=sum(sum(c.values()) for c in by_scope.values())
            if len(frame_rows)!=expected or camera_calls!=6*expected:
                raise RuntimeError(f'budget sweep completeness failure: frames {len(frame_rows)}/{expected}, camera calls {camera_calls}/{6*expected}')
            summaries=[]
            for mo in MODES:
                for b in sweep_budgets:
                    rr=[r for r in frame_rows if r['mode']==mo and float(r['budget_pct'])==float(b)]
                    if len(rr)!=sum(len(c['measure']) for c in manifest)*a.repeats:
                        raise RuntimeError(f'incomplete mode/budget group: {mo} {b}%')
                    row=dict(mode=mo,budget_pct=b,samples=len(rr),backbone_skipped=int(b==0))
                    for field in ('cnn_stream_ms','cnn_host_ms','e2e_wall_ms','raw_area_pct','padded_area_pct','budget_fill_ratio'):
                        vals=np.asarray([float(x[field]) for x in rr],dtype=np.float64)
                        row[field+'_mean']=float(vals.mean())
                        row[field+'_median']=float(np.median(vals))
                        row[field+'_p95']=float(np.percentile(vals,95))
                        row[field+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0
                    summaries.append(row)
            w=Writer(out/'budget_sweep_summary.csv');w.write(summaries);w.close()
            from plot_budget_sweep import write_budget_plots
            write_budget_plots(out,summaries,sweep_budgets)
            dump(out/'budget_sweep_report.json',dict(
                schema_version=1,budgets_pct=sweep_budgets,modes=list(MODES),
                budget_axis=a.budget_axis,
                budget_definition='per-camera hard cap on padded CNN input B*H*W relative to H_img*W_img',
                scenes=len(manifest),frames_per_scene=a.frames_per_scene,repeats=a.repeats,
                warmup_frames=a.warmup,measured_forward_count=len(frame_rows),camera_backbone_calls=camera_calls,
                primary_metric='cnn_stream_ms: sum of six per-camera CUDA-event intervals around ResNet+FPN',
                secondary_metrics=['cnn_host_ms','e2e_wall_ms','raw_area_pct','padded_area_pct','budget_fill_ratio'],
                zero_budget_semantics='ResNet/FPN bypass; zero FPN feature tensors returned; 0% is a no-Backbone reference, not a positive CNN workload',
                interpretation='Synthetic nested image-space ROI geometry; real UniAD E2E forward and actual ResNet/FPN for nonzero budgets.'
            ))
        elif a.stage=='pilot':
            if a.pilot_budgets is not None:
                pilot_budgets=a.pilot_budgets
                pilot_modes=a.pilot_modes
                run_pilot(pilot_budgets,pilot_modes)
                tested={mo:list(pilot_budgets) for mo in pilot_modes}
            else:
                pilot_budgets=[10.0,20.0,40.0]
                pilot_modes=list(MODES)
                run_pilot(pilot_budgets,pilot_modes)
                tested={mo:list(pilot_budgets) for mo in pilot_modes}
                # Extend only paths that remain flat through the highest tested tier.
                for b in (60.0,80.0,100.0):
                    if b>a.max_pilot_budget:break
                    active=[mo for mo in pilot_modes if decide(frame_rows,mo,tested[mo],a.flat,a.rise,a.min_area_gain)['status']=='plateau_continues']
                    if not active:break
                    run_pilot((b,),active)
                    for mo in active:tested[mo].append(b)
            decisions={mo:decide(frame_rows,mo,tested[mo],a.flat,a.rise,a.min_area_gain) for mo in tested}
            for mo,d in decisions.items():d['tested_budgets']=tested[mo]
            report=dict(schema_version=2,environment=env,thresholds=dict(flat=a.flat,rise=a.rise,min_area_gain=a.min_area_gain),
                        pilot_modes=pilot_modes,decisions=decisions,pilot_cases=sum(len(c['measure']) for c in manifest),
                        interpretation='Empirical latency plateau candidate, not proof of CPU launch bound; verify with Nsight Systems.')
            dump(out/'knee_report.json',report)
            summaries=[]
            for (b,mo),counts in sorted(by_scope.items()):
                rr=[r for r in frame_rows if r['mode']==mo and r['budget_pct']==b]
                summaries.append(dict(budget_pct=b,mode=mo,frames=len(rr),median_raw_area_pct=median(r['raw_area_pct'] for r in rr),median_padded_area_pct=median(r['padded_area_pct'] for r in rr),median_budget_fill_ratio=median(r['budget_fill_ratio'] for r in rr),median_cnn_stream_ms=median(r['cnn_stream_ms'] for r in rr),median_cnn_host_ms=median(r['cnn_host_ms'] for r in rr)))
            w=Writer(out/'pilot_summary.csv');w.write(summaries);w.close()
        else:
            if a.mode_budgets:
                selection=a.mode_budgets
            elif a.pilot_report:
                report=json.loads(Path(a.pilot_report).read_text())
                if report.get('schema_version') not in (1,2) or report['environment']!=env:raise ValueError('pilot environment/source/config/weight/settings differ; rerun matching pilot or use explicit --mode-budgets')
                if report['pilot_cases']<3 and not a.force_budgets:raise ValueError('smoke pilot cannot select final census budgets')
                selection={mo:report['decisions'].get(mo,{}).get('selected_budgets',[]) for mo in MODES}
            if a.force_budgets:selection={mo:a.force_budgets for mo in MODES}
            if not any(selection.values()):raise RuntimeError('no plateau verified at fixed tiers; inspect pilot or explicitly specify exploratory --mode-budgets')
            dump(out/'census_selection.json',dict(selection=selection,forced=bool(a.force_budgets or a.mode_budgets),pilot_report=a.pilot_report))
            # No high-budget sweep. Each selected condition gets a real sequential E2E history.
            for mo,bs in selection.items():
                for b in bs:
                    for case in manifest:
                        reset(m)
                        for wi in case['warm']:forward(wi,mo,b,stable_seed(a.seed,case['scene'],wi),case['layout'],False,case['scene'],0)
                        for idx in case['measure']:
                            cid=f'{case["scene"]}:{idx}';forward(idx,mo,b,stable_seed(a.seed,cid),case['layout'],True,cid,0)
            expected=sum(len(c['measure']) for c in manifest)*sum(map(len,selection.values()))
            if len(frame_rows)!=expected or sum(sum(c.values()) for c in by_scope.values())!=6*expected:raise RuntimeError('census completeness failure')
            total=Counter();hist=[];scopes=[];full_frame=[]
            for (b,mo),cnt in sorted(by_scope.items()):
                total.update(cnt);n=sum(cnt.values());rank=sorted(cnt,key=lambda s:(-cnt[s],s))
                scopes.append(dict(budget_pct=b,mode=mo,calls=n,unique_signatures=len(cnt),top_k=coverage(cnt)))
                groups=defaultdict(set)
                for cid,rep,idx,sid in by_frame[b,mo]:groups[(cid,rep,idx)].add(sid)
                for k in (1,3,5,10,20,50):
                    allowed=set(rank[:k]);full_frame.append(dict(budget_pct=b,mode=mo,k=k,all_calls_covered=sum(ids<=allowed for ids in groups.values())/len(groups)))
                for sid in rank:hist.append(dict(budget_pct=b,mode=mo,signature_id=sid,count=cnt[sid],frequency=cnt[sid]/n))
            w=Writer(out/'shape_histogram_by_budget_mode.csv');w.write(hist);w.close()
            w=Writer(out/'full_frame_topk.csv');w.write(full_frame);w.close()
            ranked=sorted(total,key=lambda s:(-total[s],s));n=sum(total.values())
            global_rows=[dict(signature_id=s,count=total[s],frequency=total[s]/n,signature=signatures[s],scope_counts={f'{b}:{mo}':cnt[s] for (b,mo),cnt in by_scope.items() if cnt[s]}) for s in ranked]
            w=Writer(out/'shape_histogram_global.csv');w.write([dict(signature_id=r['signature_id'],count=r['count'],frequency=r['frequency'],scope_counts_json=json.dumps(r['scope_counts'],sort_keys=True)) for r in global_rows]);w.close()
            dump(out/'graph_signature_manifest.json',dict(schema_version=2,environment=env,selection=selection,signatures=global_rows,scope='actual_e2e synthetic_image_roi; counts unweighted across enumerated budgets/modes'))
            dump(out/'census_report.json',dict(frames=len(frame_rows),calls=n,unique_signatures=len(total),scopes=scopes,global_top_k=coverage(total),dataset_length=len(ds),full_dataset_after_warmup=(a.scenes==0 and a.frames_per_scene==0),synthetic_roi=True))
        (out/'COMPLETE').write_text('ok\n')
    finally:framewriter.close();camwriter.close()
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
