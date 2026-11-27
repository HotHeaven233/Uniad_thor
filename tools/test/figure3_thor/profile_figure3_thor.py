#!/usr/bin/env python3
"""Figure 3, journal extension: real UniAD E2E, native eager TF32.
Runtime-only instrumentation. No internal synchronize. Area probe is synthetic
crop geometry on real images, not QC-BEV accuracy or scheduler evaluation.
"""
from __future__ import annotations
import argparse, csv, functools, importlib, importlib.util, json, os, random, subprocess, sys, time
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'projects/configs').is_dir()),None)
if ROOT is None: raise RuntimeError('Install under UniAD tools/test/figure3_thor')
sys.path.insert(0,str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
import cv2
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config'); p.add_argument('checkpoint'); p.add_argument('--out-dir',required=True)
    p.add_argument('--mode',choices=['full','original-qc'],default='full')
    p.add_argument('--adapter',help='Optional Python file defining install(model,cfg); original QC only')
    p.add_argument('--scheduler-method',help='Dotted bound method on model, mandatory in original-qc (e.g. actual scheduler wrapper)')
    p.add_argument('--profile',choices=['summary','detail','plain','area'],default='summary')
    p.add_argument('--scenes',type=int,default=12);p.add_argument('--frames-per-scene',type=int,default=8)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeats',type=int,default=2);p.add_argument('--seed',type=int,default=20261127)
    p.add_argument('--manifest');p.add_argument('--trace',action='store_true',help='CUDA profiler API start/stop for first measured forward only; NVTX')
    p.add_argument('--shapes',default='64x64,128x128,192x192,256x256,384x384,512x512,768x768,896x1536')
    a=p.parse_args()
    if a.scenes<1 or a.frames_per_scene<0 or a.warmup<1 or a.repeats<1: p.error('invalid sampling settings')
    if a.mode=='original-qc' and not a.scheduler_method:p.error('original-qc requires an actual scheduler method; no fake baseline')
    if a.mode=='full' and (a.adapter or a.scheduler_method):p.error('full must be native, no adapter/scheduler')
    if a.profile=='area' and a.mode!='full':p.error('area only supports native full + diagnostic crop patch')
    if a.trace and a.profile=='plain':p.error('trace requires instrumentation')
    if a.trace and (a.scenes!=1 or a.frames_per_scene!=1 or a.repeats!=1 or a.profile=='area'):p.error('trace requires 1 scene, 1 frame, 1 repeat, no area')
    return a

def seed(s):
    random.seed(s); np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def reset(m):
    m.prev_frame_info={'prev_bev':None,'scene_token':None,'prev_pos':0,'prev_angle':0}
    for n in ('prev_bev','scene_token','timestamp','l2g_r_mat','l2g_t','test_track_instances'):
        if hasattr(m,n):setattr(m,n,None)
    # External QC adapters can clear their own state; no guessing cache names.
    if hasattr(m,'figure3_reset'):m.figure3_reset()

def meta0(d):
    x=d.get('img_metas')
    while True:
        if hasattr(x,'data'):x=x.data
        elif isinstance(x,(tuple,list)) and x:x=x[0]
        else:return x if isinstance(x,dict) else {}

def command(cmd):
    try:return subprocess.check_output(cmd,stderr=subprocess.STDOUT,text=True,timeout=10).strip()
    except Exception as e:return f'unavailable: {e}'

class Writer:
    def __init__(self,path):self.f=open(path,'w',newline='',encoding='utf-8');self.w=None
    def write(self,rows):
        for r in rows:
            if self.w is None:self.w=csv.DictWriter(self.f,fieldnames=list(r));self.w.writeheader()
            self.w.writerow(r)
        self.f.flush()
    def close(self):self.f.close()

class Recorder:
    def __init__(self):self.active=False;self.nvtx=False;self.rows=[];self.stack=[];self.shape=None
    def begin(self,nvtx=False):
        self.rows=[];self.stack=[];self.nvtx=nvtx;self.active=True
        self.origin=torch.cuda.Event(enable_timing=True);self.origin.record();self.host_origin=time.perf_counter()
    @contextmanager
    def span(self,name,**attrs):
        if not self.active:
            yield;return
        i=len(self.rows); parent=self.stack[-1] if self.stack else -1
        ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
        stream=int(torch.cuda.current_stream().cuda_stream)
        row=dict(id=i,parent=parent,name=name,stream=stream,attrs=attrs,e0=ev0,e1=ev1)
        self.rows.append(row);self.stack.append(i)
        if self.nvtx:torch.cuda.nvtx.range_push('fig3/'+name)
        ev0.record();row['h0']=(time.perf_counter()-self.host_origin)*1000
        try:yield
        finally:
            row['h1']=(time.perf_counter()-self.host_origin)*1000
            if int(torch.cuda.current_stream().cuda_stream)!=stream:raise RuntimeError('scope changed current CUDA stream')
            ev1.record();self.stack.pop()
            if self.nvtx:torch.cuda.nvtx.range_pop()
    def finish(self):
        self.active=False
        if self.stack:raise RuntimeError('unbalanced scopes')
        return [dict(scope_id=r['id'],parent_id=r['parent'],module=r['name'],stream=r['stream'],
                     host_start_ms=r['h0'],host_end_ms=r['h1'],gpu_start_ms=self.origin.elapsed_time(r['e0']),
                     gpu_end_ms=self.origin.elapsed_time(r['e1']),attrs=json.dumps(r['attrs'],sort_keys=True)) for r in self.rows]
    def wrap(self,obj,method,name,shape=False):
        old=getattr(obj,method,None)
        if not callable(old):raise RuntimeError(f'missing callable {type(obj).__name__}.{method}')
        @functools.wraps(old)
        def wrapped(*a,**kw):
            if not self.active:return old(*a,**kw)
            attrs={}
            if shape:
                q=a[0] if a else kw.get('query')
                if torch.is_tensor(q):attrs['input_shape']=list(q.shape)
            with self.span(name,**attrs):return old(*a,**kw)
        setattr(obj,method,wrapped)

def dotted(obj,s):
    parts=s.split('.')
    for k in parts[:-1]:obj=getattr(obj,k)
    return obj,parts[-1]

def instrument(m,r,a):
    r.wrap(m,'simple_test_track','tracking')
    r.wrap(m,'get_bevs','bev_other')
    r.wrap(m,'extract_img_feat','backbone')
    r.wrap(m.pts_bbox_head,'get_bev_features','encoder')
    for attr,label in [('seg_head','mapping'),('motion_head','motion'),('occ_head','occupancy'),('planning_head','planning')]:
        r.wrap(getattr(m,attr),'forward_test',label)
    if a.scheduler_method:
        obj,method=dotted(m,a.scheduler_method);r.wrap(obj,method,'scheduler')
    if a.profile=='detail':
        r.wrap(m.img_backbone,'forward','resnet',True);r.wrap(m.img_neck,'forward','fpn')
        enc=m.pts_bbox_head.transformer.encoder
        r.wrap(enc,'forward','encoder_core')
        if hasattr(enc,'point_sampling'):r.wrap(enc,'point_sampling','point_sampling')
        for j,l in enumerate(enc.layers):
            r.wrap(l,'forward',f'layer_{j}')
            for k,name in enumerate(('tsa','sca')):
                att=l.attentions[k];r.wrap(att,'forward',name,True)
                if hasattr(att,'forward_selective'):r.wrap(att,'forward_selective',name+'_selective',True)
            for f in l.ffns:r.wrap(f,'forward','ffn',True)
            for norm in l.norms:r.wrap(norm,'forward','norm',True)

def install_area(m,r):
    native=m.extract_img_feat
    def extract(img,len_queue=None):
        if r.shape is None:return native(img,len_queue)
        if len_queue is not None or img.ndim!=5 or img.shape[:2]!=(1,6):raise RuntimeError('unexpected image batch')
        _,nc,c,H,W=img.shape;h,w=r.shape
        if h>H or w>W or h%64 or w%64:raise ValueError(f'invalid crop {h,w} for {H,W}')
        y=((H-h)//2//64)*64;x=((W-w)//2//64)*64
        flat=img.reshape(nc,c,H,W)
        if m.use_grid_mask:flat=m.grid_mask(flat)
        canvas=None
        for cam in range(nc):
            crop=flat[cam:cam+1,:,y:y+h,x:x+w].contiguous()
            with r.span('area_cnn',camera=cam,batch=1,height=h,width=w,pixels=h*w):
                feat=m.img_backbone(crop)
                if isinstance(feat,dict):feat=list(feat.values())
                feat=m.img_neck(feat)
            if len(feat)!=4:raise RuntimeError('expected FPN strides 8,16,32,64')
            if canvas is None:canvas=[f.new_zeros((1,nc,f.shape[1],(H+s-1)//s,(W+s-1)//s)) for f,s in zip(feat,(8,16,32,64))]
            for f,d,s in zip(feat,canvas,(8,16,32,64)):
                yy,xx=y//s,x//s;hh,ww=f.shape[-2:]
                if yy+hh>d.shape[-2] or xx+ww>d.shape[-1]:raise RuntimeError('crop scatter outside feature canvas')
                d[:,cam,:,yy:yy+hh,xx:xx+ww]=f
        return canvas
    m.extract_img_feat=extract

def make_manifest(ds,a):
    groups=defaultdict(list)
    for i,d in enumerate(ds.data_infos):groups[str(d['scene_token'])].append(i)
    for v in groups.values():v.sort(key=lambda i:ds.data_infos[i]['timestamp'])
    chosen=sorted(s for s,v in groups.items() if len(v)>a.warmup)
    random.Random(a.seed).shuffle(chosen)
    if a.scenes>len(chosen):raise ValueError('too many scenes')
    cases=[]
    for s in chosen[:a.scenes]:
        v=groups[s]; end=len(v) if a.frames_per_scene==0 else min(len(v),a.warmup+a.frames_per_scene)
        cases.append(dict(scene=s,warmup=v[:a.warmup],measure=v[a.warmup:end],tokens=[str(ds.data_infos[i].get('token',i)) for i in v[:end]]))
    return dict(seed=a.seed,cases=cases)

def check_manifest(ds,manifest):
    for c in manifest['cases']:
        ids=c['warmup']+c['measure'];assert ids and c['measure'] and c['warmup']
        if len(ids)!=len(set(ids)) or [str(ds.data_infos[i].get('token',i)) for i in ids]!=c['tokens']:raise RuntimeError('manifest sample mismatch')
        if any(str(ds.data_infos[i]['scene_token'])!=c['scene'] for i in ids):raise RuntimeError('manifest scene mismatch')
        t=[ds.data_infos[i]['timestamp'] for i in ids]
        if t!=sorted(t) or len(t)!=len(set(t)):raise RuntimeError('manifest timestamp order invalid')

def main():
    a=cli(); out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'frames.csv').exists():raise FileExistsError('use fresh output directory')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required on Thor')
    seed(a.seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    cfg=Config.fromfile(a.config)
    if cfg.get('fp16',None) is not None:raise RuntimeError('FP16 disallowed')
    torch.backends.cudnn.benchmark=bool(cfg.get('cudnn_benchmark',False))
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin') and cfg.get('plugin_dir'):importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/','.'))
    cfg.model.pretrained=None;cfg.model.train_cfg=None;cfg.data.test.test_mode=True
    if int(cfg.data.test.pop('samples_per_gpu',1))!=1:raise RuntimeError('expected batch size 1')
    ds=build_dataset(cfg.data.test)
    manifest=json.loads(Path(a.manifest).read_text()) if a.manifest else make_manifest(ds,a);check_manifest(ds,manifest)
    if a.trace and (len(manifest['cases'])!=1 or len(manifest['cases'][0]['measure'])!=1):raise RuntimeError('trace manifest must have 1 measured frame')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=dict(args=vars(a),gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),tf32=True,cudnn_benchmark=torch.backends.cudnn.benchmark,git=command(['git','rev-parse','HEAD']),git_status=command(['git','status','--short']),power_mode=command(['nvpmodel','-q']),jetson_clocks=command(['jetson_clocks','--show']),uname=command(['uname','-a']))
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    try:(out/'resolved_config.py').write_text(cfg.pretty_text)
    except Exception as e:(out/'resolved_config.txt').write_text(repr(cfg._cfg_dict)+'\n'+str(e))
    m=build_model(cfg.model,test_cfg=cfg.get('test_cfg'));ck=load_checkpoint(m,a.checkpoint,map_location='cpu');m.CLASSES=ck.get('meta',{}).get('CLASSES',ds.CLASSES);m.eval().cuda()
    if a.adapter:
        spec=importlib.util.spec_from_file_location('figure3_adapter',a.adapter);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install(m,cfg)
    r=Recorder()
    if a.profile=='area':install_area(m,r)
    if a.profile!='plain':instrument(m,r,a)
    net=MMDataParallel(m,device_ids=[torch.cuda.current_device()]);fw=Writer(out/'frames.csv');sw=Writer(out/'scopes.csv')
    def run(idx,case,rep,measured,shape=None,trace=False):
        # Fresh dataset object each call; native model mutates metadata.
        seed((a.seed+idx)%2**32);t=time.perf_counter();d=collate([ds[idx]],samples_per_gpu=1);load=(time.perf_counter()-t)*1000
        meta=meta0(d);sample=str(meta.get('sample_idx',ds.data_infos[idx].get('token',idx)))
        if meta.get('scene_token') and str(meta['scene_token'])!=case['scene']:raise RuntimeError('wrong scene')
        r.shape=shape;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if trace:torch.cuda.profiler.start()
        use=measured and a.profile!='plain'
        if use:r.begin(trace)
        t=time.perf_counter()
        with torch.no_grad():res=net(return_loss=False,rescale=True,**d)
        host=(time.perf_counter()-t)*1000;torch.cuda.synchronize();wall=(time.perf_counter()-t)*1000
        rows=r.finish() if use else []
        if trace:torch.cuda.profiler.stop()
        if not measured:return
        key=f'{case["scene"]}:{idx}:{rep}:{shape}'
        common=dict(frame_id=key,mode=a.mode,profile=a.profile,scene=case['scene'],dataset_index=idx,sample=sample,repeat=rep,shape='' if shape is None else f'{shape[0]}x{shape[1]}')
        if use:
            counts=Counter(x['module'] for x in rows)
            for n in ('tracking','bev_other','backbone','encoder','mapping','motion','occupancy','planning'):
                if counts[n]!=1:raise RuntimeError(f'{n} called {counts[n]} times, verify integration')
            if a.mode=='original-qc' and not counts['scheduler']:raise RuntimeError('QC scheduler not executed; cannot label original-qc')
            if shape and counts['area_cnn']!=6:raise RuntimeError('expected six actual crop calls')
            if a.profile=='detail' and a.mode=='full':
                L=len(m.pts_bbox_head.transformer.encoder.layers)
                for n,k in [('tsa',L),('sca',L),('ffn',L),('norm',3*L)]:
                    if counts[n]!=k:raise RuntimeError(f'detail scope {n}: expected {k}, got {counts[n]}')
        fw.write([dict(**common,data_load_ms=load,forward_host_ms=host,e2e_ms=wall,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)])
        sw.write([dict(**common,**x) for x in rows]);print(f'[FIG3] {a.mode}/{a.profile} idx={idx} rep={rep} shape={shape} E2E={wall:.2f} ms',flush=True)
    try:
        for rep in range(a.repeats):
            for c in manifest['cases']:
                if a.profile=='area':
                    shapes=[tuple(map(int,x.split('x'))) for x in a.shapes.split(',')];random.Random(a.seed+rep+int(c['measure'][0])).shuffle(shapes)
                    for shape in shapes:
                        # Same warm history and first measured frame for each shape.
                        reset(m)
                        for idx in c['warmup']:run(idx,c,rep,False)
                        run(c['measure'][0],c,rep,True,shape)
                else:
                    reset(m)
                    for idx in c['warmup']:run(idx,c,rep,False)
                    for idx in c['measure']:run(idx,c,rep,True,trace=a.trace)
    finally:fw.close();sw.close()
    (out/'COMPLETE').write_text('ok\n');print('[FIG3] COMPLETE',out,flush=True)
if __name__=='__main__':main()
