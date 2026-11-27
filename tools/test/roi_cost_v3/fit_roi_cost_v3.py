#!/usr/bin/env python3
"""Fit standardized ridge formulas on training scenes; calibrate decision margin on separate scenes."""
from __future__ import annotations
import argparse,csv,json,math,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from adaptive_roi_dispatcher import A,B,D,SHAPE,COMPONENTS,features,predict_exec,predict_component,shape_vector,choose

KEY=('workload_id','repeat','camera')
FKEY=('workload_id','repeat')
def read_csv(p):
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    if not rows:return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def val(r,k):return float(r[k])
def target(r,k):
    return sum(val(r,c) for c in {'cnn':['resnet_gpu_ms','fpn_gpu_ms'],'pack':['pad_gpu_ms','crop_pack_gpu_ms'],'scatter':['scatter_gpu_ms']}[k])
def actual(r):return sum(target(r,k) for k in COMPONENTS)
def ridge(xs,ys,alpha=1.):
    X=np.asarray(xs,dtype=float);y=np.asarray(ys,dtype=float)
    if X.ndim!=2 or len(y)<2 or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('invalid regression data')
    mean=X.mean(axis=0);scale=X.std(axis=0);scale[scale<1e-12]=1.;z=(X-mean)/scale
    coef=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-y.mean()))
    return dict(intercept=float(y.mean()),mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),alpha=alpha)
def predict(m,x):return max(0.,m['intercept']+sum(c*(v-u)/s for c,v,u,s in zip(m['coef'],x,m['mean'],m['scale'])))
def metrics(y,p):
    a=np.asarray(y,float);b=np.asarray(p,float);d=np.abs(a-b)
    return dict(n=len(a),mae_ms=float(d.mean()),rmse_ms=float(np.sqrt(np.mean(d*d))),p90_abs_ms=float(np.quantile(d,.9)),p90_relative_pct=float(np.quantile(d/np.maximum(a,.001),.9)*100),bias_ms=float((b-a).mean()))
def paired(rows,keys,arms=(A,B)):
    groups=defaultdict(dict)
    for r in rows:
        if r['arm'] not in arms:continue
        k=tuple(r[x] for x in keys)
        if r['arm'] in groups[k]:raise ValueError(f'duplicate row {k} {r["arm"]}')
        groups[k][r['arm']]=r
    for k,g in groups.items():
        if set(g)!=set(arms):raise ValueError(f'incomplete pair {k}')
        if len({(r['scene_token'],r['sample_idx'],r['roi_seed']) for r in g.values()})!=1:raise ValueError(f'nonidentical pair {k}')
    return list(groups.values())
def validate(folder,stage):
    if not (folder/'COMPLETE').is_file():raise ValueError('incomplete experiment, missing COMPLETE')
    man=json.loads((folder/'manifest.json').read_text());env=json.loads((folder/'environment.json').read_text())
    if env['stage']!=stage:raise ValueError('wrong experiment stage')
    cams=read_csv(folder/'roi_v3_camera.csv');frames=read_csv(folder/'roi_v3_frame.csv');reps=env['repeats']
    cases={c['workload_id']:c for c in man['cases'] if stage=='collect' or (c['split']=='test' and 'shape' not in c)}
    expected={(k,str(rep),arm) for k,c in cases.items() for rep in range(reps) for arm in ([SHAPE] if 'shape' in c else ([A,B,D] if stage=='online' else [A,B]))}
    fkeys=[tuple(r[x] for x in (*FKEY,'arm')) for r in frames];ckeys=[tuple(r[x] for x in (*FKEY,'arm','camera')) for r in cams]
    if len(fkeys)!=len(set(fkeys)) or set(fkeys)!=expected:raise ValueError('missing/duplicate frame records')
    if len(ckeys)!=len(set(ckeys)) or set(ckeys)!={(*k,str(c)) for k in expected for c in range(6)}:raise ValueError('missing/duplicate camera records')
    for r in cams+frames:
        c=cases[r['workload_id']]
        if any(str(c[k])!=r[k] for k in ('split','scene_token','dataset_index','roi_seed')):raise ValueError('manifest mismatch')
    for r in cams:
        if not all(math.isfinite(target(r,k)) and target(r,k)>=0 for k in COMPONENTS):raise ValueError('invalid GPU timing')
    for r in frames:
        if not all(math.isfinite(val(r,k)) and val(r,k)>0 for k in ('e2e_forward_ms','roi_block_gpu_stream_ms')):raise ValueError('invalid frame timing')
    return cams,frames,man

def planner_features(r):
    # All features obtainable after building the geometry. Not a pre-search gate.
    n=val(r,'raw_roi_count');k=val(r,'unique_tile_count');s=val(r,'candidate_count');j=val(r,'merge_pair_checks')
    return [n,n*n,k,s,k*s,j, float(r['arm']==B)]
def frame_features(rows,model,measured=False):
    ex=sum(predict_exec(model,r) for r in rows)
    pl=sum(val(r,'planner_wall_cpu_ms') if measured else predict(model['planner'],planner_features(r)) for r in rows)
    return [ex,pl,max(predict_exec(model,r) for r in rows),sum(val(r,'exec_crop_count') for r in rows)]
def fit(folder,out,alpha):
    cams,frames,man=validate(folder,'collect');out.mkdir(parents=True,exist_ok=True)
    scenes={s:sorted({r['scene_token'] for r in cams if r['split']==s}) for s in ('train','calibration','test')}
    if any(not v for v in scenes.values()) or sum(map(len,scenes.values()))!=len(set(sum(scenes.values(),[]))):raise ValueError('scene split leakage or empty split')
    train=[r for r in cams if r['split']=='train'];normal=[r for r in train if r['arm']!=SHAPE]
    model={'schema_version':3,'manifest_sha256':hashlib.sha256(json.dumps(man,sort_keys=True).encode()).hexdigest(),'settings':man['settings'],'train_scenes':scenes['train'],'calibration_scenes':scenes['calibration'],'test_scenes':scenes['test'],
           'components':{k:ridge([features(r,k) for r in train],[target(r,k) for r in train],alpha) for k in COMPONENTS},
           'shape_min':np.min([shape_vector(r) for r in train],axis=0).tolist(),'shape_max':np.max([shape_vector(r) for r in train],axis=0).tolist(),
           'planner':ridge([planner_features(r) for r in normal],[val(r,'planner_wall_cpu_ms') for r in normal],alpha),
           'margin_ms':0.}
    # Area-only baseline for ablation; train only.
    area=ridge([[features(r,'cnn')[0]] for r in train],[target(r,'cnn') for r in train],alpha)
    cal=[r for r in cams if r['split']=='calibration' and r['arm'] in (A,B)]
    residual=[]
    for g in paired(cal,KEY):
        a,b=g[A],g[B];residual.append((predict_exec(model,a)-predict_exec(model,b))-(actual(a)-actual(b)))
    # One-sided overestimation quantile, heuristic not an i.i.d. confidence guarantee.
    model['margin_ms']=max(0.,float(np.quantile(residual,.9)))
    groups=defaultdict(list)
    for r in cams:
        if r['arm'] in (A,B):groups[(r['workload_id'],r['repeat'],r['arm'])].append(r)
    training_frames=[r for r in frames if r['split']=='train' and r['arm'] in (A,B)]
    model['frame_models']={k:ridge([frame_features(groups[tuple(r[x] for x in (*FKEY,'arm'))],model) for r in training_frames],[val(r,k) for r in training_frames],alpha) for k in ('roi_block_gpu_stream_ms','e2e_forward_ms')}
    report={'scene_counts':{k:len(v) for k,v in scenes.items()},'margin_ms':model['margin_ms'],'split_metrics':{},'notes':['GPU event stage sum is an execution proxy, not CPU+GPU wall time.','Frame formula includes estimated planning; online decision uses remaining execution costs only.','Camera-level oracle is counterfactual; validate true mixed execution with online stage.','Shape support is a bounding box, not a memory guarantee.']}
    detail=[]
    for split in ('calibration','test'):
        rows=[r for r in cams if r['split']==split and r['arm'] in (A,B)]
        m={k:metrics([target(r,k) for r in rows],[predict_component(model,r,k) for r in rows]) for k in COMPONENTS}
        m['cnn_area_only']=metrics([target(r,'cnn') for r in rows],[predict(area,[features(r,'cnn')[0]]) for r in rows]);m['execution_proxy']=metrics([actual(r) for r in rows],[predict_exec(model,r) for r in rows]);m['planner']=metrics([val(r,'planner_wall_cpu_ms') for r in rows],[predict(model['planner'],planner_features(r)) for r in rows])
        regret=[];fixed=[];delta_y=[];delta_p=[];wins=[];supported=[]
        for g in paired(rows,KEY):
            a,b=g[A],g[B];choice,pa,pb,ok=choose(model,a,b);ta,tb=actual(a),actual(b);t=ta if choice==A else tb
            regret.append(t-min(ta,tb));fixed.append(ta-min(ta,tb));delta_y.append(ta-tb);delta_p.append(pa-pb);wins.append(t<=min(ta,tb)+1e-12);supported.append(ok)
            detail.append(dict(split=split,scene_token=a['scene_token'],workload_id=a['workload_id'],repeat=a['repeat'],camera=a['camera'],actual_a_ms=ta,actual_b_ms=tb,pred_a_ms=pa,pred_b_ms=pb,choice=choice,supported=ok,regret_ms=regret[-1]))
        m['decision']=dict(accuracy=float(np.mean(wins)),mean_regret_ms=float(np.mean(regret)),p90_regret_ms=float(np.quantile(regret,.9)),fixed_a_mean_regret_ms=float(np.mean(fixed)),delta_mae_ms=float(np.mean(np.abs(np.array(delta_y)-delta_p))),support_rate=float(np.mean(supported)))
        fs=[r for r in frames if r['split']==split and r['arm'] in (A,B)]
        for k,fm in model['frame_models'].items():m[k]=metrics([val(r,k) for r in fs],[predict(fm,frame_features(groups[tuple(r[x] for x in (*FKEY,'arm'))],model)) for r in fs])
        report['split_metrics'][split]=m
    (out/'roi_cost_model.json').write_text(json.dumps(model,indent=2,allow_nan=False));(out/'fit_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));write_csv(out/'camera_predictions.csv',detail)
    print(json.dumps(report,indent=2));return model

def online_report(folder,out):
    cams,frames,man=validate(folder,'online');out.mkdir(parents=True,exist_ok=True)
    pairs=paired(frames,FKEY,(A,B,D));rows=[]
    for g in pairs:
        a,b,d=[g[k] for k in (A,B,D)];r={k:a[k] for k in (*FKEY,'scene_token','sample_idx')}
        for k in ('e2e_forward_ms','roi_block_gpu_stream_ms','planner_wall_cpu_ms_sum','prediction_cpu_ms_sum','cuda_peak_allocated_mb'):
            for arm,rec in zip(('a','b','adaptive'),(a,b,d)):r[f'{arm}_{k}']=val(rec,k)
        r['adaptive_adjacency_cameras']=val(d,'adjacency_cameras');rows.append(r)
    rng=np.random.default_rng(12345);report={'paired_frames':len(rows),'scenes':len({r['scene_token'] for r in rows}),'metrics':{}}
    for k in ('e2e_forward_ms','roi_block_gpu_stream_ms'):
        a=np.array([r['a_'+k] for r in rows]);b=np.array([r['b_'+k] for r in rows]);d=np.array([r['adaptive_'+k] for r in rows]);delta=a-d
        # Cluster bootstrap at scene level; each scene is weighted equally.
        groups=defaultdict(list)
        for r,x in zip(rows,delta):groups[r['scene_token']].append(x)
        z=np.array([np.mean(x) for x in groups.values()]);boot=rng.choice(z,size=(5000,len(z)),replace=True).mean(axis=1)
        report['metrics'][k]={'a_mean_ms':float(a.mean()),'b_mean_ms':float(b.mean()),'adaptive_mean_ms':float(d.mean()),'a_p90_ms':float(np.quantile(a,.9)),'b_p90_ms':float(np.quantile(b,.9)),'adaptive_p90_ms':float(np.quantile(d,.9)),'adaptive_win_rate_vs_a':float(np.mean(d<a)), 'a_minus_adaptive_scene_equal_mean_ms':float(z.mean()),'scene_bootstrap_95pct_ci_ms':np.quantile(boot,[.025,.975]).tolist(),'adaptive_minus_best_fixed_mean_ms':float(np.mean(d-np.minimum(a,b)))}
    report['planner_mean_ms']={k:float(np.mean([r[k+'_planner_wall_cpu_ms_sum'] for r in rows])) for k in ('a','b','adaptive')};report['adaptive_prediction_mean_ms']=float(np.mean([r['adaptive_prediction_cpu_ms_sum'] for r in rows]));report['adaptive_adj_camera_fraction']=float(np.mean([r['adaptive_adjacency_cameras']/6 for r in rows]));report['peak_allocated_mb_max']={k:max(r[k+'_cuda_peak_allocated_mb'] for r in rows) for k in ('a','b','adaptive')}
    (out/'online_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));write_csv(out/'online_paired.csv',rows);print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--stage',choices=['fit','online-report'],default='fit');p.add_argument('--alpha',type=float,default=1.)
    a=p.parse_args()
    if a.alpha<=0:raise ValueError('alpha must be positive')
    if a.stage=='fit':fit(Path(a.input_dir),Path(a.out_dir),a.alpha)
    else:online_report(Path(a.input_dir),Path(a.out_dir))
if __name__=='__main__':main()
