#!/usr/bin/env python3
"""Fit Figure 9 linear Model A (positive partial-query branch only)."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;den=((y-y.mean())**2).sum()
    return dict(n=len(y),mae_ms=float(np.abs(e).mean()),rmse_ms=float(np.sqrt((e*e).mean())),r2=float(1-(e*e).sum()/den) if den else None,p90_absolute_error_ms=float(np.quantile(np.abs(e),.9)),p90_relative_error_pct=float(100*np.quantile(np.abs(e)/np.maximum(y,1e-9),.9)),bias_ms=float(e.mean()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',required=True);p.add_argument('--out-dir');args=p.parse_args();src=Path(args.input_dir);out=Path(args.out_dir or src/'fit');out.mkdir(parents=True,exist_ok=True)
    if not (src/'COMPLETE').is_file():raise RuntimeError('Collection incomplete')
    m=json.loads((src/'manifest.json').read_text());rows=list(csv.DictReader((src/'encoder_frame_raw.csv').open()))
    expected={(str(s),r,int(q)) for s,_,_,_ in m['pairs'] for r in range(m['repeats']) for q in m['counts']}
    got={(r['scene_token'],int(r['repeat']),int(r['actual_queries'])) for r in rows}
    if got!=expected or len(rows)!=len(expected):raise RuntimeError('missing/duplicate scene-repeat-query rows')
    if any(int(r['requested_queries'])!=int(r['actual_queries']) or int(r['executed_layers'])!=(6 if int(r['actual_queries']) else 0) for r in rows):raise RuntimeError('Query execution mismatch')
    if len({r['fpn_shapes'] for r in rows})!=1:raise RuntimeError('FPN shapes differ; do not fit a single fixed-shape Model A')
    scenes=[str(s) for s,_,_,_ in m['pairs']];test=set(scenes[-max(1,len(scenes)//3):]) if len(scenes)>=3 else set()
    partial=[r for r in rows if 0<int(r['actual_queries'])<40000]
    train=[r for r in partial if r['scene_token'] not in test];held=[r for r in partial if r['scene_token'] in test]
    def xy(rr,key):return np.array([int(r['actual_queries']) for r in rr],float),np.array([float(r[key]) for r in rr],float)
    models={};preds=[]
    for key in ('encoder_ms','encoder_core_ms'):
        x,y=xy(train,key)
        if len(set(x))<2 or not np.isfinite(y).all() or (y<=0).any():raise RuntimeError('Need two or more valid positive partial query levels')
        # scale x for conditioning; convert coefficient back to ms/query.
        coef=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x/40000]),y,rcond=None)[0];f=float(coef[0]);beta=float(coef[1]/40000)
        model=dict(f_ms=f,beta_ms_per_query=beta,train=metrics(y,f+beta*x),test=None)
        if held:
            xt,yt=xy(held,key);model['test']=metrics(yt,f+beta*xt)
        for r in partial:
            q=int(r['actual_queries']);pred=f+beta*q;preds.append(dict(target=key,scene_token=r['scene_token'],repeat=r['repeat'],actual_queries=q,split='test' if r['scene_token'] in test else 'train',observed_ms=r[key],predicted_ms=pred,residual_ms=float(r[key])-pred))
        models[key]=model
    endpoints={}
    for q in (0,40000):
        a=[float(r['encoder_ms']) for r in rows if int(r['actual_queries'])==q]
        if a:endpoints[str(q)]=dict(n=len(a),mean_ms=float(np.mean(a)),p90_ms=float(np.quantile(a,.9)),path='zero_selective_wrapper' if q==0 else 'full_cached_wrapper')
    result=dict(schema_version=1,fit_scope='0 < actual_queries < 40000 only',models=models,endpoints=endpoints,train_scenes=[s for s in scenes if s not in test],test_scenes=list(sorted(test)),settings=m,warning=None if test else 'Smoke/in-sample only, no scene holdout')
    (out/'encoder_cost_model.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'encoder_predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,5));a=models['encoder_ms'];
        for rr,label in [(train,'Train'),(held,'Held-out scenes')]:
            if rr:
                x,y=xy(rr,'encoder_ms');ax.scatter(x/400,y,s=13,alpha=.5,label=label)
        x=np.array([0,40000]);ax.plot(x/400,a['f_ms']+a['beta_ms_per_query']*x,label='Linear fit (partial path)')
        for q in (0,40000):
            rr=[r for r in rows if int(r['actual_queries'])==q]
            if rr:
                x,y=xy(rr,'encoder_ms');ax.scatter(x/400,y,marker='x',label=f'Endpoint {q} (not fit)')
        ax.set(xlabel='Active BEV queries (%)',ylabel='Encoder wrapper latency (ms)');ax.legend();fig.tight_layout();fig.savefig(out/'encoder_fig9_linear_fit.png',dpi=160);plt.close(fig)
    except ImportError:print('matplotlib unavailable; CSV/JSON still written')
    print(json.dumps({k:result[k] for k in ('models','endpoints','warning')},indent=2));print('Outputs:',out)
if __name__=='__main__':main()
