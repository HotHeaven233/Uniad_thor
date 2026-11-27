#!/usr/bin/env python3
"""Render Figure 3 from real measurements only. Missing panels are labelled."""
import argparse,csv,json,sqlite3,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODULES=['scheduler','backbone','encoder','tracking','mapping','motion','occupancy','planning','bev_other','outside']
OPS=['tsa','sca','ffn','norm','point_sampling','encoder_other']

def read(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def load(path,expected):
    p=Path(path)
    if not (p/'COMPLETE').exists():raise RuntimeError(f'incomplete collection: {p}')
    f=read(p/'frames.csv');s=read(p/'scopes.csv')
    if not f or any(x['profile']!=expected for x in f):raise ValueError(f'wrong profile: {p}')
    keys={x['frame_id'] for x in f}
    if len(keys)!=len(f) or any(x['frame_id'] not in keys for x in s):raise ValueError('duplicate frames/orphan scope')
    by=defaultdict(list)
    for x in s:by[x['frame_id']].append(x)
    if expected!='plain' and set(by)!=keys:raise ValueError('missing scope data')
    return f,by

def union_length(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('negative interval')
        if end is None or a>=end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total

def exclusive(scopes,clock):
    # Exclusive nested scopes, same host thread/current CUDA stream. No GPU active-time claim.
    children=defaultdict(list);idx={int(s['scope_id']):s for s in scopes}
    if len(idx)!=len(scopes):raise ValueError('duplicate scope IDs')
    if clock=='gpu' and len({s['stream'] for s in scopes})!=1:raise ValueError('multiple streams: no additive span decomposition')
    def interval(s):return float(s[clock+'_start_ms']),float(s[clock+'_end_ms'])
    for s in scopes:children[int(s['parent_id'])].append(s)
    result=[]
    for i,s in idx.items():
        a,b=interval(s); spans=[]
        for c in children[i]:
            x,y=interval(c)
            if x<a-0.05 or y>b+0.05:raise ValueError('child outside parent')
            spans.append((max(a,x),min(b,y)))
        result.append((s,max(0,b-a-union_length(spans))))
    return result

def group(s):
    n=s['module'];return n if n in MODULES else 'outside'

def summarize(path):
    f,by=load(path,'summary');rows=[]
    for fr in f:
        ss=by[fr['frame_id']];v={}
        for clock in ['host','gpu']:
            ex=exclusive(ss,clock); vals=defaultdict(float)
            for s,x in ex:vals[group(s)]+=x
            for k in MODULES:v[k+'_'+clock+'_ms']=vals[k]
        wall=float(fr['e2e_ms']);host=float(fr['forward_host_ms'])
        # Outside host spans include wrapper/post-head and final synchronize.
        v['outside_host_ms']+=max(0,host-sum(v[k+'_host_ms'] for k in MODULES))
        v['final_sync_ms']=max(0,wall-host)
        rows.append({**fr,**v})
    return rows

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def trace(ax,db,stats_path=None):
    if not db:
        ax.text(.5,.5,'Nsight trace not collected\nRun trace stage',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();return False
    con=sqlite3.connect(db);tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    apis=[t for t in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') if t in tables]
    kernel=next((t for t in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL') if t in tables),None)
    if not apis or not kernel or 'StringIds' not in tables:raise ValueError('Nsight SQLite missing API/kernel/StringIds tables')
    launch=[]
    for t in apis:
        launch+=list(con.execute(f'SELECT a.start,a.end,a.correlationId,s.value FROM {t} a JOIN StringIds s ON a.nameId=s.id WHERE lower(s.value) LIKE ? ORDER BY a.start',('%launch%',)))
    ks=list(con.execute(f'SELECT start,end,correlationId FROM {kernel} ORDER BY start'))
    if not launch or not ks:raise ValueError('no launches or GPU kernels')
    # Find an actual launch correlated with a GPU kernel, preferably in backbone NVTX.
    backbone=None
    if 'NVTX_EVENTS' in tables:
        cols={r[1] for r in con.execute('PRAGMA table_info(NVTX_EVENTS)')}
        if {'start','end','text'}<=cols:
            backbone=con.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='fig3/backbone' AND end IS NOT NULL ORDER BY start LIMIT 1").fetchone()
        if not backbone and {'start','end','textId'}<=cols:
            backbone=con.execute("SELECT a.start,a.end FROM NVTX_EVENTS a JOIN StringIds s ON a.textId=s.id WHERE s.value='fig3/backbone' AND a.end IS NOT NULL ORDER BY a.start LIMIT 1").fetchone()
    l=[x for x in launch if not backbone or backbone[0]<=x[0]<=backbone[1]]
    kd=defaultdict(list)
    for x in ks:kd[x[2]].append(x)
    pair=next(((x,kd[x[2]][0]) for x in sorted(l) if x[2] in kd),None)
    if not pair:raise ValueError('no correlated launch/kernel pair in selected scope')
    start=min(pair[0][0],pair[1][0]);end=max(start+3_000_000,pair[1][0]+1_000_000)
    for arr,y,color in [(launch,1,'#d08039'),(ks,0,'#3678ac')]:
        for x in arr:
            a,b=x[:2]
            if b>start and a<end:ax.broken_barh([((max(a,start)-start)/1e6,max(0,(min(b,end)-max(a,start))/1e6))],(y-.27,.54),facecolors=color)
    ax.set(yticks=[0,1],yticklabels=['GPU kernels','CPU launch APIs'],xlabel='Time (ms, selected trace window)',xlim=(0,(end-start)/1e6),ylim=(-.5,1.5))
    ax.text(.01,.04,'Backbone NVTX window' if backbone else 'E2E window (no Backbone NVTX match)',transform=ax.transAxes,fontsize=7)
    if stats_path:
        api_rows=[]
        for t in apis:
            for name,count,duration in con.execute(f'SELECT s.value,count(*),sum(a.end-a.start) FROM {t} a JOIN StringIds s ON a.nameId=s.id GROUP BY s.value ORDER BY sum(a.end-a.start) DESC'):
                api_rows.append(dict(api_table=t,api=name,count=count,total_api_duration_ms=duration/1e6))
        write_csv(stats_path/'nsys_api_stats.csv',api_rows)
        (stats_path/'nsys_trace_summary.json').write_text(json.dumps(dict(kernel_count=len(ks),kernel_duration_sum_ms=sum(x[1]-x[0] for x in ks)/1e6,kernel_interval_union_ms=union_length([(x[0],x[1]) for x in ks])/1e6,backbone_nvtx_found=bool(backbone),notes=['Captured measured forward only. API durations are not critical-path launch overhead. Runtime and driver API tables may nest; do not sum them. Kernel interval union describes captured kernel activity, not total GPU work or E2E.']),indent=2))
    con.close();return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',nargs='+',required=True);p.add_argument('--detail');p.add_argument('--area');p.add_argument('--trace-sqlite');p.add_argument('--plain');p.add_argument('--out-dir',required=True);a=p.parse_args()
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    allrows=[];stats=[];labels=[]
    for path in a.summary:
        rr=summarize(path);allrows+=rr;label=rr[0]['mode'];labels.append(label)
        for k in MODULES+['final_sync','e2e']:
            keys=['e2e_ms'] if k=='e2e' else ['final_sync_ms'] if k=='final_sync' else [k+'_host_ms',k+'_gpu_ms']
            for key in keys:
                v=np.array([float(x[key]) for x in rr]);stats.append(dict(mode=label,metric=key,n=len(v),mean_ms=v.mean(),median_ms=np.median(v),p90_ms=np.percentile(v,90),p95_ms=np.percentile(v,95),p99_ms=np.percentile(v,99),max_ms=v.max()))
    if len(set(labels))!=len(labels):raise ValueError('one summary directory per mode')
    if len(labels)>1:
        sets=[{(r['scene'],r['dataset_index'],r['repeat']) for r in allrows if r['mode']==m} for m in labels]
        if any(x!=sets[0] for x in sets[1:]):raise ValueError('summary modes not paired: reuse the same manifest/repeats')
    write_csv(out/'frame_exclusive_spans.csv',allrows);write_csv(out/'table_iv_module_stats.csv',stats)
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);ax=axs[0,0]
    # Host exclusive + final device drain: partition wall-clock boundary, NOT hardware execution shares.
    colors=plt.get_cmap('tab20').colors
    for i,label in enumerate(labels):
        rr=[x for x in allrows if x['mode']==label];left=0
        for j,k in enumerate(MODULES+['final_sync']):
            key=k+'_host_ms' if k!='final_sync' else 'final_sync_ms';v=np.mean([float(x[key]) for x in rr]);ax.barh(i,v,left=left,color=colors[j],label=k if i==0 else None);left+=v
        ax.text(left,i,f' {left:.1f}',va='center',fontsize=8)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Mean E2E wall-time partition (ms)',title='(a) Host scopes + final CUDA drain');ax.legend(fontsize=7,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.35));ax.invert_yaxis()
    ax=axs[0,1];ax.set_title('(b) CPU launch / GPU execution timeline')
    try:trace_ok=trace(ax,a.trace_sqlite,out)
    except Exception as e:
        trace_ok=False;ax.text(.5,.5,f'Trace unavailable: {e}',wrap=True,ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    ax=axs[1,0];ax.set_title('(c) Encoder operator stream spans')
    if a.detail:
        fr,by=load(a.detail,'detail');drows=[]
        for f in fr:
            ss=by[f['frame_id']];anc={int(s['scope_id']):s for s in ss};vals=defaultdict(float)
            for s,v in exclusive(ss,'gpu'):
                x=s; inside=False
                while True:
                    if x['module']=='encoder':inside=True;break
                    if int(x['parent_id'])<0:break
                    x=anc[int(x['parent_id'])]
                if inside:
                    n=s['module'].replace('_selective','');vals[n if n in OPS else 'encoder_other']+=v
            drows.append(dict(frame_id=f['frame_id'],**{n:vals[n] for n in OPS}))
        write_csv(out/'encoder_operator_frame.csv',drows)
        v=[np.mean([x[n] for x in drows]) for n in OPS];ax.barh(OPS,v,color='#3678ac');ax.invert_yaxis();ax.set_xlabel('Mean exclusive CUDA stream span (ms)')
    else:ax.text(.5,.5,'Detailed pass not collected',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    ax=axs[1,1];ax.set_title('(d) Backbone small-input characterization (B=1)')
    if a.area:
        fr,by=load(a.area,'area');rr=[]
        for f in fr:
            for s in by[f['frame_id']]:
                if s['module']=='area_cnn':
                    attrs=json.loads(s['attrs']);rr.append(dict(frame_id=f['frame_id'],scene=f['scene'],**attrs,cnn_stream_ms=float(s['gpu_end_ms'])-float(s['gpu_start_ms']),cnn_host_ms=float(s['host_end_ms'])-float(s['host_start_ms'])))
        if not rr:raise RuntimeError('no area CNN samples')
        write_csv(out/'area_camera_raw.csv',rr);groups=defaultdict(list)
        for x in rr:groups[x['pixels']].append(x['cnn_stream_ms'])
        xx=sorted(groups);yy=[np.median(groups[x]) for x in xx];lo=[np.percentile(groups[x],10) for x in xx];hi=[np.percentile(groups[x],90) for x in xx]
        ax.scatter([r['pixels'] for r in rr],[r['cnn_stream_ms'] for r in rr],s=5,alpha=.15,color='#3678ac');ax.plot(xx,yy,'o-',color='#3678ac',label='Median');ax.fill_between(xx,lo,hi,alpha=.18,color='#3678ac',label='P10–P90');ax.set(xlabel='Actual crop input pixels',ylabel='ResNet + FPN stream span (ms)');ax.legend(fontsize=8);ax.set_xscale('log')
    else:ax.text(.5,.5,'Area probe not collected',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off()
    for ax in axs.flat:ax.grid(axis='x',alpha=.18)
    fig.savefig(out/'figure3_thor.pdf');fig.savefig(out/'figure3_thor.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,len(labels),figsize=(6*len(labels),4.8),squeeze=False,constrained_layout=True)
    for ax,label in zip(axes[0],labels):
        rr=[x for x in allrows if x['mode']==label]; names=[k for k in MODULES if k!='outside' and (k!='scheduler' or label!='full')];y=np.arange(len(names));
        ax.barh(y-.18,[np.mean([float(x[k+'_host_ms']) for x in rr]) for k in names],height=.34,label='Host exclusive scope',color='#d08039')
        ax.barh(y+.18,[np.mean([float(x[k+'_gpu_ms']) for x in rr]) for k in names],height=.34,label='CUDA stream exclusive span',color='#3678ac')
        ax.set(yticks=y,yticklabels=names,xlabel='Mean time (ms; not additive)',title=label);ax.invert_yaxis();ax.legend(fontsize=8)
    fig.savefig(out/'host_vs_stream.pdf');fig.savefig(out/'host_vs_stream.png',dpi=220);plt.close(fig)
    audit={}
    if a.plain:
        fr,_=load(a.plain,'plain');base={(r['scene'],r['dataset_index'],r['repeat']):float(r['e2e_ms']) for r in fr};cmp=[r for r in allrows if r['mode']==fr[0]['mode']];d=[float(r['e2e_ms'])-base[(r['scene'],r['dataset_index'],r['repeat'])] for r in cmp if (r['scene'],r['dataset_index'],r['repeat']) in base];audit=dict(pairs=len(d),mean_summary_minus_plain_ms=float(np.mean(d)) if d else None)
    (out/'report.json').write_text(json.dumps(dict(modes=labels,frames={m:sum(r['mode']==m for r in allrows) for m in labels},trace_available=trace_ok,detail_available=bool(a.detail),area_available=bool(a.area),instrumentation_audit=audit,notes=['Host scope partition includes asynchronous waits; it is not GPU compute attribution.','GPU spans include launch gaps and idle time; do not sum host and GPU.','Area probe is synthetic crop geometry inside real E2E and not accuracy evaluation.','Only native Full is measured unless original QC port is explicitly supplied.']),indent=2))
    print('Figure 3 and Table IV:',out)
if __name__=='__main__':main()
