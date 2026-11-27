"""CPU-only analytical model and online plan choice. No CUDA or sklearn dependency."""
from __future__ import annotations
import math
import time
from roi_dispatch_v2_runtime import (generate_candidate_specs, build_grid_geometry,
    roi_batch_plan, adjacency_plan)

A='v2_roi_batch'; B='adjacency_20'; D='adaptive'; SHAPE='shape_scan'
COMPONENTS=('cnn','pack','scatter')

def features(row, component):
    b=float(row['batch_size']); h=float(row['max_batch_h'])/64; w=float(row['max_batch_w'])/64
    if min(b,h,w)<=0: raise ValueError('invalid batch shape')
    p=b*h*w
    if component=='cnn':
        return [p,b,h,w,b*h,b*w,h*w,math.sqrt(p),p*math.log2(b),p*math.log2(h),p*math.log2(w),b*max(h,w),b*min(h,w)]
    r=float(row['exec_crop_count']); area=float(row['exec_crop_area_sum_px'])/4096
    z=float(row['source_pad_added_pixels'])/4096
    if component=='pack': return [r,area,p,z,float(z>0),b*h,b*w]
    if component=='scatter': return [r,area,h,w,r*math.log2(max(1,r))]
    raise ValueError(component)

def predict_component(model,row,component):
    m=model['components'][component]; x=features(row,component)
    return max(0.0,m['intercept']+sum(c*(v-u)/s for c,v,u,s in zip(m['coef'],x,m['mean'],m['scale'])))

def predict_exec(model,row):
    return sum(predict_component(model,row,k) for k in COMPONENTS)

def shape_vector(row):
    return [float(row[k]) for k in ('batch_size','max_batch_h','max_batch_w','padded_batch_area_px')]

def supported(model,row):
    return all(lo<=v<=hi for v,lo,hi in zip(shape_vector(row),model['shape_min'],model['shape_max']))

def plan_row(g,p):
    return dict(batch_size=p['exec_crop_count'],max_batch_h=p['max_batch_h'],max_batch_w=p['max_batch_w'],
                padded_batch_area_px=p['padded_batch_area_px'],exec_crop_count=p['exec_crop_count'],
                exec_crop_area_sum_px=p['exec_crop_area_sum_px'],
                source_pad_added_pixels=g['padded_h']*g['padded_w']-g['image_h']*g['image_w'])

def choose(model, a, b):
    pa,pb=predict_exec(model,a),predict_exec(model,b)
    ok=supported(model,a) and supported(model,b)
    return (B if ok and pa-pb>model['margin_ms'] else A),pa,pb,ok

def _score(plan):
    return (plan['padded_batch_area_px'],plan['exec_crop_area_sum_px'],plan['exec_crop_count'])

def search_plans(raw,h,w,quantiles,align,mode,model=None):
    """Same geometry ranking as V2; adaptive shares mapping but builds both plans.
    All search work is timed, including discarded candidate plans. B uses V2's
    per-local-merge cap, not a global expansion constraint.
    """
    start=time.perf_counter(); spec_start=start
    specs=generate_candidate_specs(raw,h,w,quantiles,align)
    spec_ms=(time.perf_counter()-spec_start)*1000
    modes=(A,B) if mode==D else (mode,)
    best={}; mapping=0.; plan_time=0.; adj_time=0.; checks=0; totals={}; candidates=[]
    for s in specs:
        t=time.perf_counter(); g=build_grid_geometry(raw,h,w,s['tile_w'],s['tile_h'],s['phase_x'],s['phase_y'],align)
        mapping+=(time.perf_counter()-t)*1000
        for m in modes:
            t=time.perf_counter(); p=roi_batch_plan(g) if m==A else adjacency_plan(g,.2)
            elapsed=(time.perf_counter()-t)*1000; plan_time+=elapsed
            if m==B: adj_time+=elapsed; checks+=p.get("merge_pair_checks",0)
            for k,v in p.get('planner_components',{}).items(): totals[k]=totals.get(k,0.)+v
            score=_score(p)
            if m not in best or score<best[m][0]: best[m]=(score,g,p,s)
    selection_ms=0.; pa=pb=None; ok=None; chosen=mode
    if mode==D:
        t=time.perf_counter(); ga,pa_plan=best[A][1:3]; gb,pb_plan=best[B][1:3]
        chosen,pa,pb,ok=choose(model,plan_row(ga,pa_plan),plan_row(gb,pb_plan))
        selection_ms=(time.perf_counter()-t)*1000
    elif model is not None:
        t=time.perf_counter(); pred=predict_exec(model,plan_row(*best[mode][1:3])); selection_ms=(time.perf_counter()-t)*1000
        if mode==A: pa=pred
        else: pb=pred
    _,g,p,s=best[chosen]
    return g,p,{'candidate_count':len(specs),'candidate_spec_cpu_ms':spec_ms,
        'candidate_search_cpu_ms':(time.perf_counter()-start)*1000,'mapping_search_cpu_ms':mapping,
        'plan_search_cpu_ms':plan_time,'planner_components_total_ms':totals,'candidate_spec':s,
        'adjacency_search_cpu_ms':adj_time,'merge_pair_checks':checks,'prediction_cpu_ms':selection_ms,'chosen_mode':chosen,
        'pred_a_ms':pa,'pred_b_ms':pb,'both_shapes_supported':ok},candidates
