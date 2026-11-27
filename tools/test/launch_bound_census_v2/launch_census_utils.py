"""CPU-only workload geometry and conservative knee classification."""
import math
import random
import statistics as st
from collections import defaultdict, Counter

BUDGETS=(0.3125,0.625,1.25,2.5,5.0,10.0,20.0,40.0,60.0,80.0,100.0)
MODES=('roi_batch','adjacency_merge')
LAYOUTS=('compact','spread','mixed')


def raw_boxes(h,w,budget,seed,layout,align=64):
    """Nested image-space ROI union, quantized to aligned cells (edge clipped).
    Partition selected cells into disjoint small rectangles; no BEV dependency.
    The nominal target is raw image area, NOT a cap on padded CNN pixels.
    """
    if min(h,w,align)<=0 or not 0<budget<=100 or layout not in LAYOUTS:
        raise ValueError('invalid image/budget/layout')
    nr,nc=math.ceil(h/align),math.ceil(w/align)
    rng=random.Random(seed)
    cells=[(r,c) for r in range(nr) for c in range(nc)]
    centers=[(rng.uniform(0,nr-1),rng.uniform(0,nc-1)) for _ in range(3)]
    noise={x:rng.random() for x in cells}
    if layout=='spread':
        cells.sort(key=lambda x:noise[x])
    else:
        centers=centers[:1] if layout=='compact' else centers
        cells.sort(key=lambda x:(min((x[0]-r)**2+(x[1]-c)**2 for r,c in centers)+noise[x]*.5,x))
    def area(cell):
        r,c=cell;return min(align,h-r*align)*min(align,w-c*align)
    target=h*w*budget/100
    selected=set();used=0
    for cell in cells:
        a=area(cell)
        if selected and abs(used-target)<=abs(used+a-target):break
        selected.add(cell);used+=a
        if used>=target:break
    # Bound raw ROI rectangle sizes (varied per frame/camera). Preserve exact union.
    bh,bw=rng.choice((1,2,3)),rng.choice((1,2,3,4))
    pending=set(selected);boxes=[]
    while pending:
        r,c=min(pending);cw=1
        while cw<bw and (r,c+cw) in pending:cw+=1
        rh=1
        while rh<bh and all((r+rh,c+j) in pending for j in range(cw)):rh+=1
        for i in range(rh):
            for j in range(cw):pending.remove((r+i,c+j))
        boxes.append((c*align,r*align,min(w,(c+cw)*align),min(h,(r+rh)*align)))
    assert sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in boxes)==used
    return boxes,used


def plan_for_budget(h,w,budget,seed,layout,mode,axis='packed'):
    """Use existing Thor geometry. Packed axis caps B*H*W per camera.

    CPU-only candidate search, not an extra GPU budget sweep. Because the
    planner is nonmonotonic, this is a finite search, not an optimal solver.
    """
    from roi_tile_dispatch_runtime import (build_fixed_tile_geometry,
        build_execution_plan,summarize_execution_geometry)
    if axis not in ('packed','raw'):raise ValueError(axis)
    cap=h*w*budget/100
    def candidate(q):
        boxes,area=raw_boxes(h,w,q,seed,layout)
        g=build_fixed_tile_geometry(boxes,h,w,unit_align=32,feature_align=64)
        p=build_execution_plan(g,mode)
        z=summarize_execution_geometry(g,p)
        if z['raw_roi_union_area_px']!=area:raise RuntimeError('raw ROI union mismatch')
        return boxes,area,g,p,z
    if axis=='raw':return candidate(budget)
    best=None;best_score=None;seen=set()
    def evaluate(q):
        nonlocal best,best_score
        q=round(q,6)
        if q in seen:return
        seen.add(q);c=candidate(q);pixels=c[-1]['padded_batch_area_px']
        if pixels<=cap:
            # Seek close to compute budget, break ties by useful raw area.
            score=(pixels,c[1])
            if best_score is None or score>best_score:best,best_score=c,score
    fractions=(1,.85,.7,.55,.4,.25,.15,.08,.04,.02)
    for fraction in fractions:evaluate(budget*fraction)
    # Refine near the best raw area, with no monotonicity assumption.
    if best is not None:
        raw_pct=100*best[1]/(h*w)
        for delta in (-.075,-.04,.04,.075,.12):
            evaluate(max(.001,min(budget,raw_pct+budget*delta)))
    if best is None:raise ValueError('budget smaller than feasible aligned crop; no silent overrun')
    return best


def compare(rows,mode,lo,hi,flat=.15,rise=.25,min_area_gain=.25):
    """Paired case medians across repeats. No interpretation as kernel busy time."""
    groups=defaultdict(list)
    for r in rows:
        if r['mode']==mode and r['budget_pct'] in (lo,hi):
            groups[(r['case_id'],r['budget_pct'])].append(r)
    cases=sorted({k[0] for k in groups})
    rel=[];area=[];host=[]
    for c in cases:
        if (c,lo) not in groups or (c,hi) not in groups:raise ValueError('unpaired pilot cases')
        l,u=groups[c,lo],groups[c,hi]
        med=lambda rs,k:st.median(float(r[k]) for r in rs)
        t0,t1=med(l,'cnn_stream_ms'),med(u,'cnn_stream_ms')
        a0,a1=med(l,'padded_pixels'),med(u,'padded_pixels')
        if min(t0,a0)<=0:raise ValueError('invalid time/area')
        rel.append(t1/t0-1);area.append(a1/a0-1)
        host.append(med(l,'cnn_host_ms')/t0)
    if not cases:raise ValueError('no pilot cases')
    n=len(cases);needed=math.ceil(2*n/3)
    area_ok=sum(a>=min_area_gain for a in area)
    flat_n=sum(a>=min_area_gain and abs(t)<=flat for a,t in zip(area,rel))
    rise_n=sum(a>=min_area_gain and t>=rise for a,t in zip(area,rel))
    med_t,med_a=st.median(rel),st.median(area)
    if area_ok<needed:status='geometry_limited'
    elif flat_n>=needed and abs(med_t)<=flat:status='flat'
    elif rise_n>=needed and med_t>=rise:status='rising'
    else:status='mixed'
    return dict(lo=lo,hi=hi,status=status,paired_cases=n,required_cases=needed,
                median_time_gain=med_t,median_padded_area_gain=med_a,area_ok_cases=area_ok,
                flat_cases=flat_n,rising_cases=rise_n,median_low_host_to_stream=st.median(host))


def decide(rows,mode,budgets,flat=.15,rise=.25,min_area_gain=.25):
    bs=sorted(set(budgets))
    if len(bs)<2:raise ValueError('need at least two pilot budgets')
    comparisons=[compare(rows,mode,l,h,flat,rise,min_area_gain) for l,h in zip(bs,bs[1:])]
    plateau=[bs[0]];upper=None;status='unresolved'
    for c in comparisons:
        if c['status']=='flat':plateau.append(c['hi']);continue
        upper=c['hi']
        if c['status']=='rising' and len(plateau)>=2:status='bracketed_plateau'
        elif len(plateau)>=2:status='plateau_without_clear_upper'
        else:status='no_plateau_at_available_low_budgets'
        break
    else:status='plateau_continues' if len(plateau)>=2 else 'unresolved'
    # Do not pretend the minimum tested budget is confirmed launch-bound.
    selected=plateau if len(plateau)>=2 else []
    return dict(mode=mode,status=status,plateau_budgets=selected,upper_probe_budget=upper,
                candidate_knee_budget=selected[-1] if selected else None,
                selected_budgets=selected,comparisons=comparisons)


def coverage(counts,ks=(1,3,5,10,20,50)):
    counts=Counter(counts);n=sum(counts.values());rank=sorted(counts.values(),reverse=True)
    return {str(k):sum(rank[:k])/n if n else 0. for k in ks}
