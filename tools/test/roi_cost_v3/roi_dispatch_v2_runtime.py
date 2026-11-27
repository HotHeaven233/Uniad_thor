#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only geometry/planning primitives for UniAD Thor ROI dispatch V2.

All boxes are half-open image coordinates (x0, y0, x1, y1).  Grid edges are
multiples of `align`, including phase-shifted interior edges.  Edge cells may
be smaller than the nominal tile size but remain feature-stride aligned.
"""
from __future__ import annotations

import math
import time
from bisect import bisect_right
from typing import Dict, Iterable, List, Sequence, Tuple

Box = Tuple[int, int, int, int]
Cell = Tuple[int, int]


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(multiple, int(math.ceil(float(value) / multiple) * multiple))


def round_down(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(0, int(value // multiple) * multiple)


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap_area(a: Box, b: Box) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def rect_union_area(boxes: Sequence[Box]) -> int:
    """Exact union area for axis-aligned rectangles; O(n^2 log n), small n."""
    boxes = [tuple(map(int, b)) for b in boxes if box_area(b) > 0]
    if not boxes:
        return 0
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb <= xa:
            continue
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < xb and b[2] > xa)
        if not intervals:
            continue
        y0, y1 = intervals[0]
        covered = 0
        for a, z in intervals[1:]:
            if a <= y1:
                y1 = max(y1, z)
            else:
                covered += y1 - y0
                y0, y1 = a, z
        covered += y1 - y0
        total += (xb - xa) * covered
    return int(total)


def _axis_edges(length: int, tile: int, phase: int, align: int) -> List[int]:
    padded = round_up(length, align)
    phase = int(phase)
    if phase < 0 or phase >= tile or phase % align:
        raise ValueError("phase must be in [0,tile) and aligned")
    edges = [0]
    if phase > 0:
        edges.append(phase)
    x = phase + tile
    while x < padded:
        if x > edges[-1]:
            edges.append(x)
        x += tile
    if padded > edges[-1]:
        edges.append(padded)
    if len(edges) < 2:
        edges = [0, padded]
    if any((x % align) != 0 for x in edges):
        raise RuntimeError("Non-aligned grid edge generated")
    return edges


def _phase_options(tile: int, align: int) -> List[int]:
    half = round_down(tile // 2, align)
    values = [0]
    if 0 < half < tile:
        values.append(half)
    return sorted(set(values))


def generate_candidate_specs(
    raw_boxes: Sequence[Box],
    image_h: int,
    image_w: int,
    quantiles: Sequence[float] = (0.50, 0.75, 0.90, 1.00),
    align: int = 64,
) -> List[Dict]:
    """Generate deduplicated (tile size, x/y phase) candidates from ROI quantiles."""
    if not raw_boxes:
        raise ValueError("At least one ROI is required")
    widths = sorted(max(1, b[2] - b[0]) for b in raw_boxes)
    heights = sorted(max(1, b[3] - b[1]) for b in raw_boxes)

    def quantile(values, q):
        # Linear interpolation, deterministic and NumPy-independent.
        if len(values) == 1:
            return float(values[0])
        pos = (len(values) - 1) * float(q)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    specs, seen = [], set()
    for q in quantiles:
        if not (0 < float(q) <= 1):
            raise ValueError("quantiles must be in (0,1]")
        tile_w = min(round_up(int(math.ceil(quantile(widths, q))), align), round_up(image_w, align))
        tile_h = min(round_up(int(math.ceil(quantile(heights, q))), align), round_up(image_h, align))
        tile_w, tile_h = max(align, tile_w), max(align, tile_h)
        for py in _phase_options(tile_h, align):
            for px in _phase_options(tile_w, align):
                key = (tile_w, tile_h, px, py)
                if key in seen:
                    continue
                seen.add(key)
                specs.append({
                    "quantile": float(q), "tile_w": int(tile_w), "tile_h": int(tile_h),
                    "phase_x": int(px), "phase_y": int(py),
                    "candidate_id": "q{}_w{}_h{}_px{}_py{}".format(
                        str(q).replace(".", "p"), tile_w, tile_h, px, py
                    ),
                })
    return specs


def build_grid_geometry(
    raw_boxes: Sequence[Box], image_h: int, image_w: int,
    tile_w: int, tile_h: int, phase_x: int = 0, phase_y: int = 0,
    align: int = 64,
) -> Dict:
    boxes = [tuple(map(int, b)) for b in raw_boxes]
    if not boxes:
        raise ValueError("At least one ROI is required")
    for b in boxes:
        if not (0 <= b[0] < b[2] <= image_w and 0 <= b[1] < b[3] <= image_h):
            raise ValueError("ROI outside image bounds or empty: {}".format(b))
    x_edges = _axis_edges(image_w, int(tile_w), int(phase_x), align)
    y_edges = _axis_edges(image_h, int(tile_h), int(phase_y), align)
    active = set()
    roi_tile_hits = 0
    for x0, y0, x1, y1 in boxes:
        c0 = max(0, bisect_right(x_edges, x0) - 1)
        c1 = max(0, bisect_right(x_edges, x1 - 1) - 1)
        r0 = max(0, bisect_right(y_edges, y0) - 1)
        r1 = max(0, bisect_right(y_edges, y1 - 1) - 1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                active.add((r, c))
                roi_tile_hits += 1
    cells = sorted(active)
    cell_boxes = {
        (r, c): (x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1])
        for r, c in cells
    }
    raw_sum = sum(box_area(b) for b in boxes)
    raw_union = rect_union_area(boxes)
    padded_h, padded_w = y_edges[-1], x_edges[-1]
    tile_valid_area = sum(max(0, min(b[2], image_w)-b[0]) * max(0, min(b[3], image_h)-b[1])
                          for b in cell_boxes.values())
    tile_total_area = sum(box_area(b) for b in cell_boxes.values())
    return {
        "raw_boxes": boxes, "image_h": int(image_h), "image_w": int(image_w),
        "align": int(align), "tile_w": int(tile_w), "tile_h": int(tile_h),
        "phase_x": int(phase_x), "phase_y": int(phase_y),
        "x_edges": x_edges, "y_edges": y_edges,
        "padded_h": int(padded_h), "padded_w": int(padded_w),
        "cells": cells, "cell_boxes": cell_boxes,
        "raw_roi_count": len(boxes), "raw_roi_sum_area_px": int(raw_sum),
        "raw_roi_union_area_px": int(raw_union),
        "roi_tile_hits": int(roi_tile_hits), "unique_tile_count": len(cells),
        "tile_dedup_savings": int(roi_tile_hits - len(cells)),
        "tile_input_area_px": int(tile_total_area),
        "tile_input_valid_area_px": int(tile_valid_area),
        "tile_input_outside_image_area_px": int(tile_total_area-tile_valid_area),
        "grid_rows": len(y_edges) - 1, "grid_cols": len(x_edges) - 1,
        "grid_cell_count": (len(y_edges) - 1) * (len(x_edges) - 1),
        "raw_roi_union_area_ratio_pct": 100.0 * raw_union / float(image_h * image_w),
    }


def build_v1_fixed_geometry(raw_boxes: Sequence[Box], image_h: int, image_w: int, align: int = 64) -> Dict:
    """Reproduce V1 max-width/max-height, origin-zero, equal-sized tile geometry."""
    max_w = max(b[2] - b[0] for b in raw_boxes)
    max_h = max(b[3] - b[1] for b in raw_boxes)
    tile_w, tile_h = round_up(max_w, align), round_up(max_h, align)
    cols, rows = int(math.ceil(image_w / tile_w)), int(math.ceil(image_h / tile_h))
    x_edges = [i * tile_w for i in range(cols + 1)]
    y_edges = [i * tile_h for i in range(rows + 1)]
    active, hits = set(), 0
    for x0, y0, x1, y1 in raw_boxes:
        c0, c1 = x0 // tile_w, (x1 - 1) // tile_w
        r0, r1 = y0 // tile_h, (y1 - 1) // tile_h
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                active.add((r, c)); hits += 1
    cells = sorted(active)
    cell_boxes = {(r, c): (x_edges[c], y_edges[r], x_edges[c+1], y_edges[r+1]) for r,c in cells}
    raw_sum = sum(box_area(b) for b in raw_boxes)
    raw_union = rect_union_area(raw_boxes)
    tile_total_area = sum(box_area(cell_boxes[c]) for c in cells)
    tile_valid_area = sum(max(0,min(cell_boxes[c][2],image_w)-cell_boxes[c][0]) *
                          max(0,min(cell_boxes[c][3],image_h)-cell_boxes[c][1]) for c in cells)
    return {
        "raw_boxes": list(raw_boxes), "image_h": image_h, "image_w": image_w,
        "align": align, "tile_w": tile_w, "tile_h": tile_h, "phase_x": 0, "phase_y": 0,
        "x_edges": x_edges, "y_edges": y_edges, "padded_h": rows * tile_h, "padded_w": cols * tile_w,
        "cells": cells, "cell_boxes": cell_boxes, "raw_roi_count": len(raw_boxes),
        "raw_roi_sum_area_px": raw_sum, "raw_roi_union_area_px": raw_union,
        "roi_tile_hits": hits, "unique_tile_count": len(cells), "tile_dedup_savings": hits-len(cells),
        "tile_input_area_px": tile_total_area, "tile_input_valid_area_px": tile_valid_area,
        "tile_input_outside_image_area_px": tile_total_area-tile_valid_area,
        "grid_rows": rows, "grid_cols": cols, "grid_cell_count": rows*cols,
        "raw_roi_union_area_ratio_pct": 100.0*raw_union/(image_h*image_w),
    }


def batch_geometry(boxes: Sequence[Box]) -> Dict:
    if not boxes:
        raise ValueError("Execution plan cannot be empty")
    hs = [b[3]-b[1] for b in boxes]
    ws = [b[2]-b[0] for b in boxes]
    max_h, max_w = max(hs), max(ws)
    crop_area = sum(h*w for h,w in zip(hs,ws))
    padded_area = len(boxes)*max_h*max_w
    return {
        "exec_crop_count": len(boxes), "max_batch_h": max_h, "max_batch_w": max_w,
        "exec_crop_area_sum_px": int(crop_area), "padded_batch_area_px": int(padded_area),
        "batch_padding_area_px": int(padded_area-crop_area),
        "batch_padding_waste_pct": 100.0*(padded_area-crop_area)/max(1,padded_area),
        "merge_expansion_area_px": 0,
    }


def roi_batch_plan(geometry: Dict) -> Dict:
    boxes = [geometry["cell_boxes"][c] for c in geometry["cells"]]
    return {"exec_boxes": boxes, "adjacency_edges": 0, "component_count": len(geometry["cells"]),
            "exact_rect_count": len(boxes), "merge_count": 0, "planner_components": {},
            **batch_geometry(boxes)}


def sparse_adjacency(cells: Sequence[Cell]) -> Tuple[List[Tuple[int,int]], Dict[Cell,int]]:
    index = {cell: i for i, cell in enumerate(cells)}
    edges = []
    for i, (r,c) in enumerate(cells):
        for n in ((r,c+1),(r+1,c)):
            j = index.get(n)
            if j is not None:
                edges.append((i,j))
    return edges, index


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n)); self.rank = [0]*n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1


def connected_components_sparse(n: int, edges: Sequence[Tuple[int,int]]) -> List[List[int]]:
    uf = UnionFind(n)
    for a,b in edges: uf.union(a,b)
    groups: Dict[int,List[int]] = {}
    for i in range(n): groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values())


def exact_rectangle_partition(geometry: Dict) -> List[Box]:
    """Partition active grid cells into non-overlapping rectangles, no expansion."""
    cells = set(geometry["cells"])
    if not cells: return []
    max_row = max(r for r,_ in cells)
    max_col = max(c for _,c in cells)
    active_runs_by_row = {}
    for r in range(max_row+1):
        cols = sorted(c for rr,c in cells if rr == r)
        runs, start, prev = [], None, None
        for c in cols:
            if start is None: start = prev = c
            elif c == prev+1: prev = c
            else: runs.append((start,prev)); start = prev = c
        if start is not None: runs.append((start,prev))
        active_runs_by_row[r] = runs
    open_runs: Dict[Tuple[int,int],int] = {}
    output = []
    y_edges, x_edges = geometry["y_edges"], geometry["x_edges"]
    for r in range(max_row+2):
        runs = active_runs_by_row.get(r, [])
        current = set(runs)
        for run, start_row in list(open_runs.items()):
            if run not in current:
                c0,c1 = run
                output.append((x_edges[c0], y_edges[start_row], x_edges[c1+1], y_edges[r]))
                del open_runs[run]
        for run in runs:
            if run not in open_runs: open_runs[run] = r
    output.sort(key=lambda b:(b[1],b[0],b[3],b[2]))
    return output


def _boxes_overlap_any(candidate: Box, boxes: Sequence[Box], skip: set) -> bool:
    return any(i not in skip and overlap_area(candidate, b) > 0 for i,b in enumerate(boxes))


def greedy_cost_aware_merge(boxes: Sequence[Box], max_expansion: float, stats=None) -> Tuple[List[Box], int, int]:
    """Greedily merge rectangles only if padded Batch pixels decrease.

    A candidate's bounding rectangle may add area up to max_expansion relative
    to the two source rectangles. Merges that overlap a third rectangle are
    rejected so final execution crops remain disjoint (no duplicate pixels).
    """
    if max_expansion < 0: raise ValueError("max_expansion must be >= 0")
    current = list(boxes)
    merge_count, extra_total = 0, 0
    while len(current) > 1:
        base = batch_geometry(current)["padded_batch_area_px"]
        best = None
        for i in range(len(current)):
            for j in range(i+1,len(current)):
                if stats is not None: stats["merge_pair_checks"] += 1
                a,b=current[i],current[j]
                u=(min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))
                extra=box_area(u)-box_area(a)-box_area(b)+overlap_area(a,b)
                denom=max(1,box_area(a)+box_area(b)-overlap_area(a,b))
                if extra/denom > max_expansion + 1e-12: continue
                if _boxes_overlap_any(u,current,{i,j}): continue
                trial=[x for k,x in enumerate(current) if k not in (i,j)]+[u]
                new_cost=batch_geometry(trial)["padded_batch_area_px"]
                saving=base-new_cost
                if saving <= 0: continue
                key=(saving,-extra,-new_cost)
                if best is None or key>best[0]: best=(key,i,j,u,extra)
        if best is None: break
        _,i,j,u,extra=best
        current=[x for k,x in enumerate(current) if k not in (i,j)]+[u]
        current.sort(key=lambda b:(b[1],b[0],b[3],b[2]))
        merge_count += 1; extra_total += int(extra)
    return current, merge_count, extra_total


def adjacency_plan(geometry: Dict, max_expansion: float = 0.0) -> Dict:
    t0=time.perf_counter()
    edges,_=sparse_adjacency(geometry["cells"])
    adjacency_ms=(time.perf_counter()-t0)*1000.0
    t0=time.perf_counter()
    groups=connected_components_sparse(len(geometry["cells"]),edges)
    components_ms=(time.perf_counter()-t0)*1000.0
    t0=time.perf_counter()
    exact=exact_rectangle_partition(geometry)
    partition_ms=(time.perf_counter()-t0)*1000.0
    t0=time.perf_counter()
    stats={"merge_pair_checks":0}
    if max_expansion>0:
        final,merge_count,extra=greedy_cost_aware_merge(exact,max_expansion,stats)
    else:
        final,merge_count,extra=list(exact),0,0
    merge_ms=(time.perf_counter()-t0)*1000.0
    summary=batch_geometry(final)
    return {
        "exec_boxes": final, "adjacency_edges": len(edges), "component_count": len(groups),
        "exact_rect_count": len(exact), "merge_count": merge_count,
        "merge_pair_checks":stats["merge_pair_checks"],
        "planner_components": {"adjacency_build_cpu_ms": adjacency_ms,
            "union_find_cpu_ms": components_ms, "rectangle_partition_cpu_ms": partition_ms,
            "cost_aware_merge_cpu_ms": merge_ms},
        **summary,
        "merge_expansion_area_px": int(extra),
    }


def select_best_plan(raw_boxes: Sequence[Box], image_h: int, image_w: int,
                     mode: str, quantiles=(0.50,0.75,0.90,1.00),
                     max_expansion: float=0.0, align: int=64) -> Dict:
    """Search all tile-size/phase candidates and return minimum padded-pixel plan.

    Candidate search and plan construction times are returned and must be
    counted in runtime CPU overhead. This is a geometry proxy, not a learned
    latency model; the Thor experiment validates whether it predicts well.
    """
    t_all=time.perf_counter()
    t0=time.perf_counter()
    specs=generate_candidate_specs(raw_boxes,image_h,image_w,quantiles,align)
    spec_ms=(time.perf_counter()-t0)*1000.0
    candidates=[]
    component_totals = {"adjacency_build_cpu_ms": 0.0, "union_find_cpu_ms": 0.0,
                        "rectangle_partition_cpu_ms": 0.0, "cost_aware_merge_cpu_ms": 0.0}
    mapping_total_ms = 0.0
    plan_total_ms = 0.0
    for spec in specs:
        t_map=time.perf_counter()
        geom=build_grid_geometry(raw_boxes,image_h,image_w,spec["tile_w"],spec["tile_h"],
                                 spec["phase_x"],spec["phase_y"],align)
        mapping_ms=(time.perf_counter()-t_map)*1000.0
        mapping_total_ms += mapping_ms
        t_plan=time.perf_counter()
        if mode == "roi_batch":
            plan=roi_batch_plan(geom)
        elif mode == "adjacency_merge":
            plan=adjacency_plan(geom,max_expansion)
        else:
            raise ValueError("Unknown mode: {}".format(mode))
        plan_ms=(time.perf_counter()-t_plan)*1000.0
        plan_total_ms += plan_ms
        for key, value in plan.get("planner_components", {}).items():
            component_totals[key] += float(value)
        # ResNet sees every Batch item padded to max H/W. This is the primary
        # geometry proxy. Tie-break on copied pixels, then crop count.
        score=(plan["padded_batch_area_px"], plan["exec_crop_area_sum_px"], plan["exec_crop_count"])
        candidates.append((score,spec,geom,plan,mapping_ms,plan_ms))
    candidates.sort(key=lambda x:x[0])
    _,spec,geom,plan,mapping_ms,plan_ms=candidates[0]
    total_ms=(time.perf_counter()-t_all)*1000.0
    # Keep compact candidate diagnostics for every searched candidate.
    candidate_rows=[]
    for score,s,g,p,mm,pm in candidates:
        candidate_rows.append({
            "candidate_id":s["candidate_id"],"quantile":s["quantile"],"tile_w":s["tile_w"],"tile_h":s["tile_h"],
            "phase_x":s["phase_x"],"phase_y":s["phase_y"],"unique_tile_count":g["unique_tile_count"],
            "exec_crop_count":p["exec_crop_count"],"padded_batch_area_px":p["padded_batch_area_px"],
            "exec_crop_area_sum_px":p["exec_crop_area_sum_px"],"batch_padding_waste_pct":p["batch_padding_waste_pct"],
            "tile_input_valid_area_px":g["tile_input_valid_area_px"],
            "tile_input_outside_image_area_px":g["tile_input_outside_image_area_px"],
            "mapping_cpu_ms":mm,"plan_cpu_ms":pm,"selected":int(s["candidate_id"]==spec["candidate_id"]),
        })
    return {
        "geometry":geom,"plan":plan,"candidate_spec":spec,"candidate_rows":candidate_rows,
        "candidate_count":len(specs),"candidate_spec_cpu_ms":spec_ms,
        "candidate_search_cpu_ms":total_ms,"selected_mapping_cpu_ms":mapping_ms,
        "selected_plan_cpu_ms":plan_ms,
        "mapping_search_cpu_ms":mapping_total_ms,
        "plan_search_cpu_ms":plan_total_ms,
        "planner_components_total_ms":component_totals,
    }


def assert_disjoint_cover(geometry: Dict, boxes: Sequence[Box]) -> None:
    """Raise if execution rectangles overlap or fail to cover active cells exactly."""
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            if overlap_area(a,b)>0: raise AssertionError("Execution rectangles overlap")
    expected=sum(box_area(geometry["cell_boxes"][c]) for c in geometry["cells"])
    actual=sum(box_area(b) for b in boxes)
    if actual != expected:
        raise AssertionError("Exact partition area mismatch: {} != {}".format(actual,expected))
