#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry and runtime selector for fixed-tile ROI-Batch / Adjacency-Merge.

This module is independent of BEV queries and does not import PyTorch/MMCV.
Input boxes use half-open image coordinates: (x0, y0, x1, y1).

A fixed crop tile is sized from the maximum ROI width and height. The requested
alignment is 32 px. The effective alignment is lcm(32, 64)=64 px because the
UniAD FPN has a stride-64 output; this keeps every crop origin/extent on all
FPN feature lattices. 64 px is still a multiple of the requested 32 px.
"""
from __future__ import annotations

import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Box = Tuple[int, int, int, int]
Cell = Tuple[int, int]  # (row, col)

FRAME_FEATURE_NAMES = (
    "camera_count",
    "sum_raw_roi_count",
    "sum_raw_roi_sum_area_mpix",
    "sum_raw_roi_union_area_mpix",
    "sum_unique_tile_count",
    "sum_tile_input_area_mpix",
    "sum_roi_tile_hits",
    "sum_tile_dedup_savings",
    "sum_estimated_adjacency_edges",
    "mean_estimated_edge_density",
    "max_camera_unique_tiles",
    "sum_grid_cell_count",
    "mean_unit_width_norm",
    "mean_unit_height_norm",
    "mean_roi_area_cv",
    "mean_roi_aspect_cv",
    "mean_roi_center_spread",
    "sum_tile_input_area_sq",
    "sum_unique_tile_count_sq",
    "sum_estimated_edges_sq",
)

COMPONENT_TARGETS = (
    "tile_mapping_cpu_ms",
    "adjacency_matrix_cpu_ms",
    "connected_components_cpu_ms",
    "bbox_overlap_merge_cpu_ms",
    "tile_pad_host_ms",
    "tile_pad_gpu_ms",
    "feature_canvas_init_gpu_ms",
    "crop_pack_host_ms",
    "crop_pack_gpu_ms",
    "resnet_gpu_ms",
    "fpn_gpu_ms",
    "scatter_gpu_ms",
)


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return max(multiple, int(math.ceil(float(value) / multiple) * multiple))


def box_area(box: Box) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def rectangle_union_area(boxes: Sequence[Box]) -> int:
    """Exact union area of a small collection of axis-aligned rectangles."""
    boxes = [tuple(map(int, b)) for b in boxes if box_area(b) > 0]
    if not boxes:
        return 0
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb <= xa:
            continue
        intervals = sorted(
            (b[1], b[3]) for b in boxes if b[0] < xb and b[2] > xa
        )
        if not intervals:
            continue
        y0, y1 = intervals[0]
        covered = 0
        for a, z in intervals[1:]:
            if a <= y1:
                y1 = max(y1, z)
            else:
                covered += max(0, y1 - y0)
                y0, y1 = a, z
        covered += max(0, y1 - y0)
        total += (xb - xa) * covered
    return int(total)


def _overlap_area(a: Box, b: Box) -> int:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def _merge_two(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def build_fixed_tile_geometry(
    raw_boxes: Sequence[Box],
    image_h: int,
    image_w: int,
    unit_align: int = 32,
    feature_align: int = 64,
) -> Dict:
    """Map every ROI to every intersecting non-overlapping fixed tile.

    Tile size is derived independently for width and height from the largest
    ROI width/height, rounded up to `unit_align`, then rounded up further to
    `feature_align` so all crop origins align with FPN strides 8/16/32/64.
    Overlapping ROIs activate the same cells; a set removes duplicates.
    """
    boxes = [tuple(map(int, b)) for b in raw_boxes]
    if not boxes:
        raise ValueError("At least one ROI is required")
    if image_h <= 0 or image_w <= 0:
        raise ValueError("Image dimensions must be positive")
    for b in boxes:
        if not (0 <= b[0] < b[2] <= image_w and 0 <= b[1] < b[3] <= image_h):
            raise ValueError("ROI is outside image bounds or empty: {}".format(b))
    if unit_align <= 0 or feature_align <= 0:
        raise ValueError("Alignment values must be positive")

    max_roi_w = max(b[2] - b[0] for b in boxes)
    max_roi_h = max(b[3] - b[1] for b in boxes)
    unit_w_32 = round_up(max_roi_w, unit_align)
    unit_h_32 = round_up(max_roi_h, unit_align)
    # 64 is a multiple of 32 and is the coarsest FPN stride in this UniAD config.
    effective_align = math.lcm(unit_align, feature_align)
    unit_w = round_up(unit_w_32, effective_align)
    unit_h = round_up(unit_h_32, effective_align)

    grid_cols = int(math.ceil(float(image_w) / unit_w))
    grid_rows = int(math.ceil(float(image_h) / unit_h))
    pad_w = grid_cols * unit_w
    pad_h = grid_rows * unit_h

    active = set()
    roi_tile_hits = 0
    per_roi_tile_counts = []
    for x0, y0, x1, y1 in boxes:
        c0 = x0 // unit_w
        c1 = (x1 - 1) // unit_w
        r0 = y0 // unit_h
        r1 = (y1 - 1) // unit_h
        hits = 0
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                active.add((row, col))
                hits += 1
        roi_tile_hits += hits
        per_roi_tile_counts.append(hits)

    cells = sorted(active)
    unit_area = unit_w * unit_h
    raw_sum_area = sum(box_area(b) for b in boxes)
    raw_union_area = rectangle_union_area(boxes)
    total_grid_cells = grid_rows * grid_cols
    tile_input_area = len(cells) * unit_area

    widths = np.asarray([b[2] - b[0] for b in boxes], dtype=np.float64)
    heights = np.asarray([b[3] - b[1] for b in boxes], dtype=np.float64)
    areas = widths * heights
    ratios = widths / np.maximum(heights, 1.0)
    centers_x = np.asarray([(b[0] + b[2]) * 0.5 / image_w for b in boxes])
    centers_y = np.asarray([(b[1] + b[3]) * 0.5 / image_h for b in boxes])
    area_cv = float(areas.std() / max(float(areas.mean()), 1.0)) if len(areas) > 1 else 0.0
    aspect_cv = float(ratios.std() / max(float(ratios.mean()), 1e-9)) if len(ratios) > 1 else 0.0
    center_spread = float(math.sqrt(float(centers_x.var() + centers_y.var()))) if len(boxes) > 1 else 0.0

    return {
        "raw_boxes": boxes,
        "cells": cells,
        "image_h": int(image_h),
        "image_w": int(image_w),
        "unit_align_requested": int(unit_align),
        "feature_align": int(feature_align),
        "effective_align": int(effective_align),
        "unit_w_32": int(unit_w_32),
        "unit_h_32": int(unit_h_32),
        "unit_w": int(unit_w),
        "unit_h": int(unit_h),
        "grid_cols": int(grid_cols),
        "grid_rows": int(grid_rows),
        "pad_w": int(pad_w),
        "pad_h": int(pad_h),
        "raw_roi_count": int(len(boxes)),
        "raw_roi_sum_area_px": int(raw_sum_area),
        "raw_roi_union_area_px": int(raw_union_area),
        "raw_roi_union_area_ratio_pct": 100.0 * raw_union_area / float(image_h * image_w),
        "raw_roi_sum_area_ratio_pct": 100.0 * raw_sum_area / float(image_h * image_w),
        "max_roi_w": int(max_roi_w),
        "max_roi_h": int(max_roi_h),
        "grid_cell_count": int(total_grid_cells),
        "unique_tile_count": int(len(cells)),
        "roi_tile_hits": int(roi_tile_hits),
        "tile_dedup_savings": int(roi_tile_hits - len(cells)),
        "tile_dedup_ratio": float(1.0 - len(cells) / max(1, roi_tile_hits)),
        "unit_area_px": int(unit_area),
        "tile_input_area_px": int(tile_input_area),
        "tile_input_area_ratio_pct": 100.0 * tile_input_area / float(image_h * image_w),
        "grid_occupancy_ratio": float(len(cells) / max(1, total_grid_cells)),
        "mean_roi_tiles": float(np.mean(per_roi_tile_counts)),
        "roi_area_cv": area_cv,
        "roi_aspect_cv": aspect_cv,
        "roi_center_spread": center_spread,
        "raw_boxes_json": json.dumps(boxes),
        "cells_json": json.dumps(cells),
    }


def tile_cell_box(cell: Cell, unit_w: int, unit_h: int) -> Box:
    row, col = cell
    x0, y0 = col * unit_w, row * unit_h
    return (x0, y0, x0 + unit_w, y0 + unit_h)


def build_adjacency_matrix(cells: Sequence[Cell]):
    """Dense symmetric adjacency matrix; only shared-edge neighbours connect."""
    cells = list(cells)
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    adjacency = np.zeros((n, n), dtype=np.uint8)
    edge_count = 0
    for i, (row, col) in enumerate(cells):
        for neighbour in ((row, col + 1), (row + 1, col)):
            j = index.get(neighbour)
            if j is not None:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                edge_count += 1
    return adjacency, int(edge_count)


def estimate_shared_edge_count(cells: Sequence[Cell]) -> int:
    """O(K) edge-count estimate without allocating the dense adjacency matrix."""
    cell_set = set(cells)
    return sum(1 for r, c in cell_set if (r, c + 1) in cell_set) + sum(
        1 for r, c in cell_set if (r + 1, c) in cell_set
    )


def connected_components(adjacency: np.ndarray) -> List[List[int]]:
    n = int(adjacency.shape[0])
    seen = np.zeros(n, dtype=np.bool_)
    groups: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            neighbours = np.flatnonzero(adjacency[i])
            for j in neighbours.tolist():
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        groups.append(group)
    return groups


def component_boxes_from_cells(
    cells: Sequence[Cell], groups: Sequence[Sequence[int]], unit_w: int, unit_h: int
) -> List[Box]:
    boxes = []
    for group in groups:
        group_cells = [cells[i] for i in group]
        min_r = min(x[0] for x in group_cells)
        max_r = max(x[0] for x in group_cells)
        min_c = min(x[1] for x in group_cells)
        max_c = max(x[1] for x in group_cells)
        boxes.append((min_c * unit_w, min_r * unit_h,
                      (max_c + 1) * unit_w, (max_r + 1) * unit_h))
    return boxes


def merge_overlapping_component_boxes(boxes: Sequence[Box]):
    """Merge component bounding boxes that overlap in positive pixel area.

    Connected components are edge-connected tile groups. Their rectangular
    bounding boxes can overlap even when the underlying groups are disconnected.
    Merge such boxes to ensure the same image pixels are never sent to ResNet
    more than once in Adjacency-Merge.
    """
    current = [tuple(map(int, b)) for b in boxes]
    merge_count = 0
    changed = True
    while changed:
        changed = False
        n = len(current)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if _overlap_area(current[i], current[j]) > 0:
                    union(i, j)
        groups: Dict[int, List[Box]] = {}
        for i, b in enumerate(current):
            groups.setdefault(find(i), []).append(b)
        new_boxes = [_merge_two_group(group) for group in groups.values()]
        if len(new_boxes) < len(current):
            merge_count += len(current) - len(new_boxes)
            changed = True
        current = sorted(new_boxes, key=lambda b: (b[1], b[0], b[3], b[2]))
    return current, int(merge_count)


def _merge_two_group(group: Sequence[Box]) -> Box:
    return (min(b[0] for b in group), min(b[1] for b in group),
            max(b[2] for b in group), max(b[3] for b in group))


def build_execution_plan(geometry: Dict, mode: str):
    """Build crop rectangles for one camera under either execution strategy."""
    cells = list(geometry["cells"])
    unit_w, unit_h = int(geometry["unit_w"]), int(geometry["unit_h"])
    if mode == "roi_batch":
        exec_boxes = [tile_cell_box(cell, unit_w, unit_h) for cell in cells]
        return {
            "mode": mode,
            "exec_boxes": exec_boxes,
            "adjacency_matrix": None,
            "adjacency_edges": 0,
            "adjacency_density": 0.0,
            "component_count": len(cells),
            "bbox_overlap_merge_count": 0,
            "adjacency_matrix_cpu_ms": 0.0,
            "connected_components_cpu_ms": 0.0,
            "bbox_overlap_merge_cpu_ms": 0.0,
        }
    if mode != "adjacency_merge":
        raise ValueError("Unknown mode: {}".format(mode))

    import time
    t0 = time.perf_counter()
    adjacency, edge_count = build_adjacency_matrix(cells)
    matrix_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    groups = connected_components(adjacency)
    component_boxes = component_boxes_from_cells(cells, groups, unit_w, unit_h)
    component_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    exec_boxes, overlap_merges = merge_overlapping_component_boxes(component_boxes)
    overlap_ms = (time.perf_counter() - t2) * 1000.0

    density = edge_count / float(max(1, len(cells) * (len(cells) - 1) // 2))
    return {
        "mode": mode,
        "exec_boxes": exec_boxes,
        "adjacency_matrix": adjacency,
        "adjacency_edges": int(edge_count),
        "adjacency_density": float(density),
        "component_count": int(len(groups)),
        "bbox_overlap_merge_count": int(overlap_merges),
        "adjacency_matrix_cpu_ms": float(matrix_ms),
        "connected_components_cpu_ms": float(component_ms),
        "bbox_overlap_merge_cpu_ms": float(overlap_ms),
    }


def summarize_execution_geometry(geometry: Dict, plan: Dict) -> Dict[str, float]:
    boxes = plan["exec_boxes"]
    crop_areas = [box_area(b) for b in boxes]
    crop_area_sum = int(sum(crop_areas))
    max_h = max((b[3] - b[1] for b in boxes), default=0)
    max_w = max((b[2] - b[0] for b in boxes), default=0)
    padded_batch_area = int(len(boxes) * max_h * max_w)
    padding_waste = 1.0 - crop_area_sum / float(max(1, padded_batch_area))
    tile_area = int(geometry["tile_input_area_px"])
    image_area = int(geometry["image_h"] * geometry["image_w"])
    return {
        "exec_crop_count": int(len(boxes)),
        "resnet_calls_per_camera": 1 if boxes else 0,
        "fpn_calls_per_camera": 1 if boxes else 0,
        "exec_crop_area_sum_px": crop_area_sum,
        "padded_batch_area_px": padded_batch_area,
        "max_batch_crop_w": int(max_w),
        "max_batch_crop_h": int(max_h),
        "batch_padding_waste_pct": 100.0 * padding_waste,
        "tile_input_area_px": tile_area,
        "tile_input_area_ratio_pct": 100.0 * tile_area / float(max(1, image_area)),
        "exec_area_ratio_pct": 100.0 * crop_area_sum / float(max(1, image_area)),
        "merge_expansion_ratio": crop_area_sum / float(max(1, tile_area)),
        "merge_extra_area_px": max(0, crop_area_sum - tile_area),
        "merge_extra_area_ratio_pct": 100.0 * max(0, crop_area_sum - tile_area) / float(max(1, image_area)),
        "adjacency_edges": int(plan.get("adjacency_edges", 0)),
        "adjacency_density": float(plan.get("adjacency_density", 0.0)),
        "component_count": int(plan.get("component_count", 0)),
        "bbox_overlap_merge_count": int(plan.get("bbox_overlap_merge_count", 0)),
        "raw_roi_count": int(geometry["raw_roi_count"]),
        "raw_roi_sum_area_px": int(geometry["raw_roi_sum_area_px"]),
        "raw_roi_union_area_px": int(geometry["raw_roi_union_area_px"]),
        "raw_roi_union_area_ratio_pct": float(geometry["raw_roi_union_area_ratio_pct"]),
        "raw_roi_sum_area_ratio_pct": float(geometry["raw_roi_sum_area_ratio_pct"]),
        "max_roi_w": int(geometry["max_roi_w"]),
        "max_roi_h": int(geometry["max_roi_h"]),
        "unique_tile_count": int(geometry["unique_tile_count"]),
        "roi_tile_hits": int(geometry["roi_tile_hits"]),
        "tile_dedup_savings": int(geometry["tile_dedup_savings"]),
        "unit_w_32": int(geometry["unit_w_32"]),
        "unit_h_32": int(geometry["unit_h_32"]),
        "unit_w": int(geometry["unit_w"]),
        "unit_h": int(geometry["unit_h"]),
        "grid_rows": int(geometry["grid_rows"]),
        "grid_cols": int(geometry["grid_cols"]),
        "pad_h": int(geometry["pad_h"]),
        "pad_w": int(geometry["pad_w"]),
    }


def camera_predecision_features(geometry: Dict) -> Dict[str, float]:
    """Cheap features for dispatch; no dense adjacency matrix is constructed."""
    cells = geometry["cells"]
    edges = estimate_shared_edge_count(cells)
    k = len(cells)
    n = int(geometry["raw_roi_count"])
    unit_area_mpix = float(geometry["unit_area_px"]) / 1e6
    tile_area_mpix = float(geometry["tile_input_area_px"]) / 1e6
    return {
        "raw_roi_count": float(n),
        "raw_roi_sum_area_mpix": float(geometry["raw_roi_sum_area_px"]) / 1e6,
        "raw_roi_union_area_mpix": float(geometry["raw_roi_union_area_px"]) / 1e6,
        "unique_tile_count": float(k),
        "tile_input_area_mpix": tile_area_mpix,
        "roi_tile_hits": float(geometry["roi_tile_hits"]),
        "tile_dedup_savings": float(geometry["tile_dedup_savings"]),
        "estimated_adjacency_edges": float(edges),
        "estimated_edge_density": float(edges / max(1, k * (k - 1) / 2)),
        "max_camera_unique_tiles": float(k),
        "grid_cell_count": float(geometry["grid_cell_count"]),
        "unit_width_norm": float(geometry["unit_w"]) / max(1, geometry["image_w"]),
        "unit_height_norm": float(geometry["unit_h"]) / max(1, geometry["image_h"]),
        "roi_area_cv": float(geometry["roi_area_cv"]),
        "roi_aspect_cv": float(geometry["roi_aspect_cv"]),
        "roi_center_spread": float(geometry["roi_center_spread"]),
        "tile_input_area_sq": tile_area_mpix * tile_area_mpix,
        "unique_tile_count_sq": float(k * k),
        "estimated_edges_sq": float(edges * edges),
    }


def aggregate_frame_features(camera_geometries: Sequence[Dict]) -> Dict[str, float]:
    per_cam = [camera_predecision_features(g) for g in camera_geometries]
    if not per_cam:
        return {name: 0.0 for name in FRAME_FEATURE_NAMES}
    return {
        "camera_count": float(len(per_cam)),
        "sum_raw_roi_count": float(sum(x["raw_roi_count"] for x in per_cam)),
        "sum_raw_roi_sum_area_mpix": float(sum(x["raw_roi_sum_area_mpix"] for x in per_cam)),
        "sum_raw_roi_union_area_mpix": float(sum(x["raw_roi_union_area_mpix"] for x in per_cam)),
        "sum_unique_tile_count": float(sum(x["unique_tile_count"] for x in per_cam)),
        "sum_tile_input_area_mpix": float(sum(x["tile_input_area_mpix"] for x in per_cam)),
        "sum_roi_tile_hits": float(sum(x["roi_tile_hits"] for x in per_cam)),
        "sum_tile_dedup_savings": float(sum(x["tile_dedup_savings"] for x in per_cam)),
        "sum_estimated_adjacency_edges": float(sum(x["estimated_adjacency_edges"] for x in per_cam)),
        "mean_estimated_edge_density": float(np.mean([x["estimated_edge_density"] for x in per_cam])),
        "max_camera_unique_tiles": float(max(x["max_camera_unique_tiles"] for x in per_cam)),
        "sum_grid_cell_count": float(sum(x["grid_cell_count"] for x in per_cam)),
        "mean_unit_width_norm": float(np.mean([x["unit_width_norm"] for x in per_cam])),
        "mean_unit_height_norm": float(np.mean([x["unit_height_norm"] for x in per_cam])),
        "mean_roi_area_cv": float(np.mean([x["roi_area_cv"] for x in per_cam])),
        "mean_roi_aspect_cv": float(np.mean([x["roi_aspect_cv"] for x in per_cam])),
        "mean_roi_center_spread": float(np.mean([x["roi_center_spread"] for x in per_cam])),
        "sum_tile_input_area_sq": float(sum(x["tile_input_area_sq"] for x in per_cam)),
        "sum_unique_tile_count_sq": float(sum(x["unique_tile_count_sq"] for x in per_cam)),
        "sum_estimated_edges_sq": float(sum(x["estimated_edges_sq"] for x in per_cam)),
    }


def fit_ridge_model(rows: Sequence[Dict], target: str, feature_names=FRAME_FEATURE_NAMES,
                    ridge: float = 1e-2) -> Dict:
    if len(rows) < 2:
        raise ValueError("At least two rows are required to fit {}".format(target))
    x = np.asarray([[float(r.get(k, 0.0)) for k in feature_names] for r in rows], dtype=np.float64)
    y = np.asarray([float(r[target]) for r in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "target": target,
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "ridge": float(ridge),
        "samples": int(len(rows)),
    }


def predict_ridge_model(model: Dict, features: Dict[str, float]) -> float:
    names = model["feature_names"]
    x = np.asarray([float(features.get(k, 0.0)) for k in names], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    z = (x - mean) / np.maximum(scale, 1e-12)
    return max(0.0, float(coef[0] + np.dot(coef[1:], z)))


def load_dispatch_profile(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    if profile.get("schema_version") != 3:
        raise ValueError("Expected ROI tile dispatch profile schema_version=3")
    return profile


def choose_roi_execution_mode(
    profile: Dict,
    camera_roi_boxes: Sequence[Sequence[Box]],
    image_h: int,
    image_w: int,
    current_mode: str | None = None,
    switch_margin: float = 0.03,
) -> Dict:
    """Choose one mode for all cameras using geometry-only predecision features.

    This computes tile mapping and O(K) shared-edge counts, but does NOT build
    the dense adjacency matrix or run connected components before deciding.
    """
    cfg = profile.get("benchmark_config", {})
    geometries = [
        build_fixed_tile_geometry(
            boxes, image_h, image_w,
            unit_align=int(cfg.get("unit_align", 32)),
            feature_align=int(cfg.get("feature_align", 64)),
        )
        for boxes in camera_roi_boxes
    ]
    features = aggregate_frame_features(geometries)
    costs = {}
    for mode in ("roi_batch", "adjacency_merge"):
        model = profile.get("total_time_models", {}).get(mode)
        costs[mode] = float("inf") if model is None else predict_ridge_model(model, features)
    if not all(np.isfinite(v) for v in costs.values()):
        raise RuntimeError("Profile is missing a fitted model for one or both ROI modes")
    best = min(costs, key=costs.get)
    if current_mode in costs and costs[current_mode] <= costs[best] * (1.0 + float(switch_margin)):
        best = current_mode
    return {
        "selected_mode": best,
        "predicted_costs_ms": costs,
        "predecision_features": features,
        "camera_geometries": geometries,
        "switch_margin": float(switch_margin),
    }
