#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile fixed-tile ROI-Batch vs Adjacency-Merge inside real UniAD E2E.

The benchmark is image-space and query-independent. Synthetic, controlled ROI
rectangles are placed on the *real normalized nuScenes camera images* entering
UniAD. Every ROI activates every intersecting fixed crop tile; overlapping ROIs
share the same tile IDs and are therefore deduplicated before either path.

Per camera, fixed tile width/height are derived independently from max ROI
width/height, rounded to 32 px and then to 64 px (64 is still 32-aligned and is
required to keep the FPN stride-64 feature lattice spatially aligned).

ROI-Batch:
  all unique fixed tiles -> one [K,3,Htile,Wtile] batch -> one ResNet + one FPN.

Adjacency-Merge:
  dense adjacency matrix over unique tiles; only edge-sharing grid neighbours
  connect -> connected components -> component bounding rectangles -> merge
  any overlapping rectangles to prevent duplicate image-pixel computation ->
  pad variable rectangles into one batch -> one ResNet + one FPN.

Both paths are executed from the patched extract_img_feat called by the actual
UniAD.forward(return_loss=False) -> forward_test -> simple_test_track ->
get_bevs path. BEVFormer and all downstream E2E heads continue to execute.

This is a latency experiment, not an accuracy experiment. Features outside
selected crop regions are zero-filled, and crop boundaries lack full-image
receptive-field context. Do not use these outputs to claim model accuracy.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
import os
import random
import sys
import time
import types
from collections import defaultdict

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from roi_tile_dispatch_runtime import (
    COMPONENT_TARGETS,
    FRAME_FEATURE_NAMES,
    aggregate_frame_features,
    build_execution_plan,
    build_fixed_tile_geometry,
    fit_ridge_model,
    predict_ridge_model,
    summarize_execution_geometry,
)

FPN_STRIDES = (8, 16, 32, 64)
MODES = ("roi_batch", "adjacency_merge")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-tile ROI-Batch vs Adjacency-Merge profiler inside UniAD E2E on Thor"
    )
    p.add_argument("config", help="UniAD config, e.g. projects/configs/stage2_e2e/base_e2e_tf32.py")
    p.add_argument("checkpoint", help="UniAD checkpoint path")
    p.add_argument("--out-dir", default="work_dirs/roi_tile_dispatch_thor")
    p.add_argument("--roi-counts", default="2,4,8,16")
    p.add_argument("--area-fractions", default="0.05,0.10,0.20",
                   help="Target SUM of raw ROI rectangle areas / one camera image area")
    p.add_argument("--layouts", default="spread,cluster,chain",
                   help="spread,cluster,chain,random")
    p.add_argument("--aspect-ratios", default="1.0",
                   help="ROI width/height ratios; can use e.g. 0.5,1.0,2.0")
    p.add_argument("--size-variation", type=float, default=0.25)
    p.add_argument("--min-roi-side", type=int, default=64)
    p.add_argument("--frames-per-workload", type=int, default=3)
    p.add_argument("--warmup-frames", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--unit-align", type=int, default=32,
                   help="Requested fixed tile alignment; default 32 px")
    p.add_argument("--feature-align", type=int, default=64,
                   help="Coarsest FPN stride; default 64 px, effective tile size remains 32-aligned")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    p.add_argument("--max-workloads", type=int, default=0,
                   help="Limit workload count for smoke tests; 0 means all")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    return p.parse_args()


def parse_csv_values(text, cast):
    values = [cast(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_project_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)
    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", None)
        if plugin_dir:
            importlib.import_module(plugin_dir.rstrip("/").replace("/", "."))


def round_up(value: float, multiple: int) -> int:
    return max(multiple, int(math.ceil(float(value) / multiple) * multiple))


def _layout_centers(n, image_h, image_w, base_h, base_w, layout, rng):
    centers = []
    if layout == "spread":
        cols = max(1, int(math.ceil(math.sqrt(n * image_w / float(image_h)))))
        rows = int(math.ceil(n / float(cols)))
        for i in range(n):
            r, c = divmod(i, cols)
            centers.append(((c + 0.5) * image_w / cols,
                            (r + 0.5) * image_h / rows))
    elif layout == "cluster":
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / float(cols)))
        sx = max(8.0, min(base_w * 0.48, image_w / max(2.0, cols + 0.5)))
        sy = max(8.0, min(base_h * 0.48, image_h / max(2.0, rows + 0.5)))
        for i in range(n):
            r, c = divmod(i, cols)
            centers.append((image_w / 2.0 + (c - (cols - 1) / 2.0) * sx,
                            image_h / 2.0 + (r - (rows - 1) / 2.0) * sy))
    elif layout == "chain":
        sx = max(8.0, base_w * 0.95)
        sy = max(8.0, base_h * 0.95)
        cols = max(1, min(n, int(max(1.0, image_w - base_w) // sx) + 1))
        for i in range(n):
            r, c = divmod(i, cols)
            if r % 2:
                c = cols - 1 - c
            centers.append((base_w / 2.0 + c * sx, base_h / 2.0 + r * sy))
    elif layout == "random":
        centers = [(rng.uniform(0, image_w), rng.uniform(0, image_h)) for _ in range(n)]
    else:
        raise ValueError("Unknown layout: {}".format(layout))
    return centers


def generate_roi_boxes(
    image_h, image_w, n_rois, area_fraction, aspect_ratio, layout, seed,
    size_variation=0.25, min_roi_side=64,
):
    """Generate deterministic image-space ROI boxes; no BEV/Query projection."""
    if n_rois <= 0:
        raise ValueError("n_rois must be positive")
    if not (0.0 < area_fraction <= 1.0):
        raise ValueError("area_fraction must be in (0,1]")
    rng = np.random.RandomState(int(seed) % (2**32 - 1))
    target_each = area_fraction * image_h * image_w / float(n_rois)
    factors = rng.uniform(max(0.05, 1.0 - size_variation), 1.0 + size_variation, n_rois)
    factors = factors / max(float(factors.mean()), 1e-9)
    centers = _layout_centers(
        n_rois, image_h, image_w,
        math.sqrt(target_each / aspect_ratio),
        math.sqrt(target_each * aspect_ratio), layout, rng,
    )
    boxes = []
    for (cx, cy), factor in zip(centers, factors):
        ar = float(aspect_ratio) * float(np.exp(rng.uniform(-0.18, 0.18)))
        area = target_each * float(factor)
        raw_w = math.sqrt(area * ar)
        raw_h = math.sqrt(area / ar)
        w = min(image_w, max(min_roi_side, int(round(raw_w))))
        h = min(image_h, max(min_roi_side, int(round(raw_h))))
        x0 = int(round(cx - w / 2.0))
        y0 = int(round(cy - h / 2.0))
        x0 = min(max(0, x0), image_w - w)
        y0 = min(max(0, y0), image_h - h)
        boxes.append((x0, y0, x0 + w, y0 + h))
    return boxes


def build_workloads(args):
    counts = parse_csv_values(args.roi_counts, int)
    areas = parse_csv_values(args.area_fractions, float)
    layouts = parse_csv_values(args.layouts, str)
    aspects = parse_csv_values(args.aspect_ratios, float)
    if any(v <= 0 for v in counts):
        raise ValueError("ROI counts must be positive")
    if any(not (0 < v <= 1) for v in areas):
        raise ValueError("area fractions must be in (0,1]")
    if any(v <= 0 for v in aspects):
        raise ValueError("aspect ratios must be positive")
    valid = {"spread", "cluster", "chain", "random"}
    if any(v not in valid for v in layouts):
        raise ValueError("layouts must be selected from {}".format(sorted(valid)))
    workloads = []
    for n, area, layout, aspect in itertools.product(counts, areas, layouts, aspects):
        wid = "n{}_a{}_{}_ar{}".format(
            n, str(area).replace(".", "p"), layout, str(aspect).replace(".", "p")
        )
        workloads.append({
            "workload_id": wid,
            "roi_count_target": int(n),
            "area_fraction_target": float(area),
            "layout": layout,
            "aspect_ratio_target": float(aspect),
        })
    if args.max_workloads > 0:
        workloads = workloads[:args.max_workloads]
    return workloads


def normalize_backbone_outputs(feats):
    if isinstance(feats, dict):
        feats = list(feats.values())
    if not isinstance(feats, (list, tuple)):
        feats = [feats]
    return list(feats)


def cuda_event_pair():
    return (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))


def cuda_event_ms(pair):
    return float(pair[0].elapsed_time(pair[1]))


def _extract_raw_boxes(workload, n_cam, image_h, image_w, seed, size_variation, min_roi_side):
    boxes_by_camera = []
    for cam in range(n_cam):
        boxes_by_camera.append(generate_roi_boxes(
            image_h, image_w,
            workload["roi_count_target"], workload["area_fraction_target"],
            workload["aspect_ratio_target"], workload["layout"],
            seed + cam * 100003,
            size_variation=size_variation,
            min_roi_side=min_roi_side,
        ))
    return boxes_by_camera


def _make_crop_batch(camera_image, crop_boxes):
    """Create one padded batch from crop boxes; returns tensor and geometry."""
    if not crop_boxes:
        raise RuntimeError("No crop boxes were generated")
    sizes = [(b[3] - b[1], b[2] - b[0]) for b in crop_boxes]
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    channels = int(camera_image.shape[0])
    batch = camera_image.new_zeros((len(crop_boxes), channels, max_h, max_w))
    for i, (x0, y0, x1, y1) in enumerate(crop_boxes):
        crop = camera_image[:, y0:y1, x0:x1]
        h, w = crop.shape[-2:]
        batch[i, :, :h, :w].copy_(crop)
    crop_area_sum = sum(h * w for h, w in sizes)
    padded_area = len(crop_boxes) * max_h * max_w
    return batch.contiguous(), {
        "max_batch_h": int(max_h),
        "max_batch_w": int(max_w),
        "crop_area_sum_px": int(crop_area_sum),
        "padded_batch_area_px": int(padded_area),
        "padding_waste_pct": 100.0 * (padded_area - crop_area_sum) / float(max(1, padded_area)),
        "batch_size": int(len(crop_boxes)),
    }


def _make_padded_camera_image(camera_image, pad_h, pad_w):
    channels, h, w = camera_image.shape
    if pad_h == h and pad_w == w:
        return camera_image
    padded = camera_image.new_zeros((channels, pad_h, pad_w))
    padded[:, :h, :w].copy_(camera_image)
    return padded


def _init_full_feature_canvases(feats, batch_size, n_cam, image_h, image_w):
    canvases = []
    for feat, stride in zip(feats, FPN_STRIDES):
        full_h = int(math.ceil(float(image_h) / stride))
        full_w = int(math.ceil(float(image_w) / stride))
        canvases.append(feat.new_zeros((batch_size, n_cam, feat.shape[1], full_h, full_w)))
    return canvases


def _scatter_crop_features(full_features, feats, crop_boxes, camera_idx, image_h, image_w):
    for crop_idx, (x0, y0, x1, y1) in enumerate(crop_boxes):
        for level, stride in enumerate(FPN_STRIDES):
            feat = feats[level]
            dst_y0, dst_x0 = y0 // stride, x0 // stride
            valid_h = (y1 - y0) // stride
            valid_w = (x1 - x0) // stride
            dst_h = min(valid_h, full_features[level].shape[-2] - dst_y0)
            dst_w = min(valid_w, full_features[level].shape[-1] - dst_x0)
            if dst_h <= 0 or dst_w <= 0:
                continue
            full_features[level][0, camera_idx, :, dst_y0:dst_y0 + dst_h, dst_x0:dst_x0 + dst_w].copy_(
                feat[crop_idx, :, :dst_h, :dst_w]
            )


def install_runtime_patch(model, args):
    model._roi_tile_enabled = True
    model._roi_tile_mode = "roi_batch"
    model._roi_tile_workload = None
    model._roi_tile_box_cache = {}
    model._roi_tile_last_camera_rows = []
    model._roi_tile_last_context = {}

    def extract_img_feat_profiled(self, img, len_queue=None):
        if (not getattr(self, "_roi_tile_enabled", False) or img is None or len_queue is not None):
            return self._roi_tile_original_extract(img, len_queue)
        if img.dim() != 5:
            raise ValueError("Expected img [B,N,C,H,W], got {}".format(tuple(img.shape)))
        batch_size, n_cam, channels, image_h, image_w = img.shape
        if batch_size != 1:
            raise ValueError("This sequential E2E profiler requires batch_size=1")
        if len(FPN_STRIDES) != 4:
            raise RuntimeError("Unexpected FPN stride configuration")
        workload = self._roi_tile_workload
        if workload is None:
            raise RuntimeError("No ROI workload assigned before UniAD E2E forward")

        # Keep the original UniAD preprocessing order: GridMask is applied to
        # all six full images before any image-space crop is extracted.
        torch.cuda.synchronize()
        gridmask_t0 = time.perf_counter()
        flat_img = img.reshape(batch_size * n_cam, channels, image_h, image_w)
        if getattr(self, "use_grid_mask", False):
            flat_img = self.grid_mask(flat_img)
        img_full = flat_img.reshape(batch_size, n_cam, channels, image_h, image_w)
        torch.cuda.synchronize()
        gridmask_ms = (time.perf_counter() - gridmask_t0) * 1000.0

        cache_key = (workload["workload_id"], n_cam, image_h, image_w)
        box_generation_ms = 0.0
        if cache_key not in self._roi_tile_box_cache:
            t0 = time.perf_counter()
            self._roi_tile_box_cache[cache_key] = _extract_raw_boxes(
                workload, n_cam, image_h, image_w, args.seed,
                args.size_variation, args.min_roi_side,
            )
            box_generation_ms = (time.perf_counter() - t0) * 1000.0
        raw_boxes_by_camera = self._roi_tile_box_cache[cache_key]

        torch.cuda.synchronize()
        roi_block_t0 = time.perf_counter()
        camera_rows = []
        full_features = None
        canvas_event = None
        frame_geometry = []
        frame_component_sums = {name: 0.0 for name in COMPONENT_TARGETS}
        frame_component_sums.update({
            "tile_mapping_cpu_ms": 0.0,
            "adjacency_matrix_cpu_ms": 0.0,
            "connected_components_cpu_ms": 0.0,
            "bbox_overlap_merge_cpu_ms": 0.0,
            "tile_pad_gpu_ms": 0.0,
            "crop_pack_host_ms": 0.0,
            "crop_pack_gpu_ms": 0.0,
            "resnet_gpu_ms": 0.0,
            "fpn_gpu_ms": 0.0,
            "scatter_gpu_ms": 0.0,
        })
        frame_exec_area_px = 0
        frame_tile_area_px = 0
        frame_raw_union_area_px = 0
        frame_raw_sum_area_px = 0
        frame_unique_tile_count = 0
        frame_exec_crop_count = 0
        frame_adjacency_edges = 0
        frame_merge_extra_area_px = 0
        frame_padded_batch_area_px = 0
        frame_padding_waste_weighted_num = 0.0
        frame_padding_waste_weighted_den = 0.0

        for cam in range(n_cam):
            raw_boxes = raw_boxes_by_camera[cam]

            # Common ROI -> fixed-tile mapping. This stage is identical for A/B.
            mapping_t0 = time.perf_counter()
            geometry = build_fixed_tile_geometry(
                raw_boxes, image_h, image_w,
                unit_align=args.unit_align,
                feature_align=args.feature_align,
            )
            mapping_ms = (time.perf_counter() - mapping_t0) * 1000.0
            frame_geometry.append(geometry)

            # Strategy-specific graph construction / component merge.
            plan = build_execution_plan(geometry, self._roi_tile_mode)
            exec_boxes = plan["exec_boxes"]
            geom_summary = summarize_execution_geometry(geometry, plan)

            # The padded source canvas allows the last fixed tile to retain its
            # exact unit dimensions when image H/W are not multiples of tile H/W.
            pad_event = cuda_event_pair()
            pad_host_t0 = time.perf_counter()
            pad_event[0].record()
            padded_camera = _make_padded_camera_image(
                img_full[0, cam], geometry["pad_h"], geometry["pad_w"]
            )
            pad_event[1].record()
            pad_host_ms = (time.perf_counter() - pad_host_t0) * 1000.0

            pack_event = cuda_event_pair()
            pack_host_t0 = time.perf_counter()
            pack_event[0].record()
            roi_batch, pack_meta = _make_crop_batch(padded_camera, exec_boxes)
            if pack_meta["crop_area_sum_px"] != geom_summary["exec_crop_area_sum_px"]:
                raise RuntimeError("Crop pack area does not match planned execution area")
            if pack_meta["padded_batch_area_px"] != geom_summary["padded_batch_area_px"]:
                raise RuntimeError("Crop pack padding area does not match geometry summary")
            pack_event[1].record()
            pack_host_ms = (time.perf_counter() - pack_host_t0) * 1000.0

            # Exactly one ResNet call and one FPN call per camera for both modes.
            resnet_event = cuda_event_pair()
            resnet_event[0].record()
            backbone_feats = normalize_backbone_outputs(self.img_backbone(roi_batch))
            resnet_event[1].record()

            if self.with_img_neck:
                fpn_event = cuda_event_pair()
                fpn_event[0].record()
                neck_feats = normalize_backbone_outputs(self.img_neck(backbone_feats))
                fpn_event[1].record()
            else:
                fpn_event = None
                neck_feats = backbone_feats

            if len(neck_feats) != len(FPN_STRIDES):
                raise RuntimeError(
                    "Expected {} FPN levels at strides {}, got {}".format(
                        len(FPN_STRIDES), FPN_STRIDES, len(neck_feats)
                    )
                )
            for feat, stride in zip(neck_feats, FPN_STRIDES):
                expected_h = roi_batch.shape[-2] // stride
                expected_w = roi_batch.shape[-1] // stride
                if feat.shape[-2] != expected_h or feat.shape[-1] != expected_w:
                    raise RuntimeError(
                        "FPN feature shape {} does not match crop input {} at stride {}. "
                        "Check the effective tile alignment.".format(
                            tuple(feat.shape[-2:]), tuple(roi_batch.shape[-2:]), stride
                        )
                    )

            if full_features is None:
                canvas_event = cuda_event_pair()
                canvas_event[0].record()
                full_features = _init_full_feature_canvases(
                    neck_feats, batch_size, n_cam, image_h, image_w
                )
                canvas_event[1].record()

            scatter_event = cuda_event_pair()
            scatter_event[0].record()
            _scatter_crop_features(
                full_features, neck_feats, exec_boxes, cam, image_h, image_w
            )
            scatter_event[1].record()

            # Per-camera event records are resolved after the final sync below.
            row = {
                "mode": self._roi_tile_mode,
                "workload_id": workload["workload_id"],
                "roi_count_target": workload["roi_count_target"],
                "area_fraction_target": workload["area_fraction_target"],
                "layout": workload["layout"],
                "aspect_ratio_target": workload["aspect_ratio_target"],
                "camera": int(cam),
                "image_h": int(image_h),
                "image_w": int(image_w),
                "tile_mapping_cpu_ms": float(mapping_ms),
                "adjacency_matrix_cpu_ms": float(plan["adjacency_matrix_cpu_ms"]),
                "connected_components_cpu_ms": float(plan["connected_components_cpu_ms"]),
                "bbox_overlap_merge_cpu_ms": float(plan["bbox_overlap_merge_cpu_ms"]),
                "tile_pad_gpu_event": pad_event,
                "tile_pad_host_ms": float(pad_host_ms),
                "crop_pack_gpu_event": pack_event,
                "crop_pack_host_ms": float(pack_host_ms),
                "resnet_gpu_event": resnet_event,
                "fpn_gpu_event": fpn_event,
                "scatter_gpu_event": scatter_event,
                "crop_batch_shape": json.dumps(list(roi_batch.shape)),
                "batch_size": int(pack_meta["batch_size"]),
                "max_batch_crop_h": int(pack_meta["max_batch_h"]),
                "max_batch_crop_w": int(pack_meta["max_batch_w"]),
                "raw_boxes_json": geometry["raw_boxes_json"],
                "active_cells_json": geometry["cells_json"],
                "exec_boxes_json": json.dumps(exec_boxes),
                **geom_summary,
            }
            camera_rows.append(row)

            frame_component_sums["tile_mapping_cpu_ms"] += mapping_ms
            frame_component_sums["adjacency_matrix_cpu_ms"] += plan["adjacency_matrix_cpu_ms"]
            frame_component_sums["connected_components_cpu_ms"] += plan["connected_components_cpu_ms"]
            frame_component_sums["bbox_overlap_merge_cpu_ms"] += plan["bbox_overlap_merge_cpu_ms"]
            frame_component_sums["tile_pad_host_ms"] += pad_host_ms
            frame_component_sums["crop_pack_host_ms"] += pack_host_ms
            frame_exec_area_px += geom_summary["exec_crop_area_sum_px"]
            frame_tile_area_px += geom_summary["tile_input_area_px"]
            frame_raw_union_area_px += geometry["raw_roi_union_area_px"]
            frame_raw_sum_area_px += geometry["raw_roi_sum_area_px"]
            frame_unique_tile_count += geometry["unique_tile_count"]
            frame_exec_crop_count += geom_summary["exec_crop_count"]
            frame_adjacency_edges += plan["adjacency_edges"]
            frame_merge_extra_area_px += geom_summary["merge_extra_area_px"]
            frame_padded_batch_area_px += geom_summary["padded_batch_area_px"]
            frame_padding_waste_weighted_num += (
                geom_summary["batch_padding_waste_pct"] * geom_summary["padded_batch_area_px"]
            )
            frame_padding_waste_weighted_den += geom_summary["padded_batch_area_px"]

        if full_features is None:
            raise RuntimeError("No image features were generated")

        torch.cuda.synchronize()
        roi_block_wall_ms = (time.perf_counter() - roi_block_t0) * 1000.0
        gridmask_done_ms = gridmask_ms
        frame_component_sums["feature_canvas_init_gpu_ms"] += (
            cuda_event_ms(canvas_event) if canvas_event is not None else 0.0
        )

        for row in camera_rows:
            row["tile_pad_gpu_ms"] = cuda_event_ms(row["tile_pad_gpu_event"])
            row["crop_pack_gpu_ms"] = cuda_event_ms(row["crop_pack_gpu_event"])
            row["resnet_gpu_ms"] = cuda_event_ms(row["resnet_gpu_event"])
            row["fpn_gpu_ms"] = cuda_event_ms(row["fpn_gpu_event"]) if row["fpn_gpu_event"] else 0.0
            row["scatter_gpu_ms"] = cuda_event_ms(row["scatter_gpu_event"])
            for key in ("tile_pad_gpu_event", "crop_pack_gpu_event", "resnet_gpu_event", "fpn_gpu_event", "scatter_gpu_event"):
                row.pop(key, None)
            row["component_sum_diagnostic_ms"] = sum(float(row.get(k, 0.0)) for k in COMPONENT_TARGETS)
            frame_component_sums["tile_pad_gpu_ms"] += row["tile_pad_gpu_ms"]
            frame_component_sums["crop_pack_gpu_ms"] += row["crop_pack_gpu_ms"]
            frame_component_sums["resnet_gpu_ms"] += row["resnet_gpu_ms"]
            frame_component_sums["fpn_gpu_ms"] += row["fpn_gpu_ms"]
            frame_component_sums["scatter_gpu_ms"] += row["scatter_gpu_ms"]

        frame_total_image_area = float(n_cam * image_h * image_w)
        frame_features = aggregate_frame_features(frame_geometry)
        frame_row = {
            "mode": self._roi_tile_mode,
            "workload_id": workload["workload_id"],
            "roi_count_target": workload["roi_count_target"],
            "area_fraction_target": workload["area_fraction_target"],
            "layout": workload["layout"],
            "aspect_ratio_target": workload["aspect_ratio_target"],
            "camera_count": int(n_cam),
            "image_h": int(image_h),
            "image_w": int(image_w),
            "raw_roi_union_area_ratio_pct_all_cameras": 100.0 * frame_raw_union_area_px / frame_total_image_area,
            "raw_roi_sum_area_ratio_pct_all_cameras": 100.0 * frame_raw_sum_area_px / frame_total_image_area,
            "unique_tile_input_area_ratio_pct_all_cameras": 100.0 * frame_tile_area_px / frame_total_image_area,
            "executed_crop_area_ratio_pct_all_cameras": 100.0 * frame_exec_area_px / frame_total_image_area,
            "merge_extra_area_ratio_pct_all_cameras": 100.0 * frame_merge_extra_area_px / frame_total_image_area,
            "unique_tile_count_all_cameras": int(frame_unique_tile_count),
            "exec_crop_count_all_cameras": int(frame_exec_crop_count),
            "resnet_calls_all_cameras": int(sum(r["resnet_calls_per_camera"] for r in camera_rows)),
            "fpn_calls_all_cameras": int(sum(r["fpn_calls_per_camera"] for r in camera_rows)),
            "adjacency_edges_all_cameras": int(frame_adjacency_edges),
            "padded_batch_area_all_cameras_px": int(frame_padded_batch_area_px),
            "batch_padding_waste_pct_weighted": (
                frame_padding_waste_weighted_num / max(1.0, frame_padding_waste_weighted_den)
            ),
            "tile_mapping_cpu_ms": frame_component_sums["tile_mapping_cpu_ms"],
            "adjacency_matrix_cpu_ms": frame_component_sums["adjacency_matrix_cpu_ms"],
            "connected_components_cpu_ms": frame_component_sums["connected_components_cpu_ms"],
            "bbox_overlap_merge_cpu_ms": frame_component_sums["bbox_overlap_merge_cpu_ms"],
            "tile_pad_host_ms": frame_component_sums["tile_pad_host_ms"],
            "tile_pad_gpu_ms": frame_component_sums["tile_pad_gpu_ms"],
            "feature_canvas_init_gpu_ms": frame_component_sums["feature_canvas_init_gpu_ms"],
            "crop_pack_host_ms": frame_component_sums["crop_pack_host_ms"],
            "crop_pack_gpu_ms": frame_component_sums["crop_pack_gpu_ms"],
            "resnet_gpu_ms": frame_component_sums["resnet_gpu_ms"],
            "fpn_gpu_ms": frame_component_sums["fpn_gpu_ms"],
            "scatter_gpu_ms": frame_component_sums["scatter_gpu_ms"],
            "component_cpu_sum_ms": sum(frame_component_sums[k] for k in (
                "tile_mapping_cpu_ms", "adjacency_matrix_cpu_ms", "connected_components_cpu_ms",
                "bbox_overlap_merge_cpu_ms", "tile_pad_host_ms", "crop_pack_host_ms")),
            "component_gpu_sum_ms": sum(frame_component_sums[k] for k in (
                "tile_pad_gpu_ms", "feature_canvas_init_gpu_ms", "crop_pack_gpu_ms",
                "resnet_gpu_ms", "fpn_gpu_ms", "scatter_gpu_ms")),
            "roi_block_wall_ms": float(roi_block_wall_ms),
            "gridmask_ms": float(gridmask_done_ms),
            "box_generation_ms_excluded": float(box_generation_ms),
            **frame_features,
        }
        frame_row["component_sum_diagnostic_ms"] = (
            frame_row["component_cpu_sum_ms"] + frame_row["component_gpu_sum_ms"]
        )
        component_denom = max(1e-9, frame_row["component_sum_diagnostic_ms"])
        frame_row["adjacency_overhead_share_pct"] = 100.0 * sum(
            frame_component_sums[k] for k in (
                "adjacency_matrix_cpu_ms", "connected_components_cpu_ms", "bbox_overlap_merge_cpu_ms")
        ) / component_denom
        frame_row["resnet_fpn_share_pct"] = 100.0 * (
            frame_component_sums["resnet_gpu_ms"] + frame_component_sums["fpn_gpu_ms"]
        ) / component_denom
        frame_row["prepare_share_pct"] = 100.0 * sum(
            frame_component_sums[k] for k in (
                "tile_mapping_cpu_ms", "tile_pad_host_ms", "tile_pad_gpu_ms",
                "crop_pack_host_ms", "crop_pack_gpu_ms")
        ) / component_denom
        frame_row["scatter_share_pct"] = 100.0 * frame_component_sums["scatter_gpu_ms"] / component_denom
        frame_row["component_sum_to_wall_ratio"] = (
            frame_row["component_sum_diagnostic_ms"] / max(1e-9, frame_row["roi_block_wall_ms"])
        )
        self._roi_tile_last_camera_rows = camera_rows
        self._roi_tile_last_context = frame_row
        return full_features

    model._roi_tile_original_extract = model.extract_img_feat
    model.extract_img_feat = types.MethodType(extract_img_feat_profiled, model)


def reset_temporal_state(model):
    model.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    for name in ("prev_bev", "scene_token", "timestamp", "l2g_r_mat", "l2g_t", "test_track_instances"):
        if hasattr(model, name):
            setattr(model, name, None)


def save_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("\n")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(v) for k, v in row.items()})


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _get_first_meta(data):
    value = data.get("img_metas") if isinstance(data, dict) else None
    while True:
        if hasattr(value, "data"):
            value = value.data
            continue
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
            continue
        break
    return value if isinstance(value, dict) else {}


def _split_workloads(frame_rows, seed, holdout_fraction):
    workload_ids = sorted({r["workload_id"] for r in frame_rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(workload_ids)
    if len(workload_ids) < 3:
        return list(frame_rows), []
    n_hold = max(1, int(round(len(workload_ids) * holdout_fraction)))
    n_hold = min(n_hold, len(workload_ids) - 2)
    hold_ids = set(workload_ids[:n_hold])
    train = [r for r in frame_rows if r["workload_id"] not in hold_ids]
    hold = [r for r in frame_rows if r["workload_id"] in hold_ids]
    return train, hold


def _fit_models_by_mode(rows, targets, ridge):
    models = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        models[mode] = {}
        for target in targets:
            valid = [r for r in mode_rows if target in r and np.isfinite(float(r[target]))]
            if len(valid) >= 3:
                models[mode][target] = fit_ridge_model(valid, target, FRAME_FEATURE_NAMES, ridge)
    return models


def _evaluate_holdout(rows, models):
    by_key = defaultdict(dict)
    for row in rows:
        mode = row["mode"]
        if mode not in models or "roi_block_wall_ms" not in models[mode]:
            continue
        pred = predict_ridge_model(models[mode]["roi_block_wall_ms"], row)
        key = (row["workload_id"], row["frame_idx"], row["sample_idx"])
        by_key[key][mode] = (float(row["roi_block_wall_ms"]), pred)
    paired = []
    correct = 0
    regret = []
    abs_errors = []
    for key, values in by_key.items():
        if not all(m in values for m in MODES):
            continue
        a_actual, a_pred = values["roi_batch"]
        b_actual, b_pred = values["adjacency_merge"]
        actual = "roi_batch" if a_actual <= b_actual else "adjacency_merge"
        selected = "roi_batch" if a_pred <= b_pred else "adjacency_merge"
        correct += int(actual == selected)
        selected_actual = a_actual if selected == "roi_batch" else b_actual
        regret.append(selected_actual - min(a_actual, b_actual))
        abs_errors.extend([abs(a_actual - a_pred), abs(b_actual - b_pred)])
        paired.append({
            "workload_id": key[0], "frame_idx": key[1], "sample_idx": key[2],
            "roi_batch_actual_ms": a_actual,
            "adjacency_merge_actual_ms": b_actual,
            "roi_batch_predicted_ms": a_pred,
            "adjacency_merge_predicted_ms": b_pred,
            "actual_winner": actual,
            "predicted_winner": selected,
            "decision_correct": int(actual == selected),
            "selection_regret_ms": selected_actual - min(a_actual, b_actual),
        })
    metrics = {
        "paired_holdout_samples": len(paired),
        "decision_accuracy": (correct / len(paired)) if paired else None,
        "mean_selection_regret_ms": float(np.mean(regret)) if regret else None,
        "p90_selection_regret_ms": float(np.percentile(regret, 90)) if regret else None,
        "total_time_prediction_mae_ms": float(np.mean(abs_errors)) if abs_errors else None,
        "note": "Holdout is split by workload_id to avoid putting repeated frames of the same ROI geometry in both train and test.",
    }
    return paired, metrics


def _make_paired_rows(frame_rows):
    groups = defaultdict(dict)
    for row in frame_rows:
        key = (row["workload_id"], row["frame_idx"], row["sample_idx"])
        groups[key][row["mode"]] = row
    paired = []
    for key, modes in groups.items():
        if not all(m in modes for m in MODES):
            continue
        a, b = modes["roi_batch"], modes["adjacency_merge"]
        ta, tb = float(a["roi_block_wall_ms"]), float(b["roi_block_wall_ms"])
        paired.append({
            "workload_id": key[0],
            "frame_idx": key[1],
            "sample_idx": key[2],
            "scene_token": a.get("scene_token", ""),
            "raw_roi_union_area_ratio_pct": a["raw_roi_union_area_ratio_pct_all_cameras"],
            "unique_tile_input_area_ratio_pct": a["unique_tile_input_area_ratio_pct_all_cameras"],
            "roi_batch_executed_area_ratio_pct": a["executed_crop_area_ratio_pct_all_cameras"],
            "adjacency_merge_executed_area_ratio_pct": b["executed_crop_area_ratio_pct_all_cameras"],
            "roi_batch_wall_ms": ta,
            "adjacency_merge_wall_ms": tb,
            "adjacency_minus_batch_ms": tb - ta,
            "actual_winner": "roi_batch" if ta <= tb else "adjacency_merge",
            "roi_batch_e2e_ms": a["e2e_forward_ms"],
            "adjacency_merge_e2e_ms": b["e2e_forward_ms"],
        })
    return paired


def _summarize(frame_rows):
    groups = defaultdict(list)
    for row in frame_rows:
        key = (row["mode"], row["roi_count_target"], row["area_fraction_target"], row["layout"], row["aspect_ratio_target"])
        groups[key].append(row)
    targets = list(COMPONENT_TARGETS) + ["roi_block_wall_ms", "e2e_forward_ms"]
    output = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        out = {
            "mode": key[0], "roi_count_target": key[1], "area_fraction_target": key[2],
            "layout": key[3], "aspect_ratio_target": key[4], "samples": len(values),
        }
        for target in targets:
            arr = np.asarray([float(r.get(target, 0.0)) for r in values], dtype=np.float64)
            out[target + "_mean"] = float(arr.mean())
            out[target + "_median"] = float(np.median(arr))
            out[target + "_p90"] = float(np.percentile(arr, 90))
        for field in ("raw_roi_union_area_ratio_pct_all_cameras", "unique_tile_input_area_ratio_pct_all_cameras",
                      "executed_crop_area_ratio_pct_all_cameras", "merge_extra_area_ratio_pct_all_cameras",
                      "batch_padding_waste_pct_weighted", "unique_tile_count_all_cameras", "exec_crop_count_all_cameras",
                      "component_cpu_sum_ms", "component_gpu_sum_ms", "component_sum_diagnostic_ms",
                      "adjacency_overhead_share_pct", "resnet_fpn_share_pct", "prepare_share_pct",
                      "scatter_share_pct", "component_sum_to_wall_ratio"):
            arr = np.asarray([float(r.get(field, 0.0)) for r in values], dtype=np.float64)
            out[field + "_mean"] = float(arr.mean())
        output.append(out)
    return output


def _save_plots(frame_rows, paired_rows, out_dir):
    if not frame_rows:
        return
    # Plot 1: whole six-camera ROI block wall time vs true raw ROI union coverage.
    plt.figure(figsize=(8.5, 5.5))
    for mode, marker in (("roi_batch", "o"), ("adjacency_merge", "^")):
        rows = [r for r in frame_rows if r["mode"] == mode]
        xs = np.asarray([r["raw_roi_union_area_ratio_pct_all_cameras"] for r in rows], dtype=float)
        ys = np.asarray([r["roi_block_wall_ms"] for r in rows], dtype=float)
        plt.scatter(xs, ys, marker=marker, alpha=0.55, label=mode)
    plt.xlabel("Raw ROI union coverage / all camera pixels (%)")
    plt.ylabel("Six-camera ROI execution wall time (ms)")
    plt.title("Thor fixed-tile ROI execution latency vs raw ROI coverage")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roi_wall_vs_raw_union_area.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(8.5, 5.5))
    for mode, marker in (("roi_batch", "o"), ("adjacency_merge", "^")):
        rows = [r for r in frame_rows if r["mode"] == mode]
        xs = np.asarray([r["unique_tile_input_area_ratio_pct_all_cameras"] for r in rows], dtype=float)
        ys = np.asarray([r["roi_block_wall_ms"] for r in rows], dtype=float)
        plt.scatter(xs, ys, marker=marker, alpha=0.55, label=mode)
    plt.xlabel("Unique fixed-tile input area / all camera pixels (%)")
    plt.ylabel("Six-camera ROI execution wall time (ms)")
    plt.title("Thor ROI execution latency vs fixed-tile input area")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roi_wall_vs_tile_input_area.png"), dpi=220)
    plt.close()

    # Plot 2: component decomposition vs actual fixed-tile input area.
    components = [
        ("tile_mapping_cpu_ms", "ROI-to-tile mapping CPU"),
        ("adjacency_matrix_cpu_ms", "Adjacency matrix CPU"),
        ("connected_components_cpu_ms", "Connected components CPU"),
        ("bbox_overlap_merge_cpu_ms", "Overlapping bbox merge CPU"),
        ("tile_pad_host_ms", "Tile canvas pad host"),
        ("tile_pad_gpu_ms", "Tile canvas pad GPU"),
        ("crop_pack_host_ms", "Crop/pack host"),
        ("crop_pack_gpu_ms", "Crop/pack GPU"),
        ("resnet_gpu_ms", "ResNet GPU"),
        ("fpn_gpu_ms", "FPN GPU"),
        ("scatter_gpu_ms", "Feature scatter GPU"),
        ("feature_canvas_init_gpu_ms", "Feature canvas init GPU"),
        ("roi_block_wall_ms", "Measured ROI block wall"),
    ]
    fig, axes = plt.subplots(len(components), 1, figsize=(9, 19), sharex=True)
    for ax, (field, label) in zip(axes, components):
        for mode, marker in (("roi_batch", "o"), ("adjacency_merge", "^")):
            rows = [r for r in frame_rows if r["mode"] == mode]
            if rows:
                xs = [r["unique_tile_input_area_ratio_pct_all_cameras"] for r in rows]
                ys = [r[field] for r in rows]
                ax.scatter(xs, ys, marker=marker, alpha=0.45, s=14, label=mode)
        ax.set_ylabel(label + "\n(ms)")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("ROI-Batch unique fixed-tile input area / all camera pixels (%)")
    axes[0].legend()
    fig.suptitle("Thor ROI latency decomposition (component times are diagnostic, not additive)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roi_latency_components.png"), dpi=220)
    plt.close(fig)

    # Plot 3: merge expansion and batch padding.
    adj_rows = [r for r in frame_rows if r["mode"] == "adjacency_merge"]
    if adj_rows:
        plt.figure(figsize=(8.5, 5.0))
        plt.scatter(
            [r["adjacency_edges_all_cameras"] for r in adj_rows],
            [r["merge_extra_area_ratio_pct_all_cameras"] for r in adj_rows],
            c=[r["raw_roi_union_area_ratio_pct_all_cameras"] for r in adj_rows],
            cmap="viridis", s=22, alpha=0.7,
        )
        plt.xlabel("Shared-edge adjacency count (all cameras)")
        plt.ylabel("Extra merged rectangle area / all camera pixels (%)")
        plt.title("Adjacency graph connectivity vs merge expansion")
        plt.colorbar(label="Raw ROI union coverage (%)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "adjacency_merge_expansion.png"), dpi=220)
        plt.close()

    if paired_rows:
        plt.figure(figsize=(7.5, 5.0))
        counts = [sum(r["actual_winner"] == mode for r in paired_rows) for mode in MODES]
        plt.bar(["ROI-Batch", "Adjacency-Merge"], counts)
        plt.ylabel("Paired frame/workload wins")
        plt.title("Measured winner count (six-camera ROI block wall time)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roi_mode_winner_counts.png"), dpi=220)
        plt.close()

        plt.figure(figsize=(8.5, 5.0))
        plt.scatter(
            [r["unique_tile_input_area_ratio_pct"] for r in paired_rows],
            [r["adjacency_minus_batch_ms"] for r in paired_rows],
            c=[r["raw_roi_union_area_ratio_pct"] for r in paired_rows],
            cmap="plasma", s=24, alpha=0.7,
        )
        plt.axhline(0.0, linewidth=1, linestyle="--")
        plt.xlabel("Unique fixed-tile input area (%)")
        plt.ylabel("Adjacency-Merge minus ROI-Batch wall time (ms)")
        plt.title("Paired latency difference; negative means Adjacency-Merge is faster")
        plt.colorbar(label="Raw ROI union coverage (%)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paired_mode_latency_difference.png"), dpi=220)
        plt.close()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script on NVIDIA Jetson Thor")
    if args.frames_per_workload <= 0 or args.warmup_frames < 0:
        raise ValueError("Invalid frame counts")
    if args.workers < 0 or args.unit_align <= 0 or args.feature_align <= 0:
        raise ValueError("workers must be >=0 and alignments must be positive")
    if not (0.0 <= args.size_variation < 1.0):
        raise ValueError("size-variation must be in [0,1)")
    if not (0.05 <= args.holdout_fraction <= 0.5):
        raise ValueError("holdout-fraction must be between 0.05 and 0.5")

    workloads = build_workloads(args)
    if not workloads:
        raise RuntimeError("No workloads generated")
    os.chdir(_REPO_ROOT)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    cfg = Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    import_project_modules(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    if isinstance(cfg.data.test, (list, tuple)):
        raise NotImplementedError("This profiler currently expects one test dataset config, not a list")
    test_cfg = cfg.data.test
    test_cfg.test_mode = True
    samples_per_gpu = int(test_cfg.pop("samples_per_gpu", 1))
    if samples_per_gpu != 1:
        raise ValueError("Sequential UniAD profiling requires samples_per_gpu=1")

    dataset = build_dataset(test_cfg)
    loader_kwargs = {}
    if args.workers > 0:
        loader_kwargs.update(timeout=180, prefetch_factor=1)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    model.eval().cuda()
    install_runtime_patch(model, args)
    model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    base_model = model.module

    os.makedirs(args.out_dir, exist_ok=True)
    camera_rows, frame_rows = [], []
    total_needed = args.warmup_frames + args.frames_per_workload

    print("[ROI-Tile-Dispatch] GPU:", torch.cuda.get_device_name(0), flush=True)
    print("[ROI-Tile-Dispatch] PyTorch/CUDA:", torch.__version__, torch.version.cuda, flush=True)
    print("[ROI-Tile-Dispatch] TF32:", bool(args.tf32), flush=True)
    print("[ROI-Tile-Dispatch] mode: real UniAD E2E forward; query-independent image-space ROIs", flush=True)
    print("[ROI-Tile-Dispatch] tile alignment: requested {}, effective multiple of lcm({}, {})={}".format(
        args.unit_align, args.unit_align, args.feature_align, math.lcm(args.unit_align, args.feature_align)), flush=True)
    print("[ROI-Tile-Dispatch] one ResNet + FPN call per camera per mode; no batch splitting", flush=True)
    print("[ROI-Tile-Dispatch] workloads: {}, modes: {}".format(len(workloads), MODES), flush=True)

    for wi, workload in enumerate(workloads):
        mode_order = list(MODES)
        if wi % 2:
            mode_order.reverse()
        print("\n[ROI-Tile-Dispatch] workload {}/{}: {}".format(
            wi + 1, len(workloads), workload["workload_id"]), flush=True)

        for mode in mode_order:
            print("  mode:", mode, flush=True)
            reset_temporal_state(base_model)
            base_model._roi_tile_mode = mode
            base_model._roi_tile_workload = workload
            recorded = 0

            for frame_idx, data in enumerate(data_loader):
                if frame_idx >= total_needed:
                    break
                # Same seed for paired mode/frame -> same GridMask randomness.
                seed_everything(args.seed + wi * 1000003 + frame_idx * 9176)
                torch.cuda.synchronize()
                e2e_t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                e2e_ms = (time.perf_counter() - e2e_t0) * 1000.0
                if frame_idx < args.warmup_frames:
                    continue

                context = dict(base_model._roi_tile_last_context)
                rows = [dict(r) for r in base_model._roi_tile_last_camera_rows]
                if not rows:
                    raise RuntimeError("extract_img_feat hook did not record camera measurements")
                meta = _get_first_meta(data)
                sample_idx = str(meta.get("sample_idx", frame_idx))
                scene_token = str(meta.get("scene_token", ""))
                context.update({
                    "frame_idx": int(frame_idx),
                    "sample_idx": sample_idx,
                    "scene_token": scene_token,
                    "e2e_forward_ms": float(e2e_ms),
                })
                frame_rows.append(context)
                for row in rows:
                    row.update({
                        "frame_idx": int(frame_idx),
                        "sample_idx": sample_idx,
                        "scene_token": scene_token,
                        "e2e_forward_ms": float(e2e_ms),
                    })
                    camera_rows.append(row)
                recorded += 1
                if recorded == 1:
                    print("    first measured frame: ROI block={:.2f} ms, E2E={:.2f} ms, "
                          "unique tiles/camera mean={:.1f}".format(
                              context["roi_block_wall_ms"], e2e_ms,
                              np.mean([r["unique_tile_count"] for r in rows])), flush=True)

            if recorded != args.frames_per_workload:
                raise RuntimeError(
                    "Expected {} measured frames, recorded {}. Dataset may be too short.".format(
                        args.frames_per_workload, recorded
                    )
                )

    camera_path = os.path.join(args.out_dir, "roi_tile_camera_raw.csv")
    frame_path = os.path.join(args.out_dir, "roi_tile_frame_raw.csv")
    save_csv(camera_path, camera_rows)
    save_csv(frame_path, frame_rows)
    save_csv(os.path.join(args.out_dir, "roi_tile_summary.csv"), _summarize(frame_rows))
    paired_rows = _make_paired_rows(frame_rows)
    save_csv(os.path.join(args.out_dir, "roi_tile_paired.csv"), paired_rows)
    _save_plots(frame_rows, paired_rows, args.out_dir)

    train_rows, hold_rows = _split_workloads(frame_rows, args.seed, args.holdout_fraction)
    total_models = _fit_models_by_mode(train_rows, ["roi_block_wall_ms"], args.ridge)
    holdout_paired, holdout_metrics = _evaluate_holdout(hold_rows, total_models)
    save_csv(os.path.join(args.out_dir, "roi_tile_holdout_paired.csv"), holdout_paired)

    final_total_models = _fit_models_by_mode(frame_rows, ["roi_block_wall_ms"], args.ridge)
    final_component_models = _fit_models_by_mode(frame_rows, list(COMPONENT_TARGETS), args.ridge)
    profile = {
        "schema_version": 3,
        "hardware": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32": bool(args.tf32),
        "fpn_strides": list(FPN_STRIDES),
        "benchmark_config": {
            "unit_align": int(args.unit_align),
            "feature_align": int(args.feature_align),
            "effective_align": int(math.lcm(args.unit_align, args.feature_align)),
            "one_batch_per_camera": True,
            "query_independent": True,
            "synthetic_image_space_rois": True,
            "overlap_roi_tile_dedup": True,
            "adjacency_rule": "shared_edge_only",
        },
        "frame_feature_names": list(FRAME_FEATURE_NAMES),
        "component_targets": list(COMPONENT_TARGETS),
        "total_time_models": final_total_models,
        "component_time_models": final_component_models,
        "holdout_metrics": holdout_metrics,
        "train_workload_count": len({r["workload_id"] for r in train_rows}),
        "holdout_workload_count": len({r["workload_id"] for r in hold_rows}),
        "workload_count": len(workloads),
        "notes": [
            "Horizontal-axis geometry is image-space ROI union coverage and unique fixed-tile area; it is independent of BEV query count.",
            "Each camera uses one ResNet call and one FPN call per mode; no ROI batch chunking is performed.",
            "A tile is activated by every ROI it intersects; tile IDs are deduplicated before either execution mode.",
            "Adjacency connects tile cells sharing an edge; diagonal contact does not connect.",
            "Component bounding rectangles that overlap are merged again to prevent duplicate pixel computation.",
            "Effective tile dimensions are multiples of lcm(unit_align, feature_align); with defaults this is 64 px and remains 32-aligned.",
            "ROI feature canvases are zero-filled outside computed crop rectangles; this is a latency benchmark, not an accuracy evaluation.",
            "Runtime selector predicts whole six-camera ROI-block wall time from pre-adjacency geometry features and does not build the dense adjacency matrix before choosing.",
        ],
    }
    with open(os.path.join(args.out_dir, "roi_tile_dispatch_model.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "roi_tile_holdout_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(holdout_metrics, f, indent=2, ensure_ascii=False)

    print("\n[ROI-Tile-Dispatch] DONE", flush=True)
    print("  camera rows:", len(camera_rows), flush=True)
    print("  frame rows:", len(frame_rows), flush=True)
    print("  paired samples:", len(paired_rows), flush=True)
    print("  holdout metrics:", holdout_metrics, flush=True)
    print("  output directory:", os.path.abspath(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
