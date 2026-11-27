#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Profile the QC-BEV Fig.7 backbone cost on Thor from real UniAD E2E forwards.

This script does NOT call a standalone/dummy backbone benchmark.  For every
measurement it invokes the normal UniAD test path:
    model(return_loss=False, rescale=True, **data)

A nested BEV-query subset is projected through the real sample's lidar2img
matrices.  The resulting per-camera image bounding boxes are cropped, passed
through the model's actual ResNet+FPN modules inside UniADTrack.extract_img_feat,
and scattered back to full-resolution feature canvases so the ordinary
BEVFormer encoder and downstream UniAD heads continue to execute.

The reported Fig.7 y-value is the synchronized wall-clock time of the actual
img_backbone + img_neck calls made inside that E2E forward.  Full E2E forward
time is also logged as a diagnostic, but is not used for the Fig.7 fit.

Run from the UniAD repository root.  This script targets the Thor branch's
ResNet-101 + 4-level FPN configuration (strides 8/16/32/64).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
import types
from collections import defaultdict

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Make the repository importable when this file is launched from tools/test/.
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


FPN_STRIDES = (8, 16, 32, 64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate QC-BEV Fig.7-style Backbone+FPN latency plot on NVIDIA Thor"
    )
    parser.add_argument("config", help="UniAD test config")
    parser.add_argument("checkpoint", help="UniAD checkpoint")
    parser.add_argument("--out-dir", default="work_dirs/fig7_thor")
    parser.add_argument(
        "--query-fractions",
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1.0",
        help="Nested near-ego BEV query fractions used to create projected image ROIs",
    )
    parser.add_argument("--frames-per-fraction", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-patch-h", type=int, default=8)
    parser.add_argument("--n-patch-w", type=int, default=16)
    parser.add_argument("--patch-expand", type=int, default=1)
    parser.add_argument("--num-points-in-pillar", type=int, default=4)
    parser.add_argument("--bbox-align", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    return parser.parse_args()


def parse_fractions(text):
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--query-fractions must contain at least one value")
    if any(x <= 0.0 or x > 1.0 for x in values):
        raise ValueError("query fractions must be in (0, 1]")
    return sorted(set(values))


def import_project_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)

    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", None)
        if plugin_dir:
            module_name = plugin_dir.rstrip("/").replace("/", ".")
            importlib.import_module(module_name)


def build_patch_dependency_table(
    pc_range,
    lidar2img,
    img_shape,
    bev_h,
    bev_w,
    num_points_in_pillar=4,
    n_patch_h=8,
    n_patch_w=16,
    patch_expand=1,
):
    """Port of the AE projection-table construction, using the real calib."""
    lidar2img = np.asarray(lidar2img, dtype=np.float64)
    if lidar2img.ndim != 3 or lidar2img.shape[1:] != (4, 4):
        raise ValueError("img_metas['lidar2img'] must have shape [num_cams,4,4]")

    n_cam = lidar2img.shape[0]
    image_h, image_w = int(img_shape[0]), int(img_shape[1])
    n_query = int(bev_h) * int(bev_w)
    depth_samples = int(num_points_in_pillar)
    z_range = float(pc_range[5] - pc_range[2])

    # Same BEVFormer pillar reference-point construction used by the AE.
    zs = np.linspace(0.5, z_range - 0.5, depth_samples, dtype=np.float64)
    xs = np.linspace(0.5, bev_w - 0.5, bev_w, dtype=np.float64)
    ys = np.linspace(0.5, bev_h - 0.5, bev_h, dtype=np.float64)
    zs = np.broadcast_to(zs[:, None, None], (depth_samples, bev_h, bev_w)) / z_range
    xs = np.broadcast_to(xs[None, None, :], (depth_samples, bev_h, bev_w)) / bev_w
    ys = np.broadcast_to(ys[None, :, None], (depth_samples, bev_h, bev_w)) / bev_h

    x_m = (xs * (pc_range[3] - pc_range[0]) + pc_range[0]).reshape(depth_samples, n_query)
    y_m = (ys * (pc_range[4] - pc_range[1]) + pc_range[1]).reshape(depth_samples, n_query)
    z_m = (zs * z_range + pc_range[2]).reshape(depth_samples, n_query)
    pts = np.stack(
        [x_m, y_m, z_m, np.ones((depth_samples, n_query), dtype=np.float64)],
        axis=-1,
    )

    ref_cam = np.einsum("cij,dqj->cdqi", lidar2img, pts)
    eps = 1e-5
    depth = np.maximum(ref_cam[..., 2], eps)
    u = ref_cam[..., 0] / depth / image_w
    v = ref_cam[..., 1] / depth / image_h
    valid = (
        (ref_cam[..., 2] > eps)
        & (u > 0.0) & (u < 1.0)
        & (v > 0.0) & (v < 1.0)
    )
    patch_col = (u * n_patch_w).astype(np.int32)
    patch_row = (v * n_patch_h).astype(np.int32)

    table = [dict() for _ in range(n_query)]
    for q in range(n_query):
        for cam in range(n_cam):
            patches = set()
            for d in range(depth_samples):
                if not valid[cam, d, q]:
                    continue
                for dr in range(-patch_expand, patch_expand + 1):
                    for dc in range(-patch_expand, patch_expand + 1):
                        rr = int(patch_row[cam, d, q]) + dr
                        cc = int(patch_col[cam, d, q]) + dc
                        if 0 <= rr < n_patch_h and 0 <= cc < n_patch_w:
                            patches.add(rr * n_patch_w + cc)
            if patches:
                table[q][cam] = patches
    return table


def make_near_ego_query_order(bev_h, bev_w, pc_range):
    """Nested, deterministic query order: closest BEV cells to ego first."""
    rows, cols = np.indices((bev_h, bev_w), dtype=np.float64)
    x = (cols + 0.5) / bev_w * (pc_range[3] - pc_range[0]) + pc_range[0]
    y = (rows + 0.5) / bev_h * (pc_range[4] - pc_range[1]) + pc_range[1]
    return np.argsort((x * x + y * y).reshape(-1))


def query_subset_to_camera_bboxes(
    selected_queries,
    patch_table,
    n_cam,
    image_h,
    image_w,
    n_patch_h=8,
    n_patch_w=16,
    align=32,
):
    """Union query-dependent patches and make stride-aligned per-camera boxes."""
    active_patches = [set() for _ in range(n_cam)]
    for q in selected_queries:
        for cam, patches in patch_table[int(q)].items():
            active_patches[int(cam)].update(patches)

    patch_h = image_h // n_patch_h
    patch_w = image_w // n_patch_w
    bboxes = []
    total_area = 0
    for cam in range(n_cam):
        patches = active_patches[cam]
        if not patches:
            bboxes.append(None)
            continue

        rows = [p // n_patch_w for p in patches]
        cols = [p % n_patch_w for p in patches]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        y0 = (min_row * patch_h // align) * align
        x0 = (min_col * patch_w // align) * align
        raw_y1 = image_h if max_row == n_patch_h - 1 else (max_row + 1) * patch_h
        raw_x1 = image_w if max_col == n_patch_w - 1 else (max_col + 1) * patch_w
        y1 = min(((raw_y1 + align - 1) // align) * align, image_h)
        x1 = min(((raw_x1 + align - 1) // align) * align, image_w)
        if x1 <= x0 or y1 <= y0:
            bboxes.append(None)
            continue

        bboxes.append((x0, y0, x1, y1))
        total_area += (x1 - x0) * (y1 - y0)

    area_ratio = total_area / float(max(1, n_cam * image_h * image_w))
    active_cameras = sum(box is not None for box in bboxes)
    return bboxes, area_ratio, active_cameras


def unwrap_test_meta(img_metas):
    """Get the first sample metadata from MMDet's test-time aug nesting."""
    metas = img_metas
    if isinstance(metas, (list, tuple)) and metas:
        if isinstance(metas[0], (list, tuple)):
            metas = metas[0]
    if isinstance(metas, (list, tuple)):
        if not metas:
            raise ValueError("empty img_metas")
        return metas[0]
    if isinstance(metas, dict):
        return metas
    raise TypeError("Unsupported img_metas nesting: {}".format(type(img_metas)))


def install_fig7_runtime_patch(model, cfg, args):
    """Install profile-only patches on the live UniAD model instance."""
    original_forward_test = model.forward_test
    original_extract_img_feat = model.extract_img_feat

    model._fig7_pc_range = list(cfg.get("point_cloud_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
    model._fig7_query_fraction = 1.0
    model._fig7_patch_table_cache = {}
    model._fig7_bbox_cache = {}
    model._fig7_query_order = make_near_ego_query_order(model.bev_h, model.bev_w, model._fig7_pc_range)
    model._fig7_n_patch_h = args.n_patch_h
    model._fig7_n_patch_w = args.n_patch_w
    model._fig7_patch_expand = args.patch_expand
    model._fig7_num_points = args.num_points_in_pillar
    model._fig7_bbox_align = args.bbox_align
    model._fig7_last_bb_neck_ms = None
    model._fig7_last_extract_total_ms = None
    model._fig7_last_roi_overhead_ms = None
    model._fig7_last_area_pct = None
    model._fig7_last_active_cameras = None
    model._fig7_last_bboxes = None
    model._fig7_last_scene = None
    model._fig7_last_sample = None
    model._fig7_last_table_build_ms = 0.0

    def forward_test_with_fig7(self, img=None, img_metas=None, **kwargs):
        img_tensor = img[0] if isinstance(img, (list, tuple)) else img
        if img_tensor is None:
            raise ValueError("forward_test received img=None")
        meta0 = unwrap_test_meta(img_metas)
        image_h, image_w = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])
        n_cam = int(img_tensor.shape[1])
        scene = str(meta0.get("scene_token", "unknown_scene"))
        sample = str(meta0.get("sample_idx", meta0.get("token", "")))
        lidar2img = np.asarray(meta0["lidar2img"], dtype=np.float64)
        if lidar2img.shape[0] != n_cam:
            raise ValueError(
                "lidar2img camera count {} != input camera count {}".format(
                    lidar2img.shape[0], n_cam
                )
            )

        calib_hash = hashlib.sha1(lidar2img.tobytes()).hexdigest()[:12]
        table_key = (scene, image_h, image_w, calib_hash)
        if table_key not in self._fig7_patch_table_cache:
            t_build = time.perf_counter()
            table = build_patch_dependency_table(
                pc_range=self._fig7_pc_range,
                lidar2img=lidar2img,
                img_shape=(image_h, image_w),
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                num_points_in_pillar=self._fig7_num_points,
                n_patch_h=self._fig7_n_patch_h,
                n_patch_w=self._fig7_n_patch_w,
                patch_expand=self._fig7_patch_expand,
            )
            self._fig7_patch_table_cache = {table_key: table}
            self._fig7_bbox_cache = {}
            self._fig7_last_table_build_ms = (time.perf_counter() - t_build) * 1000.0
        else:
            self._fig7_last_table_build_ms = 0.0
            table = self._fig7_patch_table_cache[table_key]

        frac = float(self._fig7_query_fraction)
        bbox_key = (table_key, round(frac, 8))
        if bbox_key not in self._fig7_bbox_cache:
            n_query = self.bev_h * self.bev_w
            n_selected = max(1, int(math.ceil(n_query * frac)))
            selected = self._fig7_query_order[:n_selected]
            bboxes, area_ratio, active_cameras = query_subset_to_camera_bboxes(
                selected_queries=selected,
                patch_table=table,
                n_cam=n_cam,
                image_h=image_h,
                image_w=image_w,
                n_patch_h=self._fig7_n_patch_h,
                n_patch_w=self._fig7_n_patch_w,
                align=self._fig7_bbox_align,
            )
            self._fig7_bbox_cache[bbox_key] = (bboxes, area_ratio, active_cameras)
        bboxes, area_ratio, active_cameras = self._fig7_bbox_cache[bbox_key]

        self._fig7_last_bboxes = bboxes
        self._fig7_last_area_pct = area_ratio * 100.0
        self._fig7_last_active_cameras = active_cameras
        self._fig7_last_scene = scene
        self._fig7_last_sample = sample
        self._fig7_last_bb_neck_ms = None
        self._fig7_last_extract_total_ms = None
        self._fig7_last_roi_overhead_ms = None

        return original_forward_test(img=img, img_metas=img_metas, **kwargs)

    def extract_img_feat_with_fig7(self, img, len_queue=None):
        if img is None or len_queue is not None or getattr(self, "_fig7_last_bboxes", None) is None:
            return original_extract_img_feat(img, len_queue)

        if img.dim() != 5:
            raise ValueError("expected img shape [B,N,C,H,W], got {}".format(tuple(img.shape)))
        batch_size, n_cam, channels, image_h, image_w = img.shape
        if len(self._fig7_last_bboxes) != n_cam:
            raise ValueError("bbox camera count does not match input")

        # Keep the original GridMask placement: apply it before camera crops.
        flat_img = img.reshape(batch_size * n_cam, channels, image_h, image_w)
        if self.use_grid_mask:
            flat_img = self.grid_mask(flat_img)
        img_for_crops = flat_img.reshape(batch_size, n_cam, channels, image_h, image_w)

        crop_specs = []
        for cam, bbox in enumerate(self._fig7_last_bboxes):
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            crop = img_for_crops[:, cam, :, y0:y1, x0:x1].contiguous()
            crop_specs.append((cam, bbox, crop))

        # One synchronized wall-clock interval around the real ResNet+FPN calls.
        # Cropping and feature scattering are outside this Fig.7 module timer.
        extract_t0 = time.perf_counter()
        outputs = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for cam, bbox, crop in crop_specs:
            feats = self.img_backbone(crop)
            if isinstance(feats, dict):
                feats = list(feats.values())
            if self.with_img_neck:
                feats = self.img_neck(feats)
            if not isinstance(feats, (list, tuple)):
                feats = [feats]
            outputs.append((cam, bbox, list(feats)))
        torch.cuda.synchronize()
        self._fig7_last_bb_neck_ms = (time.perf_counter() - t0) * 1000.0

        if outputs:
            sample_feats = outputs[0][2]
            if len(sample_feats) > len(FPN_STRIDES):
                raise ValueError("FPN output levels exceed configured stride list")
            strides = FPN_STRIDES[:len(sample_feats)]
            full_features = []
            for feat, stride in zip(sample_feats, strides):
                full_h = int(math.ceil(float(image_h) / stride))
                full_w = int(math.ceil(float(image_w) / stride))
                full_features.append(
                    feat.new_zeros((batch_size, n_cam, feat.shape[1], full_h, full_w))
                )
        else:
            # Degenerate case: no camera receives any projected query patch.
            if not self.with_img_neck:
                raise RuntimeError("profile mode expects the configured FPN neck")
            num_levels = int(getattr(self.img_neck, "num_outs", 4))
            out_channels = int(getattr(self.img_neck, "out_channels", 256))
            strides = FPN_STRIDES[:num_levels]
            full_features = []
            for stride in strides:
                full_h = int(math.ceil(float(image_h) / stride))
                full_w = int(math.ceil(float(image_w) / stride))
                full_features.append(
                    img.new_zeros((batch_size, n_cam, out_channels, full_h, full_w))
                )

        # Reconstruct full-size feature maps so the normal BEV encoder and all
        # downstream heads still execute in the same UniAD forward_test call.
        for cam, bbox, feats in outputs:
            x0, y0, _, _ = bbox
            for level, (feat, stride) in enumerate(zip(feats, strides)):
                full = full_features[level]
                fy0, fx0 = y0 // stride, x0 // stride
                copy_h = min(feat.shape[-2], full.shape[-2] - fy0)
                copy_w = min(feat.shape[-1], full.shape[-1] - fx0)
                if copy_h <= 0 or copy_w <= 0:
                    continue
                full[:, cam, :, fy0:fy0 + copy_h, fx0:fx0 + copy_w] = feat[:, :, :copy_h, :copy_w]

        torch.cuda.synchronize()
        self._fig7_last_extract_total_ms = (time.perf_counter() - extract_t0) * 1000.0
        self._fig7_last_roi_overhead_ms = max(
            0.0, self._fig7_last_extract_total_ms - self._fig7_last_bb_neck_ms
        )
        return full_features

    model.forward_test = types.MethodType(forward_test_with_fig7, model)
    model.extract_img_feat = types.MethodType(extract_img_feat_with_fig7, model)


def reset_temporal_state(model):
    """Reset UniADTrack streaming state before each query-fraction sweep."""
    model.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }
    model.prev_bev = None
    model.scene_token = None
    model.timestamp = None
    model.l2g_r_mat = None
    model.l2g_t = None
    model.test_track_instances = None


def fit_hinge(x, y):
    """Fit y = intercept + slope * max(0, x - threshold)."""
    unique_x = np.unique(x)
    if len(unique_x) < 4:
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        return {
            "kind": "linear_fallback",
            "intercept_ms": float(coef[0]),
            "slope_ms_per_pct": float(coef[1]),
            "threshold_pct": None,
            "prediction": pred,
        }

    best = None
    for threshold in unique_x[1:-1]:
        z = np.maximum(0.0, x - threshold)
        design = np.column_stack([np.ones_like(x), z])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        pred = design @ coef
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(threshold), coef, pred)

    _, threshold, coef, pred = best
    return {
        "kind": "piecewise_hinge",
        "intercept_ms": float(coef[0]),
        "slope_ms_per_pct": float(coef[1]),
        "threshold_pct": threshold,
        "prediction": pred,
    }


def fit_metrics(y, pred):
    residual = y - pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rel = np.abs(residual) / np.maximum(np.abs(y), 1e-6)
    return r2, float(np.percentile(rel, 90) * 100.0)


def save_results(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "fig7_thor_raw.csv")
    raw_fields = [
        "query_fraction", "normalized_input_area_pct", "bb_fpn_ms",
        "extract_img_feat_total_ms", "roi_overhead_ms", "e2e_forward_ms",
        "active_cameras", "scene_token", "sample_idx",
        "frame_in_sweep", "table_build_ms", "camera_bbox_areas",
    ]
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["query_fraction"])].append(row)

    summary_rows = []
    for frac in sorted(grouped):
        group = grouped[frac]
        y = np.asarray([float(r["bb_fpn_ms"]) for r in group], dtype=np.float64)
        x = np.asarray([float(r["normalized_input_area_pct"]) for r in group], dtype=np.float64)
        e2e = np.asarray([float(r["e2e_forward_ms"]) for r in group], dtype=np.float64)
        summary_rows.append({
            "query_fraction": frac,
            "normalized_input_area_pct_mean": float(np.mean(x)),
            "normalized_input_area_pct_std": float(np.std(x)),
            "bb_fpn_mean_ms": float(np.mean(y)),
            "bb_fpn_median_ms": float(np.median(y)),
            "bb_fpn_std_ms": float(np.std(y)),
            "bb_fpn_p90_ms": float(np.percentile(y, 90)),
            "extract_img_feat_total_mean_ms": float(np.mean([float(r["extract_img_feat_total_ms"]) for r in group])),
            "roi_overhead_mean_ms": float(np.mean([float(r["roi_overhead_ms"]) for r in group])),
            "e2e_forward_mean_ms": float(np.mean(e2e)),
            "num_frames": int(len(group)),
        })

    summary_path = os.path.join(out_dir, "fig7_thor_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fields = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(summary_rows)

    x = np.asarray([float(r["normalized_input_area_pct"]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["bb_fpn_ms"]) for r in rows], dtype=np.float64)
    fit = fit_hinge(x, y)
    r2, p90 = fit_metrics(y, fit["prediction"])
    fit_json = {
        "fit_kind": fit["kind"],
        "intercept_ms": fit["intercept_ms"],
        "slope_ms_per_pct": fit["slope_ms_per_pct"],
        "saturation_threshold_pct": fit["threshold_pct"],
        "r2": r2,
        "p90_relative_error_pct": p90,
        "num_measurements": int(len(rows)),
        "x_definition": "sum_c(aligned_bbox_area_c) / (num_cameras * padded_image_height * padded_image_width) * 100",
        "y_definition": "synchronized wall-clock time of actual img_backbone + img_neck calls inside UniAD forward_test",
    }
    with open(os.path.join(out_dir, "fig7_thor_fit.json"), "w") as f:
        json.dump(fit_json, f, indent=2)

    # Aggregate measurements for display; raw per-frame points are retained in CSV.
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(x, y, s=18, alpha=0.55, label="Measurements inside E2E forward")
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    if fit["threshold_pct"] is None:
        y_line = fit["intercept_ms"] + fit["slope_ms_per_pct"] * x_line
    else:
        y_line = fit["intercept_ms"] + fit["slope_ms_per_pct"] * np.maximum(
            0.0, x_line - fit["threshold_pct"]
        )
    plt.plot(x_line, y_line, linewidth=2.0, label="Piecewise linear fit")
    if fit["threshold_pct"] is not None:
        plt.axvline(
            fit["threshold_pct"],
            linestyle="--",
            linewidth=1.2,
            label="Fitted saturation threshold",
        )
    plt.xlabel("Normalized Input Area (%)")
    plt.ylabel("Backbone + FPN Latency (ms)")
    plt.title("Backbone latency vs. normalized input area on NVIDIA Thor")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.text(
        0.03,
        0.97,
        "R² = {:.4f}\nP90 relative error = {:.2f}%".format(r2, p90),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig7_thor.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("[Fig7] raw CSV:     {}".format(raw_path), flush=True)
    print("[Fig7] summary CSV: {}".format(summary_path), flush=True)
    print("[Fig7] fit JSON:    {}".format(os.path.join(out_dir, "fig7_thor_fit.json")), flush=True)
    print("[Fig7] plot:        {}".format(fig_path), flush=True)
    print("[Fig7] fit: R2={:.4f}, P90 relative error={:.2f}%".format(r2, p90), flush=True)


def main():
    args = parse_args()
    fractions = parse_fractions(args.query_fractions)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this script on the Thor GPU")

    repo_root = _REPO_ROOT
    os.chdir(repo_root)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    import_project_modules(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    test_cfg = cfg.data.test
    if isinstance(test_cfg, (list, tuple)):
        for item in test_cfg:
            item.test_mode = True
        samples_per_gpu = 1
    else:
        test_cfg.test_mode = True
        samples_per_gpu = int(test_cfg.pop("samples_per_gpu", 1))
    if samples_per_gpu != 1:
        raise ValueError("This sequential UniAD profiler requires samples_per_gpu=1")

    dataset = build_dataset(test_cfg)
    workers = args.workers if args.workers is not None else int(cfg.data.workers_per_gpu)
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs.update(timeout=90, prefetch_factor=1)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers,
        num_gpus=1,
        dist=False,
        shuffle=False,
        **loader_kwargs
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    model.eval().cuda()

    install_fig7_runtime_patch(model, cfg, args)
    model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    base_model = model.module

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    total_needed = args.warmup_frames + args.frames_per_fraction

    print("[Fig7] device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())), flush=True)
    print("[Fig7] TF32: {}".format(args.tf32), flush=True)
    print("[Fig7] dataset frames: {}".format(len(dataset)), flush=True)
    print("[Fig7] query fractions: {}".format(fractions), flush=True)

    for frac in fractions:
        print("\n[Fig7] sweep query fraction = {:.4f}".format(frac), flush=True)
        reset_temporal_state(base_model)
        base_model._fig7_query_fraction = frac
        recorded = 0
        for frame_idx, data in enumerate(data_loader):
            if frame_idx >= total_needed:
                break
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            e2e_ms = (time.perf_counter() - t0) * 1000.0

            bb_ms = base_model._fig7_last_bb_neck_ms
            area_pct = base_model._fig7_last_area_pct
            if bb_ms is None or area_pct is None:
                raise RuntimeError(
                    "Fig.7 instrumentation did not run inside UniAD forward; "
                    "check that test data reaches UniAD.forward_test/extract_img_feat."
                )

            if frame_idx < args.warmup_frames:
                continue

            bbox_areas = [
                0 if box is None else int((box[2] - box[0]) * (box[3] - box[1]))
                for box in base_model._fig7_last_bboxes
            ]
            rows.append({
                "query_fraction": float(frac),
                "normalized_input_area_pct": float(area_pct),
                "bb_fpn_ms": float(bb_ms),
                "extract_img_feat_total_ms": float(base_model._fig7_last_extract_total_ms),
                "roi_overhead_ms": float(base_model._fig7_last_roi_overhead_ms),
                "e2e_forward_ms": float(e2e_ms),
                "active_cameras": int(base_model._fig7_last_active_cameras),
                "scene_token": base_model._fig7_last_scene,
                "sample_idx": base_model._fig7_last_sample,
                "frame_in_sweep": int(frame_idx),
                "table_build_ms": float(base_model._fig7_last_table_build_ms),
                "camera_bbox_areas": json.dumps(bbox_areas),
            })
            recorded += 1

            if frame_idx == args.warmup_frames:
                print(
                    "  area={:.2f}% | active cameras={} | BB+FPN={:.3f} ms | E2E={:.3f} ms".format(
                        area_pct,
                        base_model._fig7_last_active_cameras,
                        bb_ms,
                        e2e_ms,
                    ),
                    flush=True,
                )

        print("  recorded {} frames".format(recorded), flush=True)
        if recorded == 0:
            raise RuntimeError("No measured frames. Check dataset length and warmup settings.")

    if not rows:
        raise RuntimeError("No profiling rows collected")
    save_results(rows, args.out_dir)


if __name__ == "__main__":
    main()
