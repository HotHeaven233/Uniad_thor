#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 ROI dispatch benchmark inside the real UniAD E2E forward on Jetson Thor.

The benchmark compares the V1 fixed-tile baseline with V2 adaptive tile search
and sparse-adjacency exact/cost-aware rectangle plans. It patches only
extract_img_feat; the normal UniAD forward_test/simple_test_track/BEVFormer and
all downstream heads still execute.

This is a latency/geometry experiment, not an accuracy benchmark. Synthetic
image-space ROIs are injected into the real nuScenes camera-image tensor. The
unselected feature canvas is zero-filled, so predictions must not be used to
claim model accuracy.

Timing policy: no torch.cuda.synchronize() occurs inside extract_img_feat.
CUDA Events bracket asynchronous GPU stages. The main loop synchronizes only
once after the complete model forward to measure true E2E wall time and resolve
events. CPU planner/packing time is reported separately and is not added to GPU
event time because CPU/GPU work may overlap.
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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

from roi_dispatch_v2_runtime import (
    build_v1_fixed_geometry, roi_batch_plan, select_best_plan,
)

FPN_STRIDES = (8, 16, 32, 64)
DEFAULT_ARMS = "v1_roi_batch,v2_roi_batch,adjacency_0,adjacency_05,adjacency_10,adjacency_20"
ARM_TO_MODE = {
    "v1_roi_batch": ("v1_roi_batch", 0.0),
    "v2_roi_batch": ("v2_roi_batch", 0.0),
    "adjacency_0": ("adjacency_merge", 0.0),
    "adjacency_05": ("adjacency_merge", 0.05),
    "adjacency_10": ("adjacency_merge", 0.10),
    "adjacency_20": ("adjacency_merge", 0.20),
}
ADJ_ARMS = ("adjacency_0", "adjacency_05", "adjacency_10", "adjacency_20")


def parse_args():
    p = argparse.ArgumentParser(description="V2 ROI strategy benchmark in real UniAD E2E forward on Thor")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--out-dir", default="work_dirs/roi_dispatch_v2_thor")
    p.add_argument("--roi-counts", default="4,8,16")
    p.add_argument("--area-fractions", default="0.05,0.10")
    p.add_argument("--layouts", default="spread,cluster,chain,random")
    p.add_argument("--aspect-ratios", default="0.5,1.0,2.0")
    p.add_argument("--size-variation", type=float, default=0.25)
    p.add_argument("--min-roi-side", type=int, default=64)
    p.add_argument("--frames-per-workload", type=int, default=2)
    p.add_argument("--warmup-frames", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--candidate-quantiles", default="0.50,0.75,0.90,1.00")
    p.add_argument("--align", type=int, default=64,
                   help="Effective crop/feature alignment; 64 aligns FPN strides 8/16/32/64")
    p.add_argument("--arms", default=DEFAULT_ARMS,
                   help="Comma list: v1_roi_batch,v2_roi_batch,adjacency_0,adjacency_05,adjacency_10,adjacency_20")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-workloads", type=int, default=0, help="0 means all; use 2-4 for smoke tests")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    return p.parse_args()


def parse_csv_values(text, cast):
    vals = [cast(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one comma-separated value")
    return vals


def seed_everything(seed):
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
    if cfg.get("plugin", False) and cfg.get("plugin_dir", None):
        importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))


def _layout_centers(n, image_h, image_w, base_h, base_w, layout, rng):
    centers = []
    if layout == "spread":
        cols = max(1, int(math.ceil(math.sqrt(n * image_w / float(image_h)))))
        rows = int(math.ceil(n / float(cols)))
        for i in range(n):
            r, c = divmod(i, cols)
            centers.append(((c + 0.5) * image_w / cols, (r + 0.5) * image_h / rows))
    elif layout == "cluster":
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / float(cols)))
        sx = max(8.0, min(base_w * 0.48, image_w / max(2.0, cols + 0.5)))
        sy = max(8.0, min(base_h * 0.48, image_h / max(2.0, rows + 0.5)))
        for i in range(n):
            r, c = divmod(i, cols)
            centers.append((image_w/2.0 + (c-(cols-1)/2.0)*sx,
                            image_h/2.0 + (r-(rows-1)/2.0)*sy))
    elif layout == "chain":
        sx, sy = max(8.0, base_w*0.95), max(8.0, base_h*0.95)
        cols = max(1, min(n, int(max(1.0, image_w-base_w)//sx)+1))
        for i in range(n):
            r,c=divmod(i,cols)
            if r%2: c=cols-1-c
            centers.append((base_w/2.0+c*sx, base_h/2.0+r*sy))
    elif layout == "random":
        centers=[(rng.uniform(0,image_w),rng.uniform(0,image_h)) for _ in range(n)]
    else:
        raise ValueError("Unknown layout: {}".format(layout))
    return centers


def generate_roi_boxes(image_h, image_w, n_rois, area_fraction, aspect_ratio, layout, seed,
                       size_variation=0.25, min_roi_side=64):
    if not (0.0 < area_fraction <= 1.0):
        raise ValueError("area_fraction must be in (0,1]")
    rng=np.random.RandomState(seed)
    target_each=area_fraction*image_h*image_w/float(n_rois)
    base_h=math.sqrt(target_each/float(aspect_ratio))
    base_w=math.sqrt(target_each*float(aspect_ratio))
    centers=_layout_centers(n_rois,image_h,image_w,base_h,base_w,layout,rng)
    boxes=[]
    for cx,cy in centers:
        scale=float(np.exp(rng.uniform(-size_variation,size_variation)))
        ar=float(aspect_ratio)*float(np.exp(rng.uniform(-0.18,0.18)))
        area=target_each*scale*scale
        w=max(float(min_roi_side),math.sqrt(area*ar))
        h=max(float(min_roi_side),math.sqrt(area/ar))
        w=min(w,float(image_w)); h=min(h,float(image_h))
        x0=int(round(cx-w/2)); y0=int(round(cy-h/2))
        x0=max(0,min(image_w-int(round(w)),x0)); y0=max(0,min(image_h-int(round(h)),y0))
        x1=min(image_w,x0+max(1,int(round(w))))
        y1=min(image_h,y0+max(1,int(round(h))))
        if x1<=x0: x1=min(image_w,x0+1)
        if y1<=y0: y1=min(image_h,y0+1)
        boxes.append((x0,y0,x1,y1))
    return boxes


def build_workloads(args):
    counts=parse_csv_values(args.roi_counts,int)
    areas=parse_csv_values(args.area_fractions,float)
    layouts=parse_csv_values(args.layouts,str)
    aspects=parse_csv_values(args.aspect_ratios,float)
    valid={"spread","cluster","chain","random"}
    if any(n<=0 for n in counts) or any(not (0< a <=1) for a in areas):
        raise ValueError("ROI counts and area fractions must be positive")
    if any(x not in valid for x in layouts) or any(x<=0 for x in aspects):
        raise ValueError("Invalid layout/aspect ratio")
    out=[]
    for n,a,l,ar in itertools.product(counts,areas,layouts,aspects):
        wid="n{}_a{}_{}_ar{}".format(n,str(a).replace(".","p"),l,str(ar).replace(".","p"))
        out.append({"workload_id":wid,"roi_count_target":n,"area_fraction_target":a,
                    "layout":l,"aspect_ratio_target":ar})
    if args.max_workloads>0: out=out[:args.max_workloads]
    return out


def _extract_raw_boxes(workload,n_cam,image_h,image_w,seed,size_variation,min_roi_side):
    return [generate_roi_boxes(image_h,image_w,workload["roi_count_target"],workload["area_fraction_target"],
                               workload["aspect_ratio_target"],workload["layout"],seed+cam*100003,
                               size_variation,min_roi_side) for cam in range(n_cam)]


def _event_pair():
    return (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))


def _event_ms(pair):
    return float(pair[0].elapsed_time(pair[1])) if pair else 0.0


def _make_padded_camera_image(camera_image,pad_h,pad_w):
    _,h,w=camera_image.shape
    if pad_h==h and pad_w==w: return camera_image
    padded=camera_image.new_zeros((camera_image.shape[0],pad_h,pad_w))
    padded[:,:h,:w].copy_(camera_image)
    return padded


def _make_crop_batch(camera_image,crop_boxes):
    if not crop_boxes: raise RuntimeError("Empty crop batch")
    sizes=[(b[3]-b[1],b[2]-b[0]) for b in crop_boxes]
    max_h=max(x[0] for x in sizes); max_w=max(x[1] for x in sizes)
    batch=camera_image.new_zeros((len(crop_boxes),camera_image.shape[0],max_h,max_w))
    for i,(x0,y0,x1,y1) in enumerate(crop_boxes):
        crop=camera_image[:,y0:y1,x0:x1]
        h,w=crop.shape[-2:]
        batch[i,:,:h,:w].copy_(crop)
    crop_area=sum(h*w for h,w in sizes)
    padded_area=len(crop_boxes)*max_h*max_w
    return batch.contiguous(),{"max_batch_h":max_h,"max_batch_w":max_w,"crop_area_sum_px":crop_area,
        "padded_batch_area_px":padded_area,"padding_area_px":padded_area-crop_area,
        "padding_waste_pct":100.0*(padded_area-crop_area)/max(1,padded_area),"batch_size":len(crop_boxes)}


def _normalize_feats(feats):
    if isinstance(feats,dict): feats=list(feats.values())
    if not isinstance(feats,(list,tuple)): feats=[feats]
    return list(feats)


def _init_feature_canvas(feats,batch_size,n_cam,image_h,image_w):
    out=[]
    for feat,stride in zip(feats,FPN_STRIDES):
        fh=int(math.ceil(image_h/float(stride))); fw=int(math.ceil(image_w/float(stride)))
        out.append(feat.new_zeros((batch_size,n_cam,feat.shape[1],fh,fw)))
    return out


def _scatter_features(full_features,feats,crop_boxes,camera_idx):
    for crop_idx,(x0,y0,x1,y1) in enumerate(crop_boxes):
        for level,stride in enumerate(FPN_STRIDES):
            feat=feats[level]
            dy,dx=y0//stride,x0//stride
            vh,vw=(y1-y0)//stride,(x1-x0)//stride
            dh=min(vh,full_features[level].shape[-2]-dy)
            dw=min(vw,full_features[level].shape[-1]-dx)
            if dh>0 and dw>0:
                full_features[level][0,camera_idx,:,dy:dy+dh,dx:dx+dw].copy_(feat[crop_idx,:,:dh,:dw])


def _arm_plan(arm,raw_boxes,image_h,image_w,quantiles,align):
    t0=time.perf_counter()
    if arm=="v1_roi_batch":
        geom=build_v1_fixed_geometry(raw_boxes,image_h,image_w,align)
        plan=roi_batch_plan(geom)
        elapsed=(time.perf_counter()-t0)*1000.0
        candidate_rows=[{"candidate_id":"v1_max_tile_origin0","quantile":1.0,"tile_w":geom["tile_w"],
            "tile_h":geom["tile_h"],"phase_x":0,"phase_y":0,"unique_tile_count":geom["unique_tile_count"],
            "exec_crop_count":plan["exec_crop_count"],"padded_batch_area_px":plan["padded_batch_area_px"],
            "exec_crop_area_sum_px":plan["exec_crop_area_sum_px"],"batch_padding_waste_pct":plan["batch_padding_waste_pct"],
            "mapping_cpu_ms":elapsed,"plan_cpu_ms":0.0,"selected":1}]
        search={"candidate_count":1,"candidate_search_cpu_ms":elapsed,"candidate_spec_cpu_ms":0.0,
                "mapping_search_cpu_ms":elapsed,"plan_search_cpu_ms":0.0,
                "planner_components_total_ms":{}}
    elif arm=="v2_roi_batch":
        search=select_best_plan(raw_boxes,image_h,image_w,"roi_batch",quantiles,0.0,align)
        geom,plan=search["geometry"],search["plan"]
        candidate_rows=search["candidate_rows"]
    elif arm in ADJ_ARMS:
        cap=ARM_TO_MODE[arm][1]
        search=select_best_plan(raw_boxes,image_h,image_w,"adjacency_merge",quantiles,cap,align)
        geom,plan=search["geometry"],search["plan"]
        candidate_rows=search["candidate_rows"]
    else:
        raise ValueError("Unknown arm: {}".format(arm))
    planner_wall=(time.perf_counter()-t0)*1000.0
    return geom,plan,search,candidate_rows,planner_wall


def _get_first_meta(data):
    value=data.get("img_metas") if isinstance(data,dict) else None
    while True:
        if hasattr(value,"data"): value=value.data; continue
        if isinstance(value,(list,tuple)) and value: value=value[0]; continue
        break
    return value if isinstance(value,dict) else {}


def _reset_temporal_state(model):
    model.prev_frame_info={"prev_bev":None,"scene_token":None,"prev_pos":0,"prev_angle":0}
    for name in ("prev_bev","scene_token","timestamp","l2g_r_mat","l2g_t","test_track_instances"):
        if hasattr(model,name): setattr(model,name,None)


def install_runtime_patch(model,args):
    model._roi_v2_arm=None
    model._roi_v2_workload=None
    model._roi_v2_box_cache={}
    model._roi_v2_camera_rows=[]
    model._roi_v2_candidate_rows=[]
    model._roi_v2_frame_context={}

    def extract_img_feat_v2(self,img,len_queue=None):
        if img is None or len_queue is not None or img.dim()!=5:
            return self._roi_v2_original_extract(img,len_queue)
        b,ncam,channels,image_h,image_w=img.shape
        if b!=1: raise RuntimeError("V2 profiler requires batch_size=1")
        arm=self._roi_v2_arm; workload=self._roi_v2_workload
        if arm not in ARM_TO_MODE or workload is None:
            raise RuntimeError("V2 arm/workload was not set before E2E forward")

        # Original UniAD preprocessing order: GridMask before ROI crops.
        gridmask_event=_event_pair()
        gridmask_host_t0=time.perf_counter()
        gridmask_event[0].record()
        flat=img.reshape(b*ncam,channels,image_h,image_w)
        if getattr(self,"use_grid_mask",False): flat=self.grid_mask(flat)
        img_full=flat.reshape(b,ncam,channels,image_h,image_w)
        gridmask_event[1].record()
        gridmask_host_ms=(time.perf_counter()-gridmask_host_t0)*1000.0

        cache_key=(workload["workload_id"],ncam,image_h,image_w)
        if cache_key not in self._roi_v2_box_cache:
            self._roi_v2_box_cache[cache_key]=_extract_raw_boxes(workload,ncam,image_h,image_w,args.seed,
                                                                  args.size_variation,args.min_roi_side)
        boxes_by_cam=self._roi_v2_box_cache[cache_key]

        roi_start_event=torch.cuda.Event(enable_timing=True); roi_start_event.record()
        roi_host_t0=time.perf_counter()
        camera_rows=[]; candidate_rows=[]; full_features=None; canvas_event=None; canvas_host_ms=0.0
        for cam in range(ncam):
            raw_boxes=boxes_by_cam[cam]
            geom,plan,search,search_rows,planner_wall=_arm_plan(
                arm,raw_boxes,image_h,image_w,args.quantiles,args.align)
            for cr in search_rows:
                cr.update({"arm":arm,"workload_id":workload["workload_id"],"camera":cam})
                candidate_rows.append(cr)

            # Pad only the source canvas required by the selected grid; no full
            # image re-copy when the image dimensions are already aligned.
            pad_event=_event_pair(); pad_host_t0=time.perf_counter(); pad_event[0].record()
            padded=_make_padded_camera_image(img_full[0,cam],geom["padded_h"],geom["padded_w"])
            pad_event[1].record(); pad_host_ms=(time.perf_counter()-pad_host_t0)*1000.0

            pack_event=_event_pair(); pack_host_t0=time.perf_counter(); pack_event[0].record()
            roi_batch,pack=_make_crop_batch(padded,plan["exec_boxes"])
            pack_event[1].record(); pack_host_ms=(time.perf_counter()-pack_host_t0)*1000.0
            if list(roi_batch.shape[-2:]) != [plan["max_batch_h"],plan["max_batch_w"]]:
                raise RuntimeError("Crop batch shape differs from geometry plan")

            resnet_event=_event_pair(); resnet_host_t0=time.perf_counter(); resnet_event[0].record()
            backbone=_normalize_feats(self.img_backbone(roi_batch))
            resnet_event[1].record(); resnet_host_ms=(time.perf_counter()-resnet_host_t0)*1000.0

            if self.with_img_neck:
                fpn_event=_event_pair(); fpn_host_t0=time.perf_counter(); fpn_event[0].record()
                neck=_normalize_feats(self.img_neck(backbone))
                fpn_event[1].record(); fpn_host_ms=(time.perf_counter()-fpn_host_t0)*1000.0
            else:
                fpn_event=None; neck=backbone; fpn_host_ms=0.0
            if len(neck)!=len(FPN_STRIDES):
                raise RuntimeError("Expected four FPN outputs at strides {} but got {}".format(FPN_STRIDES,len(neck)))
            for feat,stride in zip(neck,FPN_STRIDES):
                if feat.shape[-2]!=roi_batch.shape[-2]//stride or feat.shape[-1]!=roi_batch.shape[-1]//stride:
                    raise RuntimeError("FPN output/crop size mismatch at stride {}: {} vs {}".format(
                        stride,tuple(feat.shape[-2:]),tuple(roi_batch.shape[-2:])))

            if full_features is None:
                canvas_event=_event_pair(); canvas_host_t0=time.perf_counter(); canvas_event[0].record()
                full_features=_init_feature_canvas(neck,b,ncam,image_h,image_w)
                canvas_event[1].record(); canvas_host_ms=(time.perf_counter()-canvas_host_t0)*1000.0
            scatter_event=_event_pair(); scatter_host_t0=time.perf_counter(); scatter_event[0].record()
            _scatter_features(full_features,neck,plan["exec_boxes"],cam)
            scatter_event[1].record(); scatter_host_ms=(time.perf_counter()-scatter_host_t0)*1000.0

            crop_boxes=plan["exec_boxes"]
            image_area=image_h*image_w
            exec_valid_area=sum(max(0,min(box[2],image_w)-max(0,box[0])) *
                                max(0,min(box[3],image_h)-max(0,box[1])) for box in crop_boxes)
            row={
                "arm":arm,"workload_id":workload["workload_id"],"roi_count_target":workload["roi_count_target"],
                "area_fraction_target":workload["area_fraction_target"],"layout":workload["layout"],
                "aspect_ratio_target":workload["aspect_ratio_target"],"camera":cam,
                "image_h":image_h,"image_w":image_w,
                "planner_wall_cpu_ms":planner_wall,
                "candidate_count":search["candidate_count"],
                "candidate_spec_cpu_ms":search.get("candidate_spec_cpu_ms",0.0),
                "candidate_search_cpu_ms":search["candidate_search_cpu_ms"],
                "mapping_search_cpu_ms":search.get("mapping_search_cpu_ms",0.0),
                "plan_search_cpu_ms":search.get("plan_search_cpu_ms",0.0),
                "adjacency_build_cpu_ms":search.get("planner_components_total_ms",{}).get("adjacency_build_cpu_ms",0.0),
                "union_find_cpu_ms":search.get("planner_components_total_ms",{}).get("union_find_cpu_ms",0.0),
                "rectangle_partition_cpu_ms":search.get("planner_components_total_ms",{}).get("rectangle_partition_cpu_ms",0.0),
                "cost_aware_merge_cpu_ms":search.get("planner_components_total_ms",{}).get("cost_aware_merge_cpu_ms",0.0),
                "tile_w":geom["tile_w"],"tile_h":geom["tile_h"],"phase_x":geom["phase_x"],"phase_y":geom["phase_y"],
                "selected_candidate_id":search.get("candidate_spec",{}).get("candidate_id","v1_max_tile_origin0"),
                "raw_roi_count":geom["raw_roi_count"],"raw_roi_sum_area_px":geom["raw_roi_sum_area_px"],
                "raw_roi_union_area_px":geom["raw_roi_union_area_px"],
                "raw_roi_union_area_ratio_pct":geom["raw_roi_union_area_ratio_pct"],
                "roi_tile_hits":geom["roi_tile_hits"],"unique_tile_count":geom["unique_tile_count"],
                "tile_dedup_savings":geom["tile_dedup_savings"],"tile_input_area_px":geom["tile_input_area_px"],
                "tile_input_valid_area_px":geom["tile_input_valid_area_px"],
                "tile_input_outside_image_area_px":geom["tile_input_outside_image_area_px"],
                "tile_input_area_ratio_pct":100.0*geom["tile_input_area_px"]/image_area,
                "exec_crop_count":plan["exec_crop_count"],"exec_crop_area_sum_px":plan["exec_crop_area_sum_px"],
                "exec_crop_area_ratio_pct":100.0*plan["exec_crop_area_sum_px"]/image_area,
                "merge_expansion_area_px":plan.get("merge_expansion_area_px",0),
                "merge_expansion_ratio":plan["exec_crop_area_sum_px"]/max(1,geom["tile_input_area_px"]),
                "batch_size":pack["batch_size"],"max_batch_h":pack["max_batch_h"],"max_batch_w":pack["max_batch_w"],
                "padded_batch_area_px":pack["padded_batch_area_px"],"batch_padding_area_px":pack["padding_area_px"],
                "batch_padding_waste_pct":pack["padding_waste_pct"],
                "batch_tensor_bytes":int(roi_batch.numel()*roi_batch.element_size()),
                "feature_canvas_bytes":int(sum(x.numel()*x.element_size() for x in full_features)),
                "source_pad_added_pixels":int(max(0,geom["padded_h"]*geom["padded_w"]-image_area)),
                "adjacency_edges":plan.get("adjacency_edges",0),"component_count":plan.get("component_count",0),
                "exact_rect_count":plan.get("exact_rect_count",0),"merge_count":plan.get("merge_count",0),
                "pad_host_ms":pad_host_ms,"crop_pack_host_ms":pack_host_ms,
                "resnet_host_enqueue_ms":resnet_host_ms,"fpn_host_enqueue_ms":fpn_host_ms,
                "scatter_host_enqueue_ms":scatter_host_ms,
                "_events":{"pad":pad_event,"pack":pack_event,"resnet":resnet_event,"fpn":fpn_event,
                           "scatter":scatter_event},
            }
            camera_rows.append(row)

        roi_end_event=torch.cuda.Event(enable_timing=True); roi_end_event.record()
        roi_host_ms=(time.perf_counter()-roi_host_t0)*1000.0
        self._roi_v2_camera_rows=camera_rows
        self._roi_v2_candidate_rows=candidate_rows
        self._roi_v2_frame_context={"arm":arm,"workload_id":workload["workload_id"],
            "roi_block_host_enqueue_ms":roi_host_ms,"gridmask_host_ms":gridmask_host_ms,
            "feature_canvas_init_host_ms":canvas_host_ms,
            "_roi_block_events":(roi_start_event,roi_end_event),
            "_gridmask_event":gridmask_event,"_canvas_event":canvas_event}
        return full_features

    model._roi_v2_original_extract=model.extract_img_feat
    model.extract_img_feat=types.MethodType(extract_img_feat_v2,model)


def _save_csv(path,rows):
    if not rows:
        with open(path,"w",newline="",encoding="utf-8") as f: f.write("\n")
        return
    keys=[]; seen=set()
    for row in rows:
        for k in row:
            if k.startswith("_"): continue
            if k not in seen: seen.add(k); keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore"); w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v)
                        for k,v in row.items() if not k.startswith("_")})


def _paired_rows(frame_rows):
    groups=defaultdict(dict)
    for r in frame_rows:
        key=(r["workload_id"],r["frame_idx"],r["sample_idx"])
        groups[key][r["arm"]]=r
    out=[]
    for key,arms in groups.items():
        if "v2_roi_batch" not in arms: continue
        a=arms["v2_roi_batch"]
        row={"workload_id":key[0],"frame_idx":key[1],"sample_idx":key[2],"layout":a["layout"],
             "roi_count_target":a["roi_count_target"],"area_fraction_target":a["area_fraction_target"],
             "aspect_ratio_target":a["aspect_ratio_target"],"v2_roi_batch_e2e_ms":a["e2e_forward_ms"],
             "v2_roi_batch_gpu_stream_ms":a["roi_block_gpu_stream_ms"]}
        if "v1_roi_batch" in arms:
            row["v1_roi_batch_e2e_ms"]=arms["v1_roi_batch"]["e2e_forward_ms"]
            row["v1_minus_v2_roi_batch_ms"]=row["v1_roi_batch_e2e_ms"]-row["v2_roi_batch_e2e_ms"]
        for arm in ADJ_ARMS:
            if arm in arms:
                b=arms[arm]
                row[arm+"_e2e_ms"]=b["e2e_forward_ms"]
                row[arm+"_gpu_stream_ms"]=b["roi_block_gpu_stream_ms"]
                row[arm+"_e2e_delta_ms"]=b["e2e_forward_ms"]-a["e2e_forward_ms"]
                row[arm+"_winner"]="v2_roi_batch" if a["e2e_forward_ms"]<=b["e2e_forward_ms"] else arm
        out.append(row)
    return out


def _bootstrap_mean_ci(values, n_boot, seed):
    vals=np.asarray(values,dtype=np.float64)
    if len(vals)==0: return [None,None]
    if len(vals)==1: return [float(vals[0]),float(vals[0])]
    rng=np.random.RandomState(seed)
    samples=rng.choice(vals,size=(max(100,int(n_boot)),len(vals)),replace=True).mean(axis=1)
    return [float(np.percentile(samples,2.5)),float(np.percentile(samples,97.5))]


def _decision(frame_rows,args):
    # Workload is the independent unit: aggregate repeated frames before
    # counting wins and bootstrapping, avoiding pseudo-replication.
    by_workload=defaultdict(lambda:defaultdict(list))
    for r in frame_rows:
        by_workload[r["workload_id"]][r["arm"]].append(float(r["e2e_forward_ms"]))
    medians_by_workload={wid:{a:float(np.median(v)) for a,v in vals.items()}
                         for wid,vals in by_workload.items()}

    roi_variants=[a for a in ("v1_roi_batch","v2_roi_batch")
                  if any(a in x for x in medians_by_workload.values())]
    roi_variant_means={a:float(np.mean([x[a] for x in medians_by_workload.values() if a in x]))
                       for a in roi_variants}
    v1v2=[vals["v1_roi_batch"]-vals["v2_roi_batch"]
          for vals in medians_by_workload.values()
          if "v1_roi_batch" in vals and "v2_roi_batch" in vals]
    v2_wins=sum(x>0 for x in v1v2)
    v2_ci=_bootstrap_mean_ci(v1v2,args.bootstrap_samples,args.seed+1991)
    if len(v1v2) and v2_wins/len(v1v2)>0.5 and v2_ci[0] is not None and v2_ci[0]>0:
        preferred_roi_geometry="v2_roi_batch"
    elif "v1_roi_batch" in roi_variant_means:
        preferred_roi_geometry="v1_roi_batch"
    else:
        preferred_roi_geometry="v2_roi_batch"

    # Compare the globally preferred ROI-Batch variant with the best measured
    # adjacency arm in each workload. Do not select V1/V2 separately per case.
    workload_rows=[]
    for wid,vals in medians_by_workload.items():
        if preferred_roi_geometry not in vals or not any(a in vals for a in ADJ_ARMS): continue
        batch=vals[preferred_roi_geometry]
        adj={a:vals[a] for a in ADJ_ARMS if a in vals}
        best_arm=min(adj,key=adj.get); best_adj=adj[best_arm]
        workload_rows.append({"workload_id":wid,"selected_roi_batch_variant":preferred_roi_geometry,
            "roi_batch_median_e2e_ms":batch,"v1_roi_batch_median_e2e_ms":vals.get("v1_roi_batch"),
            "v2_roi_batch_median_e2e_ms":vals.get("v2_roi_batch"),"best_adjacency_arm":best_arm,
            "best_adjacency_median_e2e_ms":best_adj,"adjacency_minus_batch_ms":best_adj-batch,
            "winner":"roi_batch" if batch<best_adj else best_arm})
    deltas=[r["adjacency_minus_batch_ms"] for r in workload_rows]
    wins=sum(x>0 for x in deltas); n=len(deltas)
    win_rate=wins/max(1,n)
    ci=_bootstrap_mean_ci(deltas,args.bootstrap_samples,args.seed+991)
    if n and win_rate>=0.80 and ci[0] is not None and ci[0]>0:
        status="ROI_BATCH_DOMINATES_USE_ROI_BATCH"
        recommendation="Use {} as the default ROI-Batch path. It beats the best measured adjacency variant in at least 80% of independent workload groups, with a positive 95% bootstrap CI.".format(preferred_roi_geometry)
    elif n and win_rate>0.50 and ci[0] is not None and ci[0]>0:
        status="ROI_BATCH_MAJORITY_KEEP_ROI_BATCH_DEFAULT"
        recommendation="Keep {} as the conservative default ROI-Batch path. It wins a majority, but not the 80% dominance threshold; expand holdout coverage before adding runtime switching.".format(preferred_roi_geometry)
    else:
        status="NO_ROI_BATCH_DOMINANCE_PROVEN"
        recommendation="Do not deploy an automatic dispatcher from this run. Keep {} as the safe baseline and inspect workload strata / collect more paired E2E data.".format(preferred_roi_geometry)
    return {
        "primary_metric":"complete UniAD E2E forward wall time after one end-of-forward CUDA synchronize",
        "independent_unit":"workload_id; repeated frames aggregated by median",
        "preferred_roi_batch_geometry":preferred_roi_geometry,
        "roi_batch_variant_mean_workload_median_e2e_ms":roi_variant_means,
        "v1_minus_v2_roi_batch_mean_workload_delta_ms":float(np.mean(v1v2)) if v1v2 else None,
        "v2_roi_batch_workload_wins_vs_v1":v2_wins,
        "v1_v2_workload_count":len(v1v2),
        "v2_minus_v1_bootstrap_95pct_ci_ms":v2_ci,
        "workload_count":n,"roi_batch_workload_wins":wins,"roi_batch_workload_win_rate":win_rate,
        "mean_best_adjacency_minus_batch_ms":float(np.mean(deltas)) if deltas else None,
        "median_best_adjacency_minus_batch_ms":float(np.median(deltas)) if deltas else None,
        "bootstrap_95pct_ci_mean_delta_ms":ci,
        "dominance_threshold":0.80,"status":status,"recommendation":recommendation,
        "workload_results":workload_rows,
        "note":"Best adjacency arm is selected per workload from exact/5/10/20% expansion variants; this is conservative against ROI-Batch. One ROI-Batch geometry variant is selected globally, not per workload.",
    }


def _summaries(frame_rows):
    groups=defaultdict(list)
    for r in frame_rows:
        groups[(r["arm"],r["layout"],r["roi_count_target"],r["area_fraction_target"],r["aspect_ratio_target"])].append(r)
    metrics=["e2e_forward_ms","roi_block_gpu_stream_ms","roi_block_host_enqueue_ms","planner_wall_cpu_ms_sum",
             "resnet_gpu_ms_sum","fpn_gpu_ms_sum","crop_pack_gpu_ms_sum","scatter_gpu_ms_sum",
             "candidate_search_cpu_ms_sum","batch_padding_waste_pct_mean","cuda_peak_allocated_mb"]
    out=[]
    for key,rows in sorted(groups.items(),key=lambda x:str(x[0])):
        d={"arm":key[0],"layout":key[1],"roi_count_target":key[2],"area_fraction_target":key[3],
           "aspect_ratio_target":key[4],"samples":len(rows)}
        for m in metrics:
            vals=[float(r[m]) for r in rows if m in r]
            if vals:
                d[m+"_mean"]=float(np.mean(vals)); d[m+"_median"]=float(np.median(vals)); d[m+"_p90"]=float(np.percentile(vals,90))
        out.append(d)
    return out


def _save_plots(frame_rows,paired,out_dir):
    if not frame_rows: return
    arms=list(dict.fromkeys(r["arm"] for r in frame_rows))
    means=[np.mean([r["e2e_forward_ms"] for r in frame_rows if r["arm"]==a]) for a in arms]
    plt.figure(figsize=(10,5)); plt.bar(arms,means); plt.ylabel("Complete UniAD E2E forward (ms)")
    plt.title("V2 strategy comparison on Thor (lower is better)"); plt.xticks(rotation=20,ha="right"); plt.grid(axis="y",alpha=.25)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"roi_v2_e2e_by_arm.png"),dpi=200); plt.close()

    if paired:
        for arm in ADJ_ARMS:
            vals=[r.get(arm+"_e2e_delta_ms") for r in paired if r.get(arm+"_e2e_delta_ms") is not None]
            if not vals: continue
            plt.figure(figsize=(8,4.5)); plt.hist(vals,bins=24); plt.axvline(0,linestyle="--",linewidth=1)
            plt.xlabel("Adjacency arm E2E minus V2 ROI-Batch E2E (ms)"); plt.ylabel("Paired frames")
            plt.title("Paired latency delta: {} (positive means ROI-Batch faster)".format(arm))
            plt.grid(alpha=.25); plt.tight_layout()
            plt.savefig(os.path.join(out_dir,"delta_{}.png".format(arm)),dpi=200); plt.close()

    plt.figure(figsize=(9,5))
    for arm in arms:
        rows=[r for r in frame_rows if r["arm"]==arm]
        plt.scatter([r["tile_input_area_ratio_pct_sum"] for r in rows],
                    [r["e2e_forward_ms"] for r in rows],s=18,alpha=.55,label=arm)
    plt.xlabel("Selected tile input area / six-camera image area (%)"); plt.ylabel("Complete E2E forward (ms)")
    plt.title("E2E latency vs selected tile input area"); plt.grid(alpha=.25); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"roi_v2_e2e_vs_tile_area.png"),dpi=200); plt.close()


def main():
    args=parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; run on NVIDIA Jetson Thor")
    if args.frames_per_workload<=0 or args.warmup_frames<0 or args.workers<0: raise ValueError("Invalid frame/worker count")
    if args.align!=64:
        raise ValueError("This UniAD config has FPN stride 64; use --align 64 to preserve feature-grid alignment")
    args.quantiles=parse_csv_values(args.candidate_quantiles,float)
    if any(not (0<q<=1) for q in args.quantiles): raise ValueError("Candidate quantiles must be in (0,1]")
    arms=parse_csv_values(args.arms,str)
    if len(set(arms))!=len(arms) or any(a not in ARM_TO_MODE for a in arms): raise ValueError("Invalid or duplicate --arms value")
    if "v2_roi_batch" not in arms: raise ValueError("Include v2_roi_batch as the reference arm")
    if not any(a in ADJ_ARMS for a in arms): raise ValueError("Include at least one adjacency arm")

    workloads=build_workloads(args)
    if not workloads: raise RuntimeError("No workloads generated")
    os.chdir(_REPO_ROOT)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32=bool(args.tf32)
    torch.backends.cudnn.allow_tf32=bool(args.tf32)

    cfg=Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark",False): torch.backends.cudnn.benchmark=True
    import_project_modules(cfg)
    cfg.model.pretrained=None; cfg.model.train_cfg=None
    if isinstance(cfg.data.test,(list,tuple)): raise NotImplementedError("Expected one test dataset config")
    test_cfg=cfg.data.test; test_cfg.test_mode=True
    samples_per_gpu=int(test_cfg.pop("samples_per_gpu",1))
    if samples_per_gpu!=1: raise ValueError("Sequential UniAD profiling requires samples_per_gpu=1")
    dataset=build_dataset(test_cfg)
    loader_kwargs={}
    if args.workers>0: loader_kwargs.update(timeout=180,prefetch_factor=1)
    data_loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=args.workers,num_gpus=1,
                                 dist=False,shuffle=False,**loader_kwargs)
    model=build_model(cfg.model,test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16",None) is not None: wrap_fp16_model(model)
    checkpoint=load_checkpoint(model,args.checkpoint,map_location="cpu")
    model.CLASSES=checkpoint.get("meta",{}).get("CLASSES",dataset.CLASSES)
    model.eval().cuda(); install_runtime_patch(model,args)
    model=MMDataParallel(model,device_ids=[torch.cuda.current_device()]); base=model.module

    os.makedirs(args.out_dir,exist_ok=True)
    camera_rows=[]; frame_rows=[]; candidate_rows=[]
    total_needed=args.warmup_frames+args.frames_per_workload
    env={"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
         "tf32":bool(args.tf32),"align":args.align,"fpn_strides":list(FPN_STRIDES),
         "arms":arms,"workload_count":len(workloads),"frames_per_workload":args.frames_per_workload,
         "warmup_frames":args.warmup_frames,"workers":args.workers,
         "candidate_quantiles":args.quantiles,"note":"Synthetic image-space ROIs; real UniAD E2E forward; no accuracy claim."}
    with open(os.path.join(args.out_dir,"roi_v2_environment.json"),"w",encoding="utf-8") as f: json.dump(env,f,indent=2)
    print("[ROI-V2] GPU={} Torch={} CUDA={} TF32={}".format(env["gpu"],env["torch"],env["cuda"],env["tf32"]),flush=True)
    print("[ROI-V2] workloads={} arms={}".format(len(workloads),arms),flush=True)
    print("[ROI-V2] no CUDA synchronize inside extract_img_feat; one sync after full E2E forward",flush=True)

    for wi,workload in enumerate(workloads):
        # Rotate arm order to reduce systematic thermal/order bias.
        shift=wi%len(arms); arm_order=arms[shift:]+arms[:shift]
        print("\n[ROI-V2] workload {}/{} {}".format(wi+1,len(workloads),workload["workload_id"]),flush=True)
        for arm in arm_order:
            print("  arm={}".format(arm),flush=True)
            _reset_temporal_state(base)
            base._roi_v2_arm=arm; base._roi_v2_workload=workload
            recorded=0
            for frame_idx,data in enumerate(data_loader):
                if frame_idx>=total_needed: break
                seed_everything(args.seed+wi*1000003+frame_idx*9176)
                torch.cuda.reset_peak_memory_stats()
                mem_before=torch.cuda.memory_allocated()/1048576.0
                forward_host_t0=time.perf_counter()
                with torch.no_grad():
                    _=model(return_loss=False,rescale=True,**data)
                forward_host_enqueue_ms=(time.perf_counter()-forward_host_t0)*1000.0
                # This is the only explicit synchronize in one full model forward.
                torch.cuda.synchronize()
                e2e_ms=(time.perf_counter()-forward_host_t0)*1000.0
                if frame_idx<args.warmup_frames: continue

                context=dict(base._roi_v2_frame_context)
                cam_rows=[dict(x) for x in base._roi_v2_camera_rows]
                cand_rows=[dict(x) for x in base._roi_v2_candidate_rows]
                if not cam_rows: raise RuntimeError("No camera rows recorded; extract_img_feat patch was not called")
                roi_events=context.pop("_roi_block_events")
                grid_event=context.pop("_gridmask_event")
                canvas_event=context.pop("_canvas_event")
                roi_gpu_ms=_event_ms(roi_events)
                gridmask_gpu_ms=_event_ms(grid_event)
                canvas_gpu_ms=_event_ms(canvas_event)
                for cr in cam_rows:
                    ev=cr.pop("_events")
                    cr.update({"pad_gpu_ms":_event_ms(ev["pad"]),"crop_pack_gpu_ms":_event_ms(ev["pack"]),
                               "resnet_gpu_ms":_event_ms(ev["resnet"]),"fpn_gpu_ms":_event_ms(ev["fpn"]),
                               "scatter_gpu_ms":_event_ms(ev["scatter"]),
                               "frame_idx":frame_idx})
                meta=_get_first_meta(data)
                sample_idx=str(meta.get("sample_idx",frame_idx)); scene_token=str(meta.get("scene_token",""))
                planner_sum=sum(r["planner_wall_cpu_ms"] for r in cam_rows)
                frame={"arm":arm,"workload_id":workload["workload_id"],"roi_count_target":workload["roi_count_target"],
                       "area_fraction_target":workload["area_fraction_target"],"layout":workload["layout"],
                       "aspect_ratio_target":workload["aspect_ratio_target"],"frame_idx":frame_idx,
                       "sample_idx":sample_idx,"scene_token":scene_token,"camera_count":len(cam_rows),
                       "e2e_forward_ms":e2e_ms,"forward_host_enqueue_ms":forward_host_enqueue_ms,
                       "roi_block_host_enqueue_ms":context["roi_block_host_enqueue_ms"],
                       "gridmask_host_ms":context["gridmask_host_ms"],
                       "roi_block_gpu_stream_ms":roi_gpu_ms,"gridmask_gpu_ms":gridmask_gpu_ms,
                       "feature_canvas_init_host_ms":context["feature_canvas_init_host_ms"],
                       "feature_canvas_init_gpu_ms":canvas_gpu_ms,"planner_wall_cpu_ms_sum":planner_sum,
                       "candidate_search_cpu_ms_sum":sum(r["candidate_search_cpu_ms"] for r in cam_rows),
                       "adjacency_build_cpu_ms_sum":sum(r["adjacency_build_cpu_ms"] for r in cam_rows),
                       "union_find_cpu_ms_sum":sum(r["union_find_cpu_ms"] for r in cam_rows),
                       "rectangle_partition_cpu_ms_sum":sum(r["rectangle_partition_cpu_ms"] for r in cam_rows),
                       "cost_aware_merge_cpu_ms_sum":sum(r["cost_aware_merge_cpu_ms"] for r in cam_rows),
                       "resnet_gpu_ms_sum":sum(r["resnet_gpu_ms"] for r in cam_rows),
                       "fpn_gpu_ms_sum":sum(r["fpn_gpu_ms"] for r in cam_rows),
                       "crop_pack_gpu_ms_sum":sum(r["crop_pack_gpu_ms"] for r in cam_rows),
                       "scatter_gpu_ms_sum":sum(r["scatter_gpu_ms"] for r in cam_rows),
                       "tile_input_area_ratio_pct_sum":100.0*sum(r["tile_input_area_px"] for r in cam_rows)/
                           float(sum(r["image_h"]*r["image_w"] for r in cam_rows)),
                       "tile_input_valid_area_ratio_pct_sum":100.0*sum(r["tile_input_valid_area_px"] for r in cam_rows)/
                           float(sum(r["image_h"]*r["image_w"] for r in cam_rows)),
                       "exec_crop_area_ratio_pct_sum":100.0*sum(r["exec_crop_area_sum_px"] for r in cam_rows)/
                           float(sum(r["image_h"]*r["image_w"] for r in cam_rows)),
                       "batch_padding_waste_pct_mean":float(np.mean([r["batch_padding_waste_pct"] for r in cam_rows])),
                       "exec_crop_count_sum":sum(r["exec_crop_count"] for r in cam_rows),
                       "unique_tile_count_sum":sum(r["unique_tile_count"] for r in cam_rows),
                       "batch_tensor_bytes_sum":sum(r["batch_tensor_bytes"] for r in cam_rows),
                       "feature_canvas_bytes":cam_rows[0]["feature_canvas_bytes"],
                       "cuda_memory_allocated_before_mb":mem_before,
                       "cuda_allocated_after_mb":torch.cuda.memory_allocated()/1048576.0,
                       "cuda_peak_allocated_mb":torch.cuda.max_memory_allocated()/1048576.0,
                       "cuda_peak_reserved_mb":torch.cuda.max_memory_reserved()/1048576.0}
                frame_rows.append(frame)
                for cr in cam_rows:
                    cr.update({"sample_idx":sample_idx,"scene_token":scene_token,"e2e_forward_ms":e2e_ms,
                               "roi_block_gpu_stream_ms":roi_gpu_ms,"gridmask_gpu_ms":gridmask_gpu_ms,
                               "feature_canvas_init_gpu_ms":canvas_gpu_ms,
                               "cuda_peak_allocated_mb":frame["cuda_peak_allocated_mb"],
                               "cuda_peak_reserved_mb":frame["cuda_peak_reserved_mb"]})
                    camera_rows.append(cr)
                for cr in cand_rows:
                    cr.update({"frame_idx":frame_idx,"sample_idx":sample_idx,"scene_token":scene_token})
                    candidate_rows.append(cr)
                recorded+=1
                if recorded==1:
                    print("    measured: E2E={:.2f} ms ROI-GPU-stream={:.2f} ms planner-CPU={:.2f} ms".format(
                        e2e_ms,roi_gpu_ms,planner_sum),flush=True)
            if recorded!=args.frames_per_workload:
                raise RuntimeError("Expected {} measured frames, got {}".format(args.frames_per_workload,recorded))

    paired=_paired_rows(frame_rows)
    decision=_decision(frame_rows,args)
    summary=_summaries(frame_rows)
    _save_csv(os.path.join(args.out_dir,"roi_v2_camera_components.csv"),camera_rows)
    _save_csv(os.path.join(args.out_dir,"roi_v2_frame_components.csv"),frame_rows)
    _save_csv(os.path.join(args.out_dir,"roi_v2_candidates.csv"),candidate_rows)
    _save_csv(os.path.join(args.out_dir,"roi_v2_paired_e2e.csv"),paired)
    _save_csv(os.path.join(args.out_dir,"roi_v2_summary.csv"),summary)
    with open(os.path.join(args.out_dir,"roi_v2_decision.json"),"w",encoding="utf-8") as f: json.dump(decision,f,indent=2,ensure_ascii=False)
    _save_plots(frame_rows,paired,args.out_dir)
    print("\n[ROI-V2] DECISION: {}".format(decision["status"]),flush=True)
    print("[ROI-V2] ROI-Batch workload win rate vs best adjacency: {:.1%} ({}/{})".format(
        decision["roi_batch_workload_win_rate"],decision["roi_batch_workload_wins"],decision["workload_count"]),flush=True)
    print("[ROI-V2] recommendation: {}".format(decision["recommendation"]),flush=True)
    print("[ROI-V2] outputs: {}".format(os.path.abspath(args.out_dir)),flush=True)


if __name__=="__main__":
    main()
