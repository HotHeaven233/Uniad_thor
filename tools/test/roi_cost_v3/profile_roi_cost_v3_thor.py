#!/usr/bin/env python3
"""V3: scene-disjoint real UniAD E2E calibration and online comparison."""
from __future__ import annotations

import argparse
import hashlib
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

from pathlib import Path
from mmcv.parallel import collate
from adaptive_roi_dispatcher import A,B,D,SHAPE,search_plans,plan_row,predict_exec
from roi_dispatch_v2_runtime import build_grid_geometry
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

_REPO_ROOT = next((str(x) for x in __import__("pathlib").Path(__file__).resolve().parents if (x / "projects" / "configs").is_dir()), None)
if _REPO_ROOT is None: raise RuntimeError("Cannot locate UniAD repository")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

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
ARM_TO_MODE.update({D:(D,0.0),SHAPE:(SHAPE,0.0)})


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
        if cache_key not in self._roi_v2_box_cache and arm != SHAPE:
            self._roi_v2_box_cache[cache_key]=_extract_raw_boxes(workload,ncam,image_h,image_w,workload["roi_seed"],
                                                                  args.size_variation,args.min_roi_side)
        boxes_by_cam=self._roi_v2_box_cache.get(cache_key)

        roi_start_event=torch.cuda.Event(enable_timing=True); roi_start_event.record()
        roi_host_t0=time.perf_counter()
        camera_rows=[]; candidate_rows=[]; full_features=None; canvas_event=None; canvas_host_ms=0.0
        for cam in range(ncam):
            t_plan=time.perf_counter()
            if arm == SHAPE:
                geom,plan,search,search_rows=shape_plan(workload,image_h,image_w)
            else:
                raw_boxes=boxes_by_cam[cam]
                geom,plan,search,search_rows=search_plans(raw_boxes,image_h,image_w,args.quantiles,args.align,arm,args.cost_model if arm==D else None)
            planner_wall=(time.perf_counter()-t_plan)*1000
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
                "chosen_mode":search.get("chosen_mode",arm),
                "adjacency_search_cpu_ms":search.get("adjacency_search_cpu_ms",0.0),
                "merge_pair_checks":search.get("merge_pair_checks",0),
                "prediction_cpu_ms":search.get("prediction_cpu_ms",0.0),
                "pred_a_ms":search.get("pred_a_ms"),"pred_b_ms":search.get("pred_b_ms"),
                "both_shapes_supported":search.get("both_shapes_supported"),
                "exec_valid_area_px":exec_valid_area,
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



def shape_plan(case,h,w):
    b,hh,ww=case['shape']; cols=w//ww; rows=h//hh
    if b>cols*rows or cols<1: raise ValueError(f'Shape {case["shape"]} cannot fit image {h}x{w}')
    raw=[((i%cols)*ww,(i//cols)*hh,(i%cols+1)*ww,(i//cols+1)*hh) for i in range(b)]
    g=build_grid_geometry(raw,h,w,ww,hh,0,0,64); p=roi_batch_plan(g)
    assert (p['exec_crop_count'],p['max_batch_h'],p['max_batch_w'])==(b,hh,ww)
    return g,p,dict(candidate_count=1,candidate_search_cpu_ms=0.,chosen_mode=SHAPE),[]


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config');p.add_argument('checkpoint');p.add_argument('--out-dir',required=True)
    p.add_argument('--stage',choices=['collect','online'],default='collect')
    p.add_argument('--manifest');p.add_argument('--model-json')
    p.add_argument('--scenes',type=int,default=30);p.add_argument('--cases-per-scene',type=int,default=12)
    p.add_argument('--shape-cases-per-train-scene',type=int,default=4)
    p.add_argument('--warmup-frames',type=int,default=1);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261126);p.add_argument('--no-tf32',dest='tf32',action='store_false',default=True)
    p.add_argument('--roi-counts',default='1,2,4,8,16,32');p.add_argument('--area-fractions',default='0.02,0.05,0.10,0.20')
    p.add_argument('--aspect-ratios',default='0.25,0.5,1,2,4');p.add_argument('--layouts',default='spread,cluster,chain,random')
    p.add_argument('--candidate-quantiles',default='0.50,0.75,0.90,1.00');p.add_argument('--size-variation',type=float,default=.25)
    p.add_argument('--min-roi-side',type=int,default=64);p.add_argument('--align',type=int,default=64)
    return p.parse_args()


def make_manifest(dataset,args):
    groups=defaultdict(list)
    for i,info in enumerate(dataset.data_infos):
        if not info.get('scene_token'): raise RuntimeError('dataset.data_infos missing scene_token')
        groups[str(info['scene_token'])].append(i)
    rng=random.Random(args.seed)
    eligible=sorted(k for k,v in groups.items() if len(v)>=args.warmup_frames+1)
    rng.shuffle(eligible)
    if args.scenes<3 or args.scenes>len(eligible): raise ValueError(f'Need 3..{len(eligible)} scenes')
    chosen=eligible[:args.scenes]; ncal=max(1,args.scenes//5); ntest=max(1,args.scenes//5)
    ntrain=args.scenes-ncal-ntest
    grid=list(itertools.product(parse_csv_values(args.roi_counts,int),parse_csv_values(args.area_fractions,float),parse_csv_values(args.layouts,str),parse_csv_values(args.aspect_ratios,float)))
    rng.shuffle(grid)
    # feasible for the known 928x1600 images; actual shape validated at execution.
    shapes=[(b,h,w) for b,h,w in itertools.product([1,2,4,8,16,24,32,48,64],[64,128,192,256,384,512,768,896],[64,128,192,256,384,512,768,1024,1536]) if b<=(896//h)*(1600//w)]
    rng.shuffle(shapes)
    equal_area=[(4,256,384),(6,256,256),(16,192,128),(24,128,128),(32,64,192),(48,128,64)]
    shapes=equal_area+shapes
    cases=[]; gi=si=0
    for j,scene in enumerate(chosen):
        split='train' if j<ntrain else ('calibration' if j<ntrain+ncal else 'test')
        indices=groups[scene]; valid=indices[args.warmup_frames:]; offsets=list(range(len(valid)));rng.shuffle(offsets)
        nc=args.cases_per_scene+(args.shape_cases_per_train_scene if split=='train' else 0)
        for k in range(nc):
            n,a,l,ar=grid[gi%len(grid)];gi+=1
            pos=args.warmup_frames+offsets[k%len(offsets)]
            case=dict(workload_id=f'{split}_{j:03d}_{k:03d}',split=split,scene_token=scene,dataset_index=indices[pos],
                warmup_indices=indices[pos-args.warmup_frames:pos],roi_seed=rng.randrange(1,2**30),
                roi_count_target=n,area_fraction_target=a,layout=l,aspect_ratio_target=ar)
            if k>=args.cases_per_scene:
                case['shape']=shapes[si%len(shapes)];si+=1
            cases.append(case)
    return cases


def main():
    args=parse_args(); os.chdir(_REPO_ROOT)
    if args.align!=64 or args.repeats<1 or args.warmup_frames<1 or args.cases_per_scene<1: raise ValueError('invalid alignment/repeat/warmup/cases')
    args.quantiles=parse_csv_values(args.candidate_quantiles,float)
    args.cost_model=None
    if not torch.cuda.is_available(): raise RuntimeError('Run GPU profiler on Thor')
    if args.stage=='online' and (not args.manifest or not args.model_json): raise ValueError('online requires manifest and fitted model')
    if args.model_json: args.cost_model=json.loads(Path(args.model_json).read_text())
    if args.cost_model is not None and args.cost_model.get('schema_version')!=3: raise ValueError('Unsupported model schema')
    seed_everything(args.seed); torch.backends.cuda.matmul.allow_tf32=args.tf32;torch.backends.cudnn.allow_tf32=args.tf32
    cfg=Config.fromfile(args.config); torch.backends.cudnn.benchmark=bool(cfg.get('cudnn_benchmark',False))
    if cfg.get('fp16',None) is not None: raise ValueError('FP16 config is not supported in V3')
    import_project_modules(cfg);cfg.model.pretrained=None;cfg.model.train_cfg=None
    test_cfg=cfg.data.test;test_cfg.test_mode=True
    if int(test_cfg.pop('samples_per_gpu',1))!=1: raise ValueError('samples_per_gpu must be one')
    dataset=build_dataset(test_cfg)
    settings=dict(gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,tf32=args.tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,align=args.align,quantiles=args.quantiles,config=str(Path(args.config).resolve()),
        checkpoint=str(Path(args.checkpoint).resolve()),size_variation=args.size_variation,min_roi_side=args.min_roi_side)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'roi_v3_frame.csv').exists(): raise FileExistsError(f'Use a fresh output directory: {out}')
    if args.manifest:
        manifest=json.loads(Path(args.manifest).read_text());cases=manifest['cases']
        if manifest['settings']!=settings: raise ValueError('manifest hardware/config/settings differ')
    else:
        cases=make_manifest(dataset,args);manifest={'settings':settings,'cases':cases,'seed':args.seed}
    if args.stage=='online':
        if args.cost_model['settings']!=settings: raise ValueError('cost model settings mismatch')
        if args.cost_model['manifest_sha256']!=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(): raise ValueError('manifest differs from fitted model')
        cases=[c for c in cases if c['split']=='test' and 'shape' not in c]
        if set(c['scene_token'] for c in cases)!=set(args.cost_model['test_scenes']): raise ValueError('test scenes mismatch')
    if not cases: raise ValueError('no cases')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(settings,stage=args.stage,repeats=args.repeats),indent=2))
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'));checkpoint=load_checkpoint(model,args.checkpoint,map_location='cpu')
    model.CLASSES=checkpoint.get('meta',{}).get('CLASSES',dataset.CLASSES)
    model.eval().cuda();install_runtime_patch(model,args);model=MMDataParallel(model,device_ids=[torch.cuda.current_device()]);base=model.module
    def run_index(idx,seed):
        seed_everything(seed);data=collate([dataset[idx]],samples_per_gpu=1);meta=_get_first_meta(data)
        # Extract metadata before forward (UniAD mutates img_metas).
        sample=str(meta.get('sample_idx',idx));scene=str(meta.get('scene_token',''))
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.no_grad(): result=model(return_loss=False,rescale=True,**data)
        host=(time.perf_counter()-start)*1000;torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        return sample,scene,host,elapsed
    # Stream CSV per measured forward: results survive exceptions. Stable row schema enforced.
    writers={}; handles={}
    def append(name,rows):
        if not rows:return
        if name not in writers:
            f=open(out/name,'w',newline='',encoding='utf-8');handles[name]=f
            writers[name]=csv.DictWriter(f,fieldnames=list(rows[0]));writers[name].writeheader()
        writers[name].writerows(rows);handles[name].flush()
    try:
        for ci,case in enumerate(cases):
            idx=case['dataset_index'];actual_scene=str(dataset.data_infos[idx]['scene_token'])
            if actual_scene!=case['scene_token']: raise ValueError('dataset ordering changed')
            base._roi_v2_box_cache.clear();base._roi_v2_workload=case
            print(f'[ROI-V3] {ci+1}/{len(cases)} {case["workload_id"]} scene={actual_scene}',flush=True)
            arms=[SHAPE] if 'shape' in case else ([A,B,D] if args.stage=='online' else [A,B])
            for rep in range(args.repeats):
                order=arms.copy();random.Random(case['roi_seed']+rep*123).shuffle(order)
                for order_idx,arm in enumerate(order):
                    _reset_temporal_state(base);base._roi_v2_arm=arm
                    # Fresh identical temporal prefix for each measured arm. Repeated timings not independent.
                    for j,widx in enumerate(case['warmup_indices']):run_index(widx,case['roi_seed']+j)
                    start_epoch=time.time()
                    sample,scene,host,e2e=run_index(idx,case['roi_seed']+9999)
                    if scene and scene!=actual_scene: raise RuntimeError('metadata scene mismatch')
                    context=base._roi_v2_frame_context;cam_rows=[]
                    common=dict(split=case['split'],scene_token=actual_scene,sample_idx=sample,dataset_index=idx,repeat=rep,arm_order=order_idx,roi_seed=case['roi_seed'])
                    for c in base._roi_v2_camera_rows:
                        r=dict(c);ev=r.pop('_events')
                        r.update({k+'_gpu_ms':_event_ms(ev[v]) for k,v in [('pad','pad'),('crop_pack','pack'),('resnet','resnet'),('fpn','fpn'),('scatter','scatter')]});r.update(common);cam_rows.append(r)
                    if len(cam_rows)!=6: raise RuntimeError(f'Expected 6 camera rows, got {len(cam_rows)}')
                    frame=dict(workload_id=case['workload_id'],arm=arm,**common,measurement_start_epoch_s=start_epoch,measurement_end_epoch_s=time.time(),e2e_forward_ms=e2e,forward_host_enqueue_ms=host,
                        roi_block_gpu_stream_ms=_event_ms(context['_roi_block_events']),roi_block_host_enqueue_ms=context['roi_block_host_enqueue_ms'],
                        gridmask_gpu_ms=_event_ms(context['_gridmask_event']),canvas_gpu_ms=_event_ms(context['_canvas_event']),
                        gridmask_host_ms=context['gridmask_host_ms'],canvas_host_ms=context['feature_canvas_init_host_ms'],
                        cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/1048576,cuda_peak_reserved_mb=torch.cuda.max_memory_reserved()/1048576,
                        **{k+'_sum':sum(r[k] for r in cam_rows) for k in ('planner_wall_cpu_ms','adjacency_search_cpu_ms','prediction_cpu_ms','merge_pair_checks','pad_gpu_ms','crop_pack_gpu_ms','resnet_gpu_ms','fpn_gpu_ms','scatter_gpu_ms','padded_batch_area_px','batch_padding_area_px','exec_crop_area_sum_px','tile_input_area_px','source_pad_added_pixels','batch_tensor_bytes')},
                        adjacency_cameras=sum(r['chosen_mode']==B for r in cam_rows))
                    append('roi_v3_camera.csv',cam_rows);append('roi_v3_frame.csv',[frame]);print(f'  {arm} rep={rep} E2E={e2e:.2f} ROI={frame["roi_block_gpu_stream_ms"]:.2f} planner={frame["planner_wall_cpu_ms_sum"]:.2f}',flush=True)
    finally:
        for f in handles.values():f.close()
    (out/'COMPLETE').write_text('ok\n');print('[ROI-V3] COMPLETE',out,flush=True)

if __name__=='__main__':main()
