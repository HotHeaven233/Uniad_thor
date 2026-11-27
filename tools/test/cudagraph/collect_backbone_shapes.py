#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect actual ResNet/FPN execution signatures over the UniAD test dataset.

The script runs the configured model's real E2E inference path, in dataset order,
and records every invocation of model.img_backbone and model.img_neck. It does
not synthesize ROIs, alter image tensors, or capture CUDA Graphs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import random
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'projects/configs').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Cannot locate UniAD root. Install under /home/uestc/uniad/tools/test/cudagraph/')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from shape_census_utils import describe_tree, first_tensor, jsonable, signature_id, summarize_calls


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', help='UniAD test config')
    p.add_argument('checkpoint', help='model checkpoint')
    p.add_argument('--out-dir', required=True, help='fresh output directory')
    p.add_argument('--max-frames', type=int, default=0,
                   help='0=all test frames; use a small positive value only for smoke test')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261127)
    p.add_argument('--no-tf32', action='store_true', help='disable TF32; default matches Thor TF32 config')
    p.add_argument('--adapter', default=None,
                   help='optional runtime adapter Python file; must expose install(model,cfg)')
    p.add_argument('--path-rows-attr', default='auto',
                   help='model attribute holding per-backbone-call metadata rows; auto checks common ROI row attributes')
    p.add_argument('--no-sync-each-frame', action='store_true',
                   help='skip per-frame CUDA sync; census is faster but async errors may be reported later')
    a = p.parse_args()
    if a.max_frames < 0 or a.log_every < 1:
        p.error('--max-frames must be >=0 and --log-every >=1')
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def command(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception as exc:
        return 'unavailable: {}'.format(exc)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def meta0(data):
    value = data.get('img_metas') if isinstance(data, dict) else None
    while True:
        if hasattr(value, 'data'):
            value = value.data
        elif isinstance(value, (tuple, list)) and value:
            value = value[0]
        else:
            return value if isinstance(value, dict) else {}


def scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def import_project_modules(cfg):
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin') and cfg.get('plugin_dir'):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))


def load_adapter(path, model, cfg):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location('cudagraph_shape_census_adapter', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import adapter: {}'.format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'install', None)):
        raise RuntimeError('Adapter must define install(model, cfg)')
    module.install(model, cfg)
    print('[CENSUS] runtime adapter installed: {}'.format(path), flush=True)
    return module


def source_info(obj, method_name):
    fn = getattr(obj, method_name, None)
    if fn is None:
        return None
    try:
        source_file = inspect.getsourcefile(fn)
        _, line = inspect.getsourcelines(fn)
        return {'file': source_file, 'first_line': int(line), 'qualname': getattr(fn, '__qualname__', repr(fn))}
    except Exception as exc:
        return {'file': None, 'first_line': None, 'qualname': getattr(fn, '__qualname__', repr(fn)), 'error': str(exc)}


def path_rows_snapshot(model, args):
    attrs = [] if args.path_rows_attr in ('', 'none', 'None') else (
        [args.path_rows_attr] if args.path_rows_attr != 'auto' else [
            '_roi_v2_camera_rows', '_roi_v3_camera_rows', '_roi_camera_rows', 'roi_camera_rows'
        ]
    )
    snapshot = {}
    for attr in attrs:
        value = getattr(model, attr, None)
        try:
            content = json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False)
        except Exception:
            content = repr(value)
        snapshot[attr] = (id(value), content)
    return attrs, snapshot


def find_path_rows(model, adapter, args, frame_info, call_count, snapshot):
    rows = None
    if adapter is not None and callable(getattr(adapter, 'get_backbone_call_metadata', None)):
        rows = adapter.get_backbone_call_metadata(model, frame_info)
    if rows is None:
        attrs, before = snapshot
        for attr in attrs:
            value = getattr(model, attr, None)
            try:
                content = json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False)
            except Exception:
                content = repr(value)
            old_id, old_content = before[attr]
            # Accept only per-frame metadata that was replaced or changed by this forward.
            if (id(value) != old_id or content != old_content) and isinstance(value, (list, tuple)) and len(value) == call_count:
                rows = value
                break
    if rows is None:
        static = None
        for attr in ('_roi_v3_arm', '_roi_v2_arm', '_roi_arm', 'backbone_path'):
            value = getattr(model, attr, None)
            if isinstance(value, (str, int, float)):
                static = str(value)
                break
        return [({'path_label': static} if static else {'path_label': 'native_or_unlabelled'})
                for _ in range(call_count)]
    if len(rows) != call_count:
        raise RuntimeError('adapter/path metadata rows ({}) do not match backbone calls ({}) at dataset index {}'
                           .format(len(rows), call_count, frame_info['dataset_index']))
    out = []
    for row in rows:
        row = jsonable(row if isinstance(row, dict) else {'value': row})
        label = None
        for key in ('chosen_mode', 'path_label', 'path', 'mode', 'arm', 'selected_mode'):
            if row.get(key) is not None:
                label = str(row[key])
                break
        row['path_label'] = label or 'adapter_unlabelled'
        out.append(row)
    return out


class ShapeHooks:
    def __init__(self, model):
        self.model = model
        self.current = None
        self.handles = []
        if not hasattr(model, 'img_backbone'):
            raise RuntimeError('model has no img_backbone module')
        self.handles.append(model.img_backbone.register_forward_pre_hook(self._backbone_pre))
        self.handles.append(model.img_backbone.register_forward_hook(self._backbone_post))
        if getattr(model, 'with_img_neck', hasattr(model, 'img_neck')) and hasattr(model, 'img_neck'):
            self.handles.append(model.img_neck.register_forward_pre_hook(self._fpn_pre))
            self.handles.append(model.img_neck.register_forward_hook(self._fpn_post))

    def begin_frame(self, frame_info):
        self.current = {'frame_info': frame_info, 'calls': []}

    def _backbone_pre(self, module, inputs):
        if self.current is None:
            return
        x = first_tensor(inputs)
        if x is None:
            raise RuntimeError('img_backbone was called without a Tensor input')
        call = {
            'call_index': len(self.current['calls']),
            'input': describe_tree(inputs),
            'input_tensor': {
                'shape': [int(v) for v in x.shape],
                'stride': [int(v) for v in x.stride()],
                'dtype': str(x.dtype),
                'device': str(x.device),
                'layout': str(x.layout),
                'memory_format': ('channels_last' if x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last)
                                  else ('contiguous' if x.is_contiguous() else 'non_contiguous')),
            },
            'resnet_output': None,
            'fpn_input': None,
            'fpn_output': None,
        }
        self.current['calls'].append(call)

    def _backbone_post(self, module, inputs, output):
        if self.current is None or not self.current['calls']:
            return
        self.current['calls'][-1]['resnet_output'] = describe_tree(output)

    def _fpn_pre(self, module, inputs):
        if self.current is None or not self.current['calls']:
            return
        self.current['calls'][-1]['fpn_input'] = describe_tree(inputs)

    def _fpn_post(self, module, inputs, output):
        if self.current is None or not self.current['calls']:
            return
        self.current['calls'][-1]['fpn_output'] = describe_tree(output)

    def end_frame(self):
        result = self.current
        self.current = None
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ['empty']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        'frame_summary.csv', 'frame_shape_trace.csv', 'shape_histogram.csv',
        'path_shape_summary.csv', 'shape_manifest.json', 'census_report.json',
        'environment.json', 'COMPLETE'
    ]
    existing = [name for name in expected_outputs if (out / name).exists()]
    if existing:
        raise FileExistsError('output files already exist in {}: {}; use a fresh --out-dir'.format(out, existing))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this script is intended for Thor')

    seed_everything(args.seed)
    tf32 = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError('config not found: {}'.format(cfg_path))
    if not ckpt_path.is_file():
        raise FileNotFoundError('checkpoint not found: {}'.format(ckpt_path))

    cfg = Config.fromfile(str(cfg_path))
    if cfg.get('fp16', None) is not None:
        raise RuntimeError('FP16 config is not supported by this census; use the TF32 FP32 config')
    torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
    import_project_modules(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    test_cfg = cfg.data.test
    if isinstance(test_cfg, (list, tuple)):
        raise RuntimeError('This script currently expects one test dataset config, not a list/ConcatDataset config')
    test_cfg.test_mode = True
    samples_per_gpu = int(test_cfg.pop('samples_per_gpu', 1))
    if samples_per_gpu != 1:
        raise RuntimeError('Shape census requires samples_per_gpu=1 to preserve UniAD sequential state')

    print('[CENSUS] building test dataset...', flush=True)
    dataset = build_dataset(test_cfg)
    dataset_length = len(dataset)
    total = min(dataset_length, args.max_frames) if args.max_frames else dataset_length
    if total <= 0:
        raise RuntimeError('test dataset is empty')

    print('[CENSUS] building model/checkpoint...', flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, str(ckpt_path), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', getattr(dataset, 'CLASSES', None))
    model.eval().cuda()
    adapter = load_adapter(args.adapter, model, cfg)

    # Hooks are attached to the actual modules called by the configured E2E model.
    hooks = ShapeHooks(model)
    net = MMDataParallel(model, device_ids=[torch.cuda.current_device()])

    backbone_source = source_info(model, 'extract_img_feat')
    env = {
        'config': str(cfg_path),
        'config_sha256': sha256_file(cfg_path),
        'checkpoint': str(ckpt_path),
        'checkpoint_size_bytes': ckpt_path.stat().st_size,
        'checkpoint_mtime': ckpt_path.stat().st_mtime,
        'repo_root': str(ROOT),
        'git_commit': command(['git', 'rev-parse', 'HEAD']),
        'git_status': command(['git', 'status', '--short']),
        'gpu': torch.cuda.get_device_name(torch.cuda.current_device()),
        'torch': str(torch.__version__),
        'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'fp16_config': False,
        'dataset_class': type(dataset).__name__,
        'dataset_length': dataset_length,
        'processed_frames': total,
        'dataset_test_config': repr(test_cfg),
        'adapter': str(Path(args.adapter).resolve()) if args.adapter else None,
        'extract_img_feat_source': backbone_source,
        'img_backbone_type': type(model.img_backbone).__name__ if hasattr(model, 'img_backbone') else None,
        'img_neck_type': type(model.img_neck).__name__ if hasattr(model, 'img_neck') else None,
        'nvpmodel': command(['nvpmodel', '-q']),
        'jetson_clocks': command(['jetson_clocks', '--show']),
        'uname': command(['uname', '-a']),
    }
    (out / 'environment.json').write_text(json.dumps(env, indent=2, ensure_ascii=False))
    try:
        (out / 'resolved_config.py').write_text(cfg.pretty_text)
    except Exception:
        pass

    frame_rows = []
    call_rows = []
    scene_counts = Counter()
    path_counts = Counter()
    scene_last_timestamp = {}
    noncontiguous_reentries = []
    seen_scenes = set()
    last_scene = None
    frame_start = time.perf_counter()

    print('[CENSUS] GPU={} torch={} cuda={} TF32={} dataset={} frames={}'.format(
        env['gpu'], env['torch'], env['cuda'], tf32, env['dataset_class'], total), flush=True)
    print('[CENSUS] extract_img_feat source={}'.format(backbone_source), flush=True)
    print('[CENSUS] Running real E2E forward in dataset index order; no ROI synthesis and no CUDA Graph capture.', flush=True)

    try:
        for idx in range(total):
            t_load = time.perf_counter()
            sample_data = dataset[idx]
            batch = collate([sample_data], samples_per_gpu=1)
            data_load_ms = (time.perf_counter() - t_load) * 1000.0
            meta = meta0(batch)
            scene = str(meta.get('scene_token', 'unknown_scene'))
            sample = str(meta.get('sample_idx', meta.get('token', idx)))
            timestamp = scalar(meta.get('timestamp'))
            if isinstance(timestamp, list) and len(timestamp) == 1:
                timestamp = timestamp[0]
            frame_info = {
                'dataset_index': idx,
                'scene_token': scene,
                'sample_token': sample,
                'timestamp': timestamp,
            }
            if scene != last_scene:
                if scene in seen_scenes:
                    noncontiguous_reentries.append({'dataset_index': idx, 'scene_token': scene})
                seen_scenes.add(scene)
                last_scene = scene
            scene_counts[scene] += 1

            # The model's own simple_test_track resets temporal state when scene_token changes.
            # Do not mutate model state. Snapshot optional path metadata so stale rows
            # from the previous frame are not mistaken for current-frame decisions.
            path_snapshot = path_rows_snapshot(model, args)
            if adapter is not None and callable(getattr(adapter, 'before_forward', None)):
                adapter.before_forward(model, frame_info)
            hooks.begin_frame(frame_info)
            torch.cuda.synchronize()
            t_forward = time.perf_counter()
            with torch.no_grad():
                result = net(return_loss=False, rescale=True, **batch)
            forward_host_ms = (time.perf_counter() - t_forward) * 1000.0
            if not args.no_sync_each_frame:
                torch.cuda.synchronize()
            else:
                # Synchronize at least periodically to surface async failures and drain work.
                if (idx + 1) % 32 == 0 or idx + 1 == total:
                    torch.cuda.synchronize()
            forward_wall_ms = (time.perf_counter() - t_forward) * 1000.0
            frame_capture = hooks.end_frame()
            calls = frame_capture['calls']
            path_rows = find_path_rows(model, adapter, args, frame_info, len(calls), path_snapshot)

            frame_signature_ids = []
            for call, path_meta in zip(calls, path_rows):
                call['path_metadata'] = jsonable(path_meta)
                call['path_label'] = str(path_meta.get('path_label', 'native_or_unlabelled'))
                path_counts[call['path_label']] += 1
                signature = {
                    'backbone_input': call['input'],
                    'resnet_output': call['resnet_output'],
                    'fpn_input': call['fpn_input'],
                    'fpn_output': call['fpn_output'],
                    'execution_context': {
                        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
                        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
                        'autocast_cuda': False,
                        'model_training': bool(model.training),
                    },
                }
                sid = signature_id(signature)
                frame_signature_ids.append(sid)
                row = {
                    **frame_info,
                    'call_index': call['call_index'],
                    'path_label': call['path_label'],
                    'path_metadata_json': json.dumps(call['path_metadata'], sort_keys=True, ensure_ascii=False),
                    'signature_id': sid,
                    'input_shape': json.dumps(call['input_tensor']['shape']),
                    'input_stride': json.dumps(call['input_tensor']['stride']),
                    'input_dtype': call['input_tensor']['dtype'],
                    'input_device': call['input_tensor']['device'],
                    'input_memory_format': call['input_tensor']['memory_format'],
                    'resnet_output_json': json.dumps(call['resnet_output'], sort_keys=True),
                    'fpn_input_json': json.dumps(call['fpn_input'], sort_keys=True),
                    'fpn_output_json': json.dumps(call['fpn_output'], sort_keys=True),
                    'signature_json': json.dumps(signature, sort_keys=True),
                }
                call_rows.append(row)

            frame_rows.append({
                **frame_info,
                'data_load_ms': round(data_load_ms, 4),
                'forward_host_ms': round(forward_host_ms, 4),
                'forward_wall_ms': round(forward_wall_ms, 4),
                'backbone_call_count': len(calls),
                'unique_signature_count': len(set(frame_signature_ids)),
                'signature_ids': json.dumps(frame_signature_ids),
            })

            del result, batch, sample_data
            if (idx + 1) % args.log_every == 0 or idx + 1 == total:
                elapsed = time.perf_counter() - frame_start
                print('[CENSUS] {}/{} frames | backbone_calls={} | unique_shapes={} | elapsed={:.1f} min | scene={}'.format(
                    idx + 1, total, len(call_rows), len(set(r['signature_id'] for r in call_rows)),
                    elapsed / 60.0, scene), flush=True)

    except KeyboardInterrupt:
        print('[CENSUS] interrupted by user; writing partial outputs (no COMPLETE marker)', flush=True)
        raise
    finally:
        hooks.close()
        torch.cuda.synchronize()
        write_csv(out / 'frame_summary.csv', frame_rows)
        write_csv(out / 'frame_shape_trace.csv', call_rows)

    # Exact-signature aggregation. The signature deliberately excludes path label:
    # identical Tensor/FPN signatures can share one graph across ROI-Batch/Adjacency.
    aggregate_input = []
    for row in call_rows:
        aggregate_input.append({
            'signature_id': row['signature_id'],
            'signature': json.loads(row['signature_json']),
            'path_label': row['path_label'],
            'dataset_index': row['dataset_index'],
        })
    signatures, aggregate_paths = summarize_calls(aggregate_input)

    histogram_rows = []
    path_hist = defaultdict(Counter)
    for item in signatures:
        histogram_rows.append({
            'signature_id': item['signature_id'],
            'count': item['count'],
            'frequency_pct': 100.0 * item['frequency'],
            'path_counts_json': json.dumps(item['path_counts'], sort_keys=True),
            'input_shape': json.dumps(item['signature']['backbone_input']),
            'resnet_output': json.dumps(item['signature']['resnet_output'], sort_keys=True),
            'fpn_output': json.dumps(item['signature']['fpn_output'], sort_keys=True),
        })
        for path, count in item['path_counts'].items():
            path_hist[path][item['signature_id']] += count
    write_csv(out / 'shape_histogram.csv', histogram_rows)

    path_rows_out = []
    for path, counter in sorted(path_hist.items()):
        for sid, count in counter.most_common():
            sig = next(x['signature'] for x in signatures if x['signature_id'] == sid)
            path_rows_out.append({
                'path_label': path,
                'signature_id': sid,
                'count': count,
                'frequency_within_path_pct': 100.0 * count / max(1, sum(counter.values())),
                'input_shape': json.dumps(sig['backbone_input']),
                'fpn_output': json.dumps(sig['fpn_output'], sort_keys=True),
            })
    write_csv(out / 'path_shape_summary.csv', path_rows_out)

    total_calls = len(call_rows)
    top_coverages = {}
    for k in (1, 3, 5, 10, 20, 50):
        top_coverages[str(k)] = (sum(x['count'] for x in signatures[:k]) / total_calls) if total_calls else 0.0
    no_call_frames = [r for r in frame_rows if r['backbone_call_count'] == 0]
    multi_call_frames = [r for r in frame_rows if r['backbone_call_count'] > 1]
    report = {
        'status': 'complete',
        'dataset_length': dataset_length,
        'processed_frames': len(frame_rows),
        'backbone_invocations': total_calls,
        'unique_execution_signatures': len(signatures),
        'path_call_counts': aggregate_paths,
        'top_k_signature_coverage': top_coverages,
        'frames_without_backbone_call': len(no_call_frames),
        'frames_with_multiple_backbone_calls': len(multi_call_frames),
        'max_backbone_calls_per_frame': max([r['backbone_call_count'] for r in frame_rows] or [0]),
        'scene_count': len(scene_counts),
        'scene_frame_counts': dict(scene_counts),
        'noncontiguous_scene_reentries': noncontiguous_reentries,
        'top_signatures': signatures[:20],
        'important_note': (
            'One exact signature is a candidate graph key, not proof that the module is CUDA-graph-capturable. '
            'Capture compatibility, static buffer lifetime, output aliasing, custom CUDA ops, and graph memory '
            'must be validated in Phase 2. Path labels are only authoritative when provided by an adapter or '
            'runtime camera-row attribute; native_or_unlabelled means the model did not expose per-call path metadata.'
        ),
    }
    manifest = {
        'schema_version': 1,
        'manifest_type': 'actual_e2e_backbone_fpn_execution_signatures',
        'graph_reuse_policy': 'same exact input/ResNet-output/FPN-output signature may share a captured graph across paths',
        'global_context': env,
        'signatures': signatures,
    }
    (out / 'shape_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / 'census_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / 'COMPLETE').write_text('ok\n')

    print('[CENSUS] COMPLETE', flush=True)
    print('[CENSUS] processed_frames={} backbone_calls={} unique_signatures={}'.format(
        len(frame_rows), total_calls, len(signatures)), flush=True)
    print('[CENSUS] top-K coverage={}'.format(json.dumps(top_coverages, sort_keys=True)), flush=True)
    print('[CENSUS] outputs={}'.format(out), flush=True)


if __name__ == '__main__':
    main()
