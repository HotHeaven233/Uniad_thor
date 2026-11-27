#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure helpers for CUDA Graph backbone shape census."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any


def _memory_format(tensor):
    if tensor.ndim == 4:
        try:
            if tensor.is_contiguous(memory_format=__import__('torch').channels_last):
                return 'channels_last'
        except Exception:
            pass
    try:
        if tensor.is_contiguous():
            return 'contiguous'
    except Exception:
        pass
    return 'non_contiguous'


def tensor_meta(tensor):
    """Return execution-relevant metadata without retaining the Tensor."""
    return {
        'shape': [int(x) for x in tensor.shape],
        'stride': [int(x) for x in tensor.stride()],
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'layout': str(tensor.layout),
        'memory_format': _memory_format(tensor),
        'is_contiguous': bool(tensor.is_contiguous()),
        'requires_grad': bool(tensor.requires_grad),
    }


def describe_tree(value: Any):
    """Describe nested tensor structures (tuple/list/dict) by shape only."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return {'type': 'tensor', **tensor_meta(value)}
    if isinstance(value, tuple):
        return {'type': 'tuple', 'items': [describe_tree(x) for x in value]}
    if isinstance(value, list):
        return {'type': 'list', 'items': [describe_tree(x) for x in value]}
    if isinstance(value, dict):
        return {'type': 'dict', 'items': {str(k): describe_tree(value[k]) for k in sorted(value, key=str)}}
    if value is None:
        return {'type': 'none'}
    return {'type': type(value).__name__}


def first_tensor(value):
    """Find first Tensor in a nested structure."""
    try:
        import torch
    except Exception:
        return None
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            found = first_tensor(value[key])
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def jsonable(value):
    """Convert adapter metadata to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    try:
        import torch
        if torch.is_tensor(value):
            if value.numel() <= 64:
                return value.detach().cpu().tolist()
            return {'tensor_shape': list(value.shape), 'dtype': str(value.dtype)}
    except Exception:
        pass
    return repr(value)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def signature_id(signature):
    return hashlib.sha256(canonical_json(signature).encode('utf-8')).hexdigest()[:16]


def summarize_calls(call_rows):
    """Aggregate invocation rows by exact execution signature and path."""
    by_sig = {}
    path_counts = Counter()
    for row in call_rows:
        sid = row['signature_id']
        path = row.get('path_label') or 'unlabelled'
        path_counts[path] += 1
        if sid not in by_sig:
            by_sig[sid] = {
                'signature_id': sid,
                'signature': row['signature'],
                'count': 0,
                'path_counts': Counter(),
                'first_seen': row.get('dataset_index'),
                'last_seen': row.get('dataset_index'),
            }
        item = by_sig[sid]
        item['count'] += 1
        item['path_counts'][path] += 1
        item['last_seen'] = row.get('dataset_index')
    results = []
    total = len(call_rows)
    for sid, item in by_sig.items():
        results.append({
            'signature_id': sid,
            'signature': item['signature'],
            'count': item['count'],
            'frequency': item['count'] / total if total else 0.0,
            'path_counts': dict(item['path_counts']),
            'first_seen_dataset_index': item['first_seen'],
            'last_seen_dataset_index': item['last_seen'],
        })
    results.sort(key=lambda x: (-x['count'], x['signature_id']))
    return results, dict(path_counts)
