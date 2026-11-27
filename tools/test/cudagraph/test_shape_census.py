#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
import torch
from shape_census_utils import describe_tree, first_tensor, signature_id, summarize_calls


class ShapeCensusTests(unittest.TestCase):
    def test_tensor_tree_signature(self):
        x = torch.zeros((2, 3, 256, 320), dtype=torch.float32)
        tree = describe_tree((x, [torch.zeros((2, 64, 32, 40))]))
        self.assertEqual(tree['type'], 'tuple')
        self.assertEqual(tree['items'][0]['shape'], [2, 3, 256, 320])
        self.assertEqual(tree['items'][0]['stride'], [245760, 81920, 320, 1])

    def test_first_tensor_nested(self):
        x = torch.ones((1, 3, 64, 64))
        self.assertIs(first_tensor({'x': [x]}), x)

    def test_signature_id_stable(self):
        a = {'shape': [1, 3, 128, 128], 'dtype': 'torch.float32'}
        b = {'dtype': 'torch.float32', 'shape': [1, 3, 128, 128]}
        self.assertEqual(signature_id(a), signature_id(b))

    def test_aggregate_exact_signatures_and_paths(self):
        sig = {'backbone_input': {'shape': [1, 3, 128, 128]}}
        sid = signature_id(sig)
        rows = [
            {'signature_id': sid, 'signature': sig, 'path_label': 'roi_batch', 'dataset_index': 0},
            {'signature_id': sid, 'signature': sig, 'path_label': 'adjacency', 'dataset_index': 1},
        ]
        agg, paths = summarize_calls(rows)
        self.assertEqual(len(agg), 1)
        self.assertEqual(agg[0]['count'], 2)
        self.assertEqual(paths, {'roi_batch': 1, 'adjacency': 1})


if __name__ == '__main__':
    unittest.main(verbosity=2)
