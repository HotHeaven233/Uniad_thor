#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only unit tests for fixed-tile ROI mapping and adjacency merge."""
import unittest

from roi_tile_dispatch_runtime import (
    build_adjacency_matrix,
    build_execution_plan,
    build_fixed_tile_geometry,
    connected_components,
    merge_overlapping_component_boxes,
    summarize_execution_geometry,
    FRAME_FEATURE_NAMES,
    fit_ridge_model,
    choose_roi_execution_mode,
)


class FixedTileDispatchTests(unittest.TestCase):
    def test_tile_dimensions_are_max_roi_dims_aligned(self):
        boxes = [(0, 0, 100, 80), (200, 200, 292, 340)]
        g = build_fixed_tile_geometry(boxes, 512, 512, unit_align=32, feature_align=64)
        self.assertGreaterEqual(g["unit_w_32"], 100)
        self.assertGreaterEqual(g["unit_h_32"], 140)
        self.assertEqual(g["unit_w_32"] % 32, 0)
        self.assertEqual(g["unit_h_32"] % 32, 0)
        self.assertEqual(g["unit_w"] % 64, 0)
        self.assertEqual(g["unit_h"] % 64, 0)

    def test_all_intersecting_tiles_are_activated_and_deduplicated(self):
        boxes = [(0, 0, 100, 100), (50, 50, 150, 150)]
        g = build_fixed_tile_geometry(boxes, 512, 512, unit_align=32, feature_align=64)
        self.assertGreater(g["roi_tile_hits"], g["unique_tile_count"])
        self.assertEqual(g["tile_dedup_savings"], g["roi_tile_hits"] - g["unique_tile_count"])
        self.assertEqual(len(g["cells"]), len(set(g["cells"])))

    def test_adjacency_is_shared_edge_only_not_diagonal(self):
        cells = [(0, 0), (0, 1), (1, 1), (2, 2)]
        adjacency, edges = build_adjacency_matrix(cells)
        groups = connected_components(adjacency)
        self.assertEqual(edges, 2)
        self.assertEqual(sorted(sorted(g) for g in groups), [[0, 1, 2], [3]])
        self.assertEqual(int(adjacency[2, 3]), 0)  # diagonal contact is not adjacent

    def test_roi_batch_and_adjacency_use_same_deduplicated_tiles(self):
        boxes = [(30, 30, 180, 180), (100, 100, 240, 240), (350, 350, 470, 470)]
        g = build_fixed_tile_geometry(boxes, 512, 512)
        a = build_execution_plan(g, "roi_batch")
        b = build_execution_plan(g, "adjacency_merge")
        self.assertEqual(len(a["exec_boxes"]), g["unique_tile_count"])
        self.assertLessEqual(len(b["exec_boxes"]), len(a["exec_boxes"]))
        ga = summarize_execution_geometry(g, a)
        gb = summarize_execution_geometry(g, b)
        self.assertAlmostEqual(ga["merge_expansion_ratio"], 1.0)
        self.assertGreaterEqual(gb["merge_expansion_ratio"], 1.0)

    def test_overlapping_component_bboxes_are_merged_to_avoid_duplicate_compute(self):
        boxes = [(0, 0, 192, 192), (128, 128, 320, 320), (384, 0, 512, 128)]
        merged, count = merge_overlapping_component_boxes(boxes)
        self.assertEqual(count, 1)
        self.assertEqual(len(merged), 2)
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                x_overlap = max(0, min(merged[i][2], merged[j][2]) - max(merged[i][0], merged[j][0]))
                y_overlap = max(0, min(merged[i][3], merged[j][3]) - max(merged[i][1], merged[j][1]))
                self.assertEqual(x_overlap * y_overlap, 0)

    def test_runtime_selector_uses_profile_and_returns_geometry_without_dense_matrix(self):
        boxes = [(20, 20, 140, 140), (100, 100, 220, 220)]
        sample_rows = []
        for i in range(6):
            row = {name: float(i + 1) for name in FRAME_FEATURE_NAMES}
            row["roi_block_wall_ms"] = 10.0 + i
            sample_rows.append(row)
        model = fit_ridge_model(sample_rows, "roi_block_wall_ms", FRAME_FEATURE_NAMES, ridge=1e-2)
        profile = {
            "schema_version": 3,
            "benchmark_config": {"unit_align": 32, "feature_align": 64},
            "total_time_models": {"roi_batch": model, "adjacency_merge": model},
        }
        decision = choose_roi_execution_mode(profile, [boxes, boxes], 512, 512)
        self.assertIn(decision["selected_mode"], ("roi_batch", "adjacency_merge"))
        self.assertEqual(len(decision["camera_geometries"]), 2)
        self.assertNotIn("adjacency_matrix", decision["camera_geometries"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
