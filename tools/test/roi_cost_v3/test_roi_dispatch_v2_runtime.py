#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from roi_dispatch_v2_runtime import (
    build_grid_geometry, build_v1_fixed_geometry, generate_candidate_specs,
    sparse_adjacency, connected_components_sparse, exact_rectangle_partition,
    greedy_cost_aware_merge, roi_batch_plan, adjacency_plan, select_best_plan,
    assert_disjoint_cover, box_area,
)


class TestROIDispatchV2(unittest.TestCase):
    def setUp(self):
        self.image_h = 512
        self.image_w = 768
        self.boxes = [(20, 20, 170, 150), (100, 100, 250, 230),
                      (500, 20, 650, 150), (520, 150, 700, 300)]

    def test_phase_shifted_edges_are_aligned(self):
        specs = generate_candidate_specs(self.boxes, self.image_h, self.image_w,
                                          quantiles=(0.5, 0.75, 1.0), align=64)
        self.assertTrue(specs)
        for spec in specs:
            g = build_grid_geometry(self.boxes, self.image_h, self.image_w,
                                    spec["tile_w"], spec["tile_h"],
                                    spec["phase_x"], spec["phase_y"], 64)
            self.assertTrue(all(x % 64 == 0 for x in g["x_edges"]))
            self.assertTrue(all(y % 64 == 0 for y in g["y_edges"]))
            self.assertEqual(g["x_edges"][0], 0)
            self.assertEqual(g["y_edges"][0], 0)

    def test_roi_mapping_deduplicates_overlaps(self):
        boxes = [(10, 10, 120, 120), (20, 20, 100, 100)]
        g = build_grid_geometry(boxes, 256, 256, 128, 128, 0, 0, 64)
        self.assertEqual(g["roi_tile_hits"], 2)
        self.assertEqual(g["unique_tile_count"], 1)
        self.assertEqual(g["tile_dedup_savings"], 1)

    def test_sparse_adjacency_shared_edges_only(self):
        cells = [(0, 0), (0, 1), (1, 1), (3, 3)]
        edges, _ = sparse_adjacency(cells)
        self.assertEqual(len(edges), 2)
        groups = connected_components_sparse(len(cells), edges)
        self.assertEqual(sorted(map(len, groups)), [1, 3])

    def test_exact_partition_covers_active_cells_once(self):
        g = build_grid_geometry(self.boxes, self.image_h, self.image_w, 128, 128, 64, 64, 64)
        boxes = exact_rectangle_partition(g)
        assert_disjoint_cover(g, boxes)
        self.assertEqual(sum(box_area(b) for b in boxes), g["tile_input_area_px"])

    def test_adjacency_plan_never_overlaps_and_tracks_expansion(self):
        g = build_grid_geometry(self.boxes, self.image_h, self.image_w, 128, 128, 0, 0, 64)
        for cap in (0.0, 0.05, 0.10, 0.20):
            p = adjacency_plan(g, cap)
            # Exact active-cell coverage is guaranteed for cap=0. For expanded
            # merges, area may grow but execution rectangles remain disjoint.
            for i, a in enumerate(p["exec_boxes"]):
                for b in p["exec_boxes"][i+1:]:
                    ax=max(0,min(a[2],b[2])-max(a[0],b[0]))
                    ay=max(0,min(a[3],b[3])-max(a[1],b[1]))
                    self.assertEqual(ax*ay,0)
            self.assertLessEqual(p["merge_expansion_area_px"],
                                 int(g["tile_input_area_px"]*cap)+1)

    def test_candidate_search_returns_valid_plan(self):
        result = select_best_plan(self.boxes, self.image_h, self.image_w,
                                  "roi_batch", quantiles=(0.5, 0.75, 1.0), align=64)
        self.assertGreater(result["candidate_count"], 0)
        self.assertTrue(result["candidate_spec"]["candidate_id"])
        assert_disjoint_cover(result["geometry"], result["plan"]["exec_boxes"])
        self.assertGreaterEqual(result["candidate_search_cpu_ms"], 0)

    def test_adjacency_candidate_search_is_valid_and_disjoint(self):
        result = select_best_plan(self.boxes, self.image_h, self.image_w,
                                  "adjacency_merge", quantiles=(0.5, 0.75, 1.0),
                                  max_expansion=0.10, align=64)
        self.assertGreater(result["candidate_count"], 0)
        plan = result["plan"]
        self.assertGreaterEqual(plan["exec_crop_count"], 1)
        for i, a in enumerate(plan["exec_boxes"]):
            for b in plan["exec_boxes"][i+1:]:
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                self.assertEqual(ix * iy, 0)
        self.assertGreaterEqual(result["planner_components_total_ms"]["adjacency_build_cpu_ms"], 0)

    def test_v1_baseline_still_valid(self):
        g = build_v1_fixed_geometry(self.boxes, self.image_h, self.image_w, 64)
        p = roi_batch_plan(g)
        self.assertGreater(p["exec_crop_count"], 0)
        self.assertEqual(p["exec_crop_count"], g["unique_tile_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
