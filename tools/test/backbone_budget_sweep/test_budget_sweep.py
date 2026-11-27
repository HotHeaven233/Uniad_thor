#!/usr/bin/env python3
import unittest
from launch_census_utils import raw_boxes

class BudgetSweepHelperTests(unittest.TestCase):
    def test_requested_grid_has_21_points(self):
        budgets = list(range(0, 101, 5))
        self.assertEqual(len(budgets), 21)
        self.assertEqual(budgets[0], 0)
        self.assertEqual(budgets[-1], 100)
        self.assertTrue(all((b % 5) == 0 for b in budgets))

    def test_five_percent_raw_geometry_is_nonempty_and_bounded(self):
        h, w = 928, 1600
        boxes, area = raw_boxes(h, w, 5.0, 12345, 'compact')
        self.assertTrue(boxes)
        self.assertGreater(area, 0)
        self.assertLessEqual(area, h * w * 0.05 + 1)

    def test_full_raw_geometry_stays_within_image(self):
        h, w = 928, 1600
        boxes, area = raw_boxes(h, w, 100.0, 12345, 'spread')
        self.assertTrue(boxes)
        self.assertLessEqual(area, h * w)
        for x0, y0, x1, y1 in boxes:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, w)
            self.assertLessEqual(y1, h)
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)

if __name__ == '__main__':
    unittest.main()
