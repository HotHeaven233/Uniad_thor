import unittest
from launch_census_utils import raw_boxes,plan_for_budget,decide,coverage,MODES,LAYOUTS,BUDGETS

class CensusTests(unittest.TestCase):
    def test_budget_union_and_nesting(self):
        for h,w in ((928,1600),(97,130)):
            for layout in LAYOUTS:
                prev=set()
                for b in BUDGETS:
                    boxes,area=raw_boxes(h,w,b,43,layout)
                    cells=set()
                    for x,y,x1,y1 in boxes:
                        self.assertTrue(0<=x<x1<=w and 0<=y<y1<=h)
                        for r in range(y//64,(y1-1)//64+1):
                            for c in range(x//64,(x1-1)//64+1):
                                self.assertNotIn((r,c),cells);cells.add((r,c))
                    self.assertTrue(prev<=cells);prev=cells
                    self.assertLessEqual(abs(area-h*w*b/100),64*64)
                    self.assertEqual(area,sum((x1-x)*(y1-y) for x,y,x1,y1 in boxes))
                self.assertEqual(area,h*w)
    def fixture(self,times,areas):
        return [dict(mode=MODES[0],case_id=str(c),budget_pct=b,cnn_stream_ms=t,cnn_host_ms=t*.95,padded_pixels=a) for c in range(6) for b,t,a in zip((10,20,40),times,areas) for _ in range(2)]
    def test_bracket(self):
        d=decide(self.fixture([13,14,30],[100,200,400]),MODES[0],[10,20,40]);self.assertEqual(d['status'],'bracketed_plateau');self.assertEqual(d['selected_budgets'],[10,20]);self.assertEqual(d['upper_probe_budget'],40)
    def test_geometry_limited(self):
        d=decide(self.fixture([13,13,30],[100,100,400]),MODES[0],[10,20,40]);self.assertEqual(d['selected_budgets'],[]);self.assertEqual(d['comparisons'][0]['status'],'geometry_limited')
    def test_rising_at_minimum(self):
        d=decide(self.fixture([13,20,30],[100,200,400]),MODES[0],[10,20,40]);self.assertEqual(d['selected_budgets'],[])
    def test_all_flat(self):
        d=decide(self.fixture([13,13.5,14],[100,200,400]),MODES[0],[10,20,40]);self.assertEqual(d['status'],'plateau_continues')
    def test_coverage(self):self.assertEqual(coverage({'a':8,'b':2})['1'],.8)
    def test_hard_packed_cap(self):
        try:import roi_tile_dispatch_runtime
        except ImportError:self.skipTest('repo geometry unavailable')
        for l in LAYOUTS:
            for mo in MODES:
                for b in (10,20,40):
                    c=plan_for_budget(928,1600,b,99,l,mo)
                    self.assertLessEqual(c[-1]['padded_batch_area_px'],928*1600*b/100)
                    self.assertGreater(c[-1]['padded_batch_area_px'],0)
    def test_repo_plans(self):
        try:from roi_tile_dispatch_runtime import build_fixed_tile_geometry,build_execution_plan,summarize_execution_geometry
        except ImportError:self.skipTest('repo geometry unavailable')
        for layout in LAYOUTS:
            for b in (10,20,40):
                boxes,area=raw_boxes(928,1600,b,123,layout)
                g=build_fixed_tile_geometry(boxes,928,1600)
                self.assertEqual(g['raw_roi_union_area_px'],area)
                for mo in MODES:
                    s=summarize_execution_geometry(g,build_execution_plan(g,mo))
                    self.assertEqual(s['padded_batch_area_px'],s['exec_crop_count']*s['max_batch_crop_h']*s['max_batch_crop_w'])
                    self.assertGreaterEqual(s['exec_crop_area_sum_px'],area)

if __name__=='__main__':unittest.main()
