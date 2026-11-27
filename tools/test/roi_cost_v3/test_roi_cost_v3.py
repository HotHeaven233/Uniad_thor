import unittest, math, tempfile, json
from pathlib import Path
from adaptive_roi_dispatcher import A,B,D,choose,features,plan_row,search_plans
from roi_dispatch_v2_runtime import select_best_plan, overlap_area, box_area
from fit_roi_cost_v3 import ridge,predict,fit,online_report,write_csv

class TestV3(unittest.TestCase):
    def setUp(self):
        self.rois=[(20,30,200,180),(110,90,280,240),(410,100,550,270)]
    def model(self):
        r={'batch_size':4,'max_batch_h':128,'max_batch_w':128,'padded_batch_area_px':65536,'exec_crop_count':4,'exec_crop_area_sum_px':65536,'source_pad_added_pixels':0}
        m={'components':{},'shape_min':[1,64,64,4096],'shape_max':[100,1024,2048,10000000],'margin_ms':1.}
        for k in ('cnn','pack','scatter'):
            n=len(features(r,k));m['components'][k]={'intercept':2.,'mean':[0.]*n,'scale':[1.]*n,'coef':([1.]+[0.]*(n-1)) if k=='cnn' else [0.]*n}
        return m
    def test_shared_search_preserves_v2_choices(self):
        for arm,mode,cap in [(A,'roi_batch',0),(B,'adjacency_merge',.2)]:
            g,p,s,_=search_plans(self.rois,512,768,(.5,.75,.9,1),64,arm)
            ref=select_best_plan(self.rois,512,768,mode,(.5,.75,.9,1),cap,64)
            self.assertEqual(p['exec_boxes'],ref['plan']['exec_boxes']);self.assertEqual(g['cells'],ref['geometry']['cells'])
            for r in self.rois:self.assertEqual(sum(overlap_area(r,x) for x in p['exec_boxes']),box_area(r))
            for i,x in enumerate(p['exec_boxes']):self.assertTrue(all(overlap_area(x,y)==0 for y in p['exec_boxes'][i+1:]))
            self.assertEqual(p.get('merge_expansion_area_px',0),p['exec_crop_area_sum_px']-g['tile_input_area_px'])
    def test_shared_adaptive_plan_matches_fixed_choice(self):
        m=self.model();g,p,s,_=search_plans(self.rois,512,768,(.5,.75,.9,1),64,D,m)
        gg,pp,_,_=search_plans(self.rois,512,768,(.5,.75,.9,1),64,s['chosen_mode'])
        self.assertEqual(p['exec_boxes'],pp['exec_boxes']);self.assertEqual(g['cells'],gg['cells']);self.assertGreaterEqual(s['prediction_cpu_ms'],0)
    def test_decision_margin_and_support(self):
        m=self.model();g,p,_,_=search_plans(self.rois,512,768,(.5,.75,.9,1),64,A);a=plan_row(g,p);b=dict(a)
        self.assertEqual(choose(m,a,b)[0],A);m['shape_max']=[0,0,0,0];self.assertFalse(choose(m,a,b)[3])
    def test_ridge_serialization(self):
        m=ridge([[x] for x in range(10)],[2*x+3 for x in range(10)],.001)
        self.assertAlmostEqual(predict(json.loads(json.dumps(m)),[4]),11,places=3)
    def test_fit_and_online_schema(self):
        # Artificial timing fixture for plumbing only, NOT performance evidence.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);collect=root/'collect';online=root/'online';collect.mkdir();online.mkdir()
            cases=[]
            for i,split in enumerate(('train','calibration','test')):
                for j in range(2):cases.append(dict(workload_id=f'{i}_{j}',scene_token=str(i),split=split,dataset_index=i*2+j,roi_seed=i*20+j))
            man=dict(settings={'tf32':True},cases=cases)
            for folder,stage in [(collect,'collect'),(online,'online')]:
                cams=[];frames=[]
                for c in cases:
                    if stage=='online' and c['split']!='test':continue
                    for arm in ([A,B] if stage=='collect' else [A,B,D]):
                        common=dict(c,arm=arm,repeat=0,sample_idx=str(c['dataset_index']));total=0
                        for cam in range(6):
                            b=2+cam if arm==A else 1+cam;h=w=128;cnn=2+b*1.5
                            row=dict(common,camera=cam,batch_size=b,max_batch_h=h,max_batch_w=w,padded_batch_area_px=b*h*w,exec_crop_count=b,exec_crop_area_sum_px=b*h*w,source_pad_added_pixels=0,raw_roi_count=4,unique_tile_count=b,candidate_count=4,merge_pair_checks=0,planner_wall_cpu_ms=.3,resnet_gpu_ms=cnn,fpn_gpu_ms=1,pad_gpu_ms=.1,crop_pack_gpu_ms=.2,scatter_gpu_ms=.3)
                            cams.append(row);total+=cnn+1.6
                        frames.append(dict(common,e2e_forward_ms=total+400,roi_block_gpu_stream_ms=total+3,planner_wall_cpu_ms_sum=1.8,prediction_cpu_ms_sum=.2,cuda_peak_allocated_mb=1000,adjacency_cameras=3))
                (folder/'manifest.json').write_text(json.dumps(man));(folder/'environment.json').write_text(json.dumps(dict(stage=stage,repeats=1)));(folder/'COMPLETE').write_text('ok');write_csv(folder/'roi_v3_camera.csv',cams);write_csv(folder/'roi_v3_frame.csv',frames)
            fit(collect,root/'model',1.);online_report(online,root/'report')
            self.assertTrue((root/'model'/'roi_cost_model.json').is_file());self.assertTrue((root/'report'/'online_metrics.json').is_file())
if __name__=='__main__':unittest.main()
