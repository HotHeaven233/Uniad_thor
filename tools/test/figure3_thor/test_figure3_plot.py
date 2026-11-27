"""CPU-only tests for interval attribution and report pipeline (no Thor timings)."""
import csv,json,sqlite3,subprocess,sys,tempfile,unittest
from pathlib import Path
from plot_figure3_thor import exclusive,union_length,write_csv,summarize

class Test(unittest.TestCase):
    def test_union(self):self.assertEqual(union_length([(0,5),(3,8),(10,12)]),10)
    def test_nested(self):
        def row(i,p,n,a,b):return dict(scope_id=i,parent_id=p,module=n,stream=0,host_start_ms=a,host_end_ms=b,gpu_start_ms=a,gpu_end_ms=b)
        r=[row(0,-1,'tracking',0,10),row(1,0,'bev_other',1,8),row(2,1,'backbone',1,5),row(3,1,'encoder',5,8)]
        self.assertEqual([v for s,v in exclusive(r,'gpu')],[3,0,4,3])
    def test_multiple_streams_rejected(self):
        r=[dict(scope_id=i,parent_id=-1,module='x',stream=i,gpu_start_ms=0,gpu_end_ms=1) for i in range(2)]
        with self.assertRaises(ValueError):exclusive(r,'gpu')
    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for profile in ('summary','plain','detail','area'):
                p=root/profile;p.mkdir();(p/'COMPLETE').touch()
                f=dict(frame_id='a',mode='full',profile=profile,scene='s',dataset_index=1,sample='sample',repeat=0,shape='',e2e_ms=20,forward_host_ms=16)
                write_csv(p/'frames.csv',[f])
                def row(i,parent,name,a,b,attrs={}):return dict(**f,scope_id=i,parent_id=parent,module=name,stream=0,host_start_ms=a,host_end_ms=b,gpu_start_ms=a,gpu_end_ms=b,attrs=json.dumps(attrs))
                rows=[row(0,-1,'tracking',0,10),row(1,0,'bev_other',1,8),row(2,1,'backbone',1,5),row(3,1,'encoder',5,8),row(4,-1,'planning',10,16)]
                if profile=='detail':rows.append(row(5,3,'tsa',5,6))
                if profile=='area':rows.append(row(5,2,'area_cnn',1,2,dict(camera=0,batch=1,height=64,width=64,pixels=4096)))
                if profile=='plain':(p/'scopes.csv').touch()
                else:write_csv(p/'scopes.csv',rows)
            self.assertAlmostEqual(summarize(root/'summary')[0]['tracking_host_ms'],3)
            db=root/'trace.sqlite';c=sqlite3.connect(db);c.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); INSERT INTO StringIds VALUES (1,"cudaLaunchKernel"); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,nameId INTEGER); INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,10000,1,1); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER); INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10000,20000,1);');c.commit();c.close()
            subprocess.run([sys.executable,str(Path(__file__).with_name('plot_figure3_thor.py')),'--summary',str(root/'summary'),'--detail',str(root/'detail'),'--area',str(root/'area'),'--plain',str(root/'plain'),'--trace-sqlite',str(db),'--out-dir',str(root/'report')],check=True)
            self.assertTrue((root/'report/figure3_thor.pdf').exists());self.assertTrue(json.loads((root/'report/report.json').read_text())['trace_available'])
if __name__=='__main__':unittest.main()
