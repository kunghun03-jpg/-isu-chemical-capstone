"""Synthetic V3 inference/UI/storage parity without fitting models."""
from copy import deepcopy
from pathlib import Path
import os
import unittest
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest
from tests import test_dashboard_v2 as fixtures
from vibration.dashboard_v2 import configuration,analyze
from vibration.dashboard_v3 import analyze_selected,references,rotation_orders,save_references,read_references
from vibration.candidate_analysis import analyze_with_engine
from vibration.history import HistoryStore

ROOT=Path(__file__).resolve().parents[1]

class V3Tests(unittest.TestCase):
    setUp=fixtures.DashboardV2Tests.setUp
    upload=fixtures.DashboardV2Tests.upload

    def app(self):
        return AppTest.from_file(str(ROOT/'streamlit_app_v3.py'),default_timeout=40).run()

    def result(self): return analyze_selected(self.raw,900,'measurement.mat',configuration())

    def test_exact_ae_parity_and_independent_missing_optional_artifact(self):
        old,versions=analyze(self.raw,900,'measurement.mat',configuration())
        new,new_versions=self.result()
        self.assertEqual(old['models'],new['models']);self.assertEqual(versions,new_versions)
        (self.bundles/'900rpm/ocsvm.joblib').unlink()
        current,_=self.result();self.assertEqual(new['models'],current['models'])
        with self.assertRaises(FileNotFoundError):analyze(self.raw,900,'measurement.mat',configuration())
        rows=references(self.raw,900,configuration(),new)
        self.assertEqual(next(r for r in rows if r['model']=='ocsvm')['status'],'모델 미연결')
        self.assertEqual(next(r for r in rows if r['model']=='iforest')['status'],'완료')

    def test_models_exact_saved_scores_and_policy_no_mutation(self):
        result,_=self.result();before=deepcopy(result)
        rows=references(self.raw,900,configuration(),result)
        for model in ('ocsvm','iforest'):
            expected,_=analyze(self.raw,900,'measurement.mat',__import__('vibration.dashboard_v2',fromlist=['DashboardConfig']).DashboardConfig(model,self.bundles))
            actual=next(r for r in rows if r['model']==model)
            self.assertEqual(actual['score'],expected['models'][model]['file_score'])
            self.assertEqual(actual['threshold'],expected['models'][model]['file_threshold'])
            self.assertEqual(actual['decision'],'이상' if actual['score']>actual['threshold'] else '정상')
        self.assertEqual(result,before)
        with patch('vibration.dashboard_v3.bundle_status',return_value={'ready':False,'label':'RPM 적용 범위 미확인'}),patch('vibration.dashboard_v3.predict_dae',side_effect=AssertionError('blocked')):
            self.assertEqual(references(self.raw,900,configuration(),result)[1]['status'],'RPM 적용 범위 미확인')

    def test_corrupt_optional_reports_error_and_selected_corruption_still_fails(self):
        result,_=self.result()
        with (self.bundles/'900rpm/ocsvm.joblib').open('ab') as f:f.write(b'bad')
        rows=references(self.raw,900,configuration(),result)
        self.assertEqual(next(r for r in rows if r['model']=='ocsvm')['status'],'실행 오류')
        new,_=self.result();self.assertEqual(new['models'],result['models'])
        with (self.bundles/'900rpm/autoencoder.pt').open('ab') as f:f.write(b'bad')
        with self.assertRaises(ValueError):self.result()

    def test_saved_references_duplicate_restart_and_old_history(self):
        result,versions=self.result();candidate,_=analyze_with_engine(self.raw,900,result)
        rows=references(self.raw,900,configuration(),result);store=HistoryStore(self.db)
        identity,created=save_references(store,result,versions,candidate,rows);self.assertTrue(created)
        self.assertFalse(save_references(store,result,versions,candidate,rows)[1])
        self.assertEqual(read_references(HistoryStore(self.db),identity),rows)
        import subprocess,sys,json
        code='import sys,json; from vibration.history import HistoryStore; from vibration.dashboard_v3 import read_references; print(json.dumps(read_references(HistoryStore(sys.argv[1]),int(sys.argv[2]))))'
        loaded=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(self.db),str(identity)],cwd=ROOT,text=True))
        self.assertEqual(loaded,rows)
        old=HistoryStore(self.root/'old.sqlite3');oldid,_=old.save(result,versions)
        self.assertIsNone(read_references(old,oldid))
        changed=deepcopy(rows);changed[0]['score']=123
        with self.assertRaises(ValueError):save_references(store,result,versions,candidate,changed)
        self.assertEqual(read_references(store,identity),rows)

    def test_orders_actual_peak_and_correct_nominal_frequencies(self):
        frequency=np.arange(0,201,.1);amplitude=np.zeros_like(frequency)
        for n in range(1,6):amplitude[round(n*20/.1)]=n
        frame=rotation_orders({'frequency':frequency,'amplitude':amplitude},1200)
        self.assertEqual(frame['명목 주파수 (Hz)'].tolist(),[20,40,60,80,100])
        self.assertEqual(frame['진폭 (raw)'].tolist(),[1,2,3,4,5])

    def test_app_upload_analyze_save_history_pages_and_input_reset(self):
        with self.upload():
            app=self.app();self.assertFalse(app.exception)
            app.button(key='v3_run').click().run();self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state['v3_rows']),4)
            app.button(key='v3_save').click().run();self.assertFalse(app.exception)
            app.radio(key='v2_page').set_value('분석 이력').run();self.assertFalse(app.exception)
            app.radio(key='v2_page').set_value('모델 정보').run();self.assertFalse(app.exception)
            self.assertTrue(any(i.value=='평가 자료 없음' for i in app.info))
            app.radio(key='v2_page').set_value('분석하기').run()
            app.selectbox(key='v2_rpm').set_value(1200).run()
            self.assertNotIn('v2_result',app.session_state);self.assertNotIn('v3_rows',app.session_state)

    def test_duplicate_run_disabled_and_file_change_clears_results(self):
        with self.upload():
            app=self.app();app.button(key='v3_run').click().run()
            self.assertTrue(app.button(key='v3_run').disabled)
        with self.upload(raw=b'invalid changed MAT'):
            app.run()
            self.assertNotIn('v2_result',app.session_state)
            self.assertNotIn('v3_rows',app.session_state)
            self.assertTrue(app.button(key='v3_run').disabled)

    def test_v3_rf_actual_returned_raw_and_policy_match_v2_path(self):
        from tests import test_v2_integration as fixture
        with self.upload(),patch.dict(os.environ,{'ISU_CHEMICAL_CANDIDATE_BUNDLE':str(self.root/'mock'),'ISU_CHEMICAL_CANDIDATE_LABELS':'0'}),patch('vibration.candidate_analysis.CandidateEngine') as engine:
            engine.return_value.predict.return_value=fixture.IntegrationTests.fake(self)
            app=self.app();app.button(key='v3_run').click().run()
            result=app.session_state['v2_result'];actual=app.session_state['v2_candidate']
            expected,_=analyze_with_engine(self.raw,900,result)
            self.assertEqual(actual,expected)
            self.assertEqual(actual['engine']['raw_prediction'],'Bearing')
            self.assertEqual(actual['candidate'],'결함 후보 불명확')
            self.assertTrue(actual['abstained'])


class DAEReferenceTests(unittest.TestCase):
    def setUp(self):
        from tests import test_dae_ui as fixture
        fixture.DAEUITests.setUp(self)

    def test_real_synthetic_dae_bundle_ready_and_rpm_block(self):
        from tests import test_dae_ui as fixture
        from vibration.dae_inference import SavedDAEBundleInference
        from vibration.dashboard_v2 import DashboardConfig
        fixture.DAEUITests.convert(self,known=True)
        config=DashboardConfig('autoencoder',self.folder/'missing_classic')
        ae={'models':{'autoencoder':{'file_score':0.,'file_threshold':1.,'predicted_anomaly':0}},'bundle_id':'synthetic'}
        expected=SavedDAEBundleInference(self.bundle).predict_bytes(self.raw,rpm=900)
        rows=references(self.raw,900,config,ae)
        dae=next(r for r in rows if r['model']=='dae')
        self.assertEqual(dae['status'],'완료');self.assertEqual(dae['score'],expected['mse'])
        self.assertEqual(dae['threshold'],expected['threshold']);self.assertEqual(dae['decision'],expected['prediction'])
        self.assertNotEqual(references(self.raw,1200,config,ae)[1]['status'],'완료')
