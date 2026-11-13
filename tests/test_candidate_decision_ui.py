"""Read-only raw/final display tests. RF.fit is forbidden, including in these tests."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from tests import test_dashboard_v2 as fixtures
from vibration.candidate_analysis import analyze_with_engine, candidate_analysis, UNCLEAR
from vibration.candidate_decision_ui import decision_view
from vibration.candidate_engine import CandidateEngine, decide, sha256
from vibration.candidate_history import read_candidate
from vibration.dashboard_v2 import analyze, configuration
from vibration.history import HistoryStore


class DecisionUITests(unittest.TestCase):
    app = fixtures.DashboardV2Tests.app
    upload = fixtures.DashboardV2Tests.upload

    def setUp(self):
        fixtures.DashboardV2Tests.setUp(self)
        guard = patch('sklearn.ensemble.RandomForestClassifier.fit', side_effect=AssertionError('RF retraining forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def ae(self):
        return analyze(self.raw, 900, 'measurement.mat', configuration())

    def fake_prediction(self, raw='Bearing', reasons=None):
        return {'kind':'RandomForest', 'model_version':'synthetic-display-fixture', 'raw_prediction':raw,
                'accepted':False, 'labels_enabled':False, 'reasons':reasons or ['자동 후보명 표시 비활성화'],
                'policy_rule':None, 'score_margin':.5,
                'scores_uncalibrated':{'Normal':.1,'Bearing':.6,'Looseness':.2,'Unbalance':.1}}

    def test_reference_is_recorded_class_not_final_label_or_score_argmax(self):
        report={'candidate':UNCLEAR,'status':'abstained','engine':self.fake_prediction(), 'reasons':['정책 보류']}
        report['engine']['scores_uncalibrated']={'Normal':.9,'Bearing':.1}
        before=deepcopy(report)
        view=decision_view(report)
        self.assertEqual(view['raw_class'],'Bearing')
        self.assertEqual(view['raw_label'],'베어링 결함')
        self.assertEqual(view['final_candidate'],UNCLEAR)
        self.assertEqual(report,before)
        report['engine']=None
        report['candidate']='베어링 결함 의심'
        self.assertIsNone(decision_view(report)['raw_class'])

    def test_legacy_and_skipped_records_do_not_fabricate_raw_prediction(self):
        self.assertIsNone(decision_view({'candidate':UNCLEAR,'status':'unclear'})['raw_class'])
        skipped={'candidate':'AE 정상 · 후보 추정 생략','status':'skipped_normal','engine':self.fake_prediction()}
        self.assertEqual(decision_view(skipped)['raw_label'],'실행 생략')
        self.assertIsNone(decision_view(skipped)['raw_class'])

    def test_normal_still_skips_rf(self):
        result,_=self.ae()
        result['models']['autoencoder'].update(file_threshold=1e10,predicted_anomaly=0)
        with patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('must skip')):
            report,_=analyze_with_engine(self.raw,900,result)
        self.assertIsNone(decision_view(report)['raw_class'])
        self.assertEqual(report['status'],'skipped_normal')

    def test_disabled_final_still_displays_raw_and_round_trips_db_history(self):
        expected,_=self.ae()
        with self.upload(), patch.dict(os.environ, {'ISU_CHEMICAL_CANDIDATE_BUNDLE':str(self.root/'fixture'),
                                                   'ISU_CHEMICAL_CANDIDATE_LABELS':'0'}), \
                patch('vibration.candidate_analysis.CandidateEngine') as engine:
            engine.return_value.predict.return_value=self.fake_prediction()
            app=self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result']['models'],expected['models'])
            self.assertTrue(any(m.label=='RF 원시 예측' and m.value=='베어링 결함' for m in app.metric))
            self.assertTrue(any(m.label=='결함 후보' and m.value==UNCLEAR for m in app.metric))
            report=deepcopy(app.session_state['v2_candidate'])
            self.assertFalse(report['engine']['labels_enabled'])
            self.assertTrue(report['abstained'])
            app.button(key='v2_save').click().run()
            saved=read_candidate(HistoryStore(self.db),1)
            self.assertEqual(saved,report)
            app.radio(key='v2_page').set_value('분석 이력').run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label=='RF 원시 예측' and m.value=='베어링 결함' for m in app.metric))
            self.assertEqual(read_candidate(HistoryStore(self.db),1),saved)

    def test_actual_policy_reasons_pass_through_unchanged(self):
        bounds={'min':[0.]*9,'max':[2.]*9}
        policy={'rules':{'900:Bearing':{'margin_min':.8}}}
        cases=[('Normal',.9,[1.]*9,True,policy),('Bearing',.1,[1.]*9,True,policy),
               ('Bearing',.9,[10.,*([1.]*8)],True,policy),('Bearing',.9,[1.]*9,True,{'rules':{}}),
               ('Bearing',.9,[1.]*9,False,policy)]
        for raw,margin,values,enabled,p in cases:
            accepted,reasons,rule=decide(raw,margin,values,900,bounds,p,[900],enabled)
            report={'candidate':UNCLEAR,'status':'abstained','engine':self.fake_prediction(raw,reasons),
                    'abstained':not accepted,'reasons':list(reasons)}
            before=deepcopy(report)
            self.assertEqual(decision_view(report)['reasons'],tuple(reasons))
            self.assertEqual(report,before)

    def test_missing_model_has_no_raw_and_ae_is_unchanged(self):
        result,_=self.ae()
        before=deepcopy(result)
        with patch.dict(os.environ,{'ISU_CHEMICAL_CANDIDATE_BUNDLE':str(self.root/'missing')}):
            report,_=analyze_with_engine(self.raw,900,result)
        self.assertEqual(result,before)
        self.assertIsNone(decision_view(report)['raw_class'])
        self.assertEqual(report['candidate'],UNCLEAR)
        self.assertTrue(any('FileNotFoundError' in r for r in report['reasons']))

    def test_legacy_payload_remains_readable_without_schema_change(self):
        result,versions=self.ae()
        report,_=candidate_analysis(self.raw,900)
        store=HistoryStore(self.db)
        store.save(result,versions,candidate_report=report)
        before=self.db.read_bytes()
        self.assertIsNone(decision_view(read_candidate(store,1))['raw_class'])
        self.assertEqual(self.db.read_bytes(),before)

    @unittest.skipUnless(os.environ.get('ISU_TEST_CANDIDATE_BUNDLE'),'read-only candidate artifact not supplied')
    def test_existing_rf_real_inference_filename_independent_no_fit(self):
        folder=Path(os.environ['ISU_TEST_CANDIDATE_BUNDLE'])
        hashes={p:sha256(p) for p in folder.iterdir() if p.is_file()}
        result,_=self.ae()
        observation,_=candidate_analysis(self.raw,900)
        actual=CandidateEngine(folder).predict(observation['features'],900,labels_enabled=False)
        with patch.dict(os.environ,{'ISU_CHEMICAL_CANDIDATE_BUNDLE':str(folder),'ISU_CHEMICAL_CANDIDATE_LABELS':'0'}):
            for name in ('Normal.mat','IR.mat','OR.mat','Roller.mat','Unbalance.mat','Looseness.mat'):
                input_result=deepcopy(result)
                input_result['input']['name']=name
                report,_=analyze_with_engine(self.raw,900,input_result)
                self.assertEqual(report['engine'],actual)
                self.assertEqual(decision_view(report)['raw_class'],actual['raw_prediction'])
                self.assertEqual(report['candidate'],UNCLEAR)
        self.assertEqual({p:sha256(p) for p in hashes},hashes)

    @unittest.skipUnless(os.environ.get('ISU_TEST_CANDIDATE_BUNDLE'),'read-only candidate artifact not supplied')
    def test_corrupt_copy_has_no_raw_prediction_original_artifact_unchanged(self):
        original=Path(os.environ['ISU_TEST_CANDIDATE_BUNDLE'])
        before=sha256(original/'model.joblib')
        copy=self.root/'corrupt-copy'
        shutil.copytree(original,copy)
        with (copy/'model.joblib').open('ab') as stream:
            stream.write(b'corrupt-copy-only')
        result,_=self.ae()
        with patch.dict(os.environ,{'ISU_CHEMICAL_CANDIDATE_BUNDLE':str(copy)}):
            report,_=analyze_with_engine(self.raw,900,result)
        self.assertIsNone(report['engine'])
        self.assertIsNone(decision_view(report)['raw_class'])
        self.assertEqual(report['candidate'],UNCLEAR)
        self.assertEqual(sha256(original/'model.joblib'),before)


if __name__=='__main__':
    unittest.main()
