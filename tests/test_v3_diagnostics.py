"""Synthetic model diagnostics; no real signal performance claim."""
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from tests import test_dashboard_v2 as fixture
from scripts.diagnose_v3_models import ocsvm_details,run
from vibration.v3_presentation import dae_diagnostics

class DiagnosticsTests(unittest.TestCase):
    setUp=fixture.DashboardV2Tests.setUp

    def test_exact_offline_v3_and_raw_ocsvm_math(self):
        result=run(self.path,900,self.bundles)
        self.assertTrue(result['input_unchanged'])
        self.assertTrue(all(result['models'][m]['exact_match'] for m in ('autoencoder','ocsvm','iforest')))
        oc=result['models']['ocsvm']['calculation']
        self.assertEqual(oc['file_score'],float(np.mean(-np.array(oc['score_samples']))))
        self.assertEqual(oc['predicted_anomaly'],int(oc['file_score']>oc['threshold']))

    def test_zero_score_negative_threshold_is_anomalous(self):
        engine=MagicMock();engine.preprocess.return_value=(None,np.zeros((8,1,3201)))
        engine._estimators={'ocsvm':MagicMock()};engine._estimators['ocsvm'].score_samples.return_value=np.zeros(8)
        engine.manifest={'models':{'ocsvm':{'file_threshold':-.37}}}
        engine.predict_bytes.return_value={'models':{'ocsvm':{'file_score':0.,'predicted_anomaly':1}}}
        with patch('scripts.diagnose_v3_models.SavedModelInference',return_value=engine):
            result=ocsvm_details(b'test',900,self.bundles)
        self.assertTrue(result['all_raw_scores_exactly_zero']);self.assertEqual(result['predicted_anomaly'],1)

    def test_missing_dae_reports_actual_folder_and_files(self):
        result=dae_diagnostics(900)
        self.assertEqual(result['label'],'모델 미준비')
        self.assertFalse(result['files_present']['manifest.json'])
        self.assertIn('configured_folder',result)
