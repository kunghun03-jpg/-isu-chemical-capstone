"""Synthetic audit correctness; no real 50-file result is claimed."""
from pathlib import Path
import unittest
from unittest.mock import patch
from scipy.io import savemat
import numpy as np
from tests import test_dashboard_v2 as fixture
from scripts.analyze_unbalance_preliminary import run,REFERENCE,REFERENCE_SCORE,REFERENCE_ATOL

class UnbalanceTests(unittest.TestCase):
    setUp=fixture.DashboardV2Tests.setUp

    def prepare(self):
        folder=self.root/'data/20260919/20260919_1800RPM/Unbalance_-1/Vib';folder.mkdir(parents=True)
        path=folder/REFERENCE;savemat(path,{'signals':np.random.default_rng(4).normal(size=(6,65536)),'fs':25600})
        return folder,path

    def test_v3_exact_result_no_rf_db_and_no_fake_50(self):
        from vibration.dashboard_v3 import analyze_selected
        from vibration.dashboard_v2 import configuration
        folder,p=self.prepare();expected,_=analyze_selected(p.read_bytes(),1800,'uploaded.mat',configuration())
        with patch('sqlite3.connect',side_effect=AssertionError('no DB')),patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('no RF')):
            report=run(folder,self.bundles,self.root/'result','synthetic')
        self.assertEqual(report['reference_actual_score'],expected['models']['autoencoder']['file_score'])
        self.assertEqual(report['discovered_count'],1);self.assertFalse(report['expected_count_matches'])
        self.assertFalse(report['independent_test_performance_validated'])
        for name in ('ae_results_50.csv','summary.json','README.md'):self.assertTrue((self.root/'result'/name).is_file())
        with self.assertRaises(FileExistsError):run(folder,self.bundles,self.root/'result','synthetic')

    def test_errors_not_normal_and_baseline_value_is_comparison_only(self):
        folder,p=self.prepare();bad=folder/'20260919_132507_Vib.mat';bad.write_bytes(b'bad MAT')
        with patch('scripts.analyze_unbalance_preliminary.analyze_selected') as engine:
            engine.return_value=({'models':{'autoencoder':{'file_score':REFERENCE_SCORE+REFERENCE_ATOL/2,'file_threshold':.445470,'predicted_anomaly':0}},'bundle_id':'synthetic'}, {'autoencoder':'synthetic-version'})
            # Invalid MAT is made to fail by inference, not by condition filename.
            def infer(raw,*a,**k):
                if raw==b'bad MAT':raise ValueError('invalid MAT')
                return engine.return_value
            engine.side_effect=infer
            report=run(folder,self.bundles,self.root/'result','synthetic')
        self.assertEqual(report['success_count'],1);self.assertEqual(report['error_count'],1)
        self.assertEqual(report['normal_miss_candidate_count'],1)
        self.assertEqual(report['normal_fraction_total'],.5);self.assertEqual(report['normal_fraction_successful'],1)
        self.assertTrue(report['normal_fraction_total_is_lower_bound']);self.assertTrue(report['reference_score_matches'])
        self.assertFalse(report['completed_and_checked'])

    def test_all_50_cases_counted_and_success_denominator(self):
        from contextlib import redirect_stdout
        from io import StringIO
        folder=self.root/'data/20260919/20260919_1800RPM/Unbalance_-1/Vib';folder.mkdir(parents=True)
        for i in range(50):(folder/f'20260919_{132506+i}_Vib.mat').write_bytes(str(i).encode())
        def fixture_inference(raw,*a,**k):
            normal=int(raw)%2==0;score=REFERENCE_SCORE if normal else .5
            return ({'models':{'autoencoder':{'file_score':score,'file_threshold':.445470,'predicted_anomaly':int(not normal)}},'bundle_id':'synthetic'}, {'autoencoder':'synthetic-version'})
        with patch('scripts.analyze_unbalance_preliminary.analyze_selected',side_effect=fixture_inference) as infer,redirect_stdout(StringIO()):
            report=run(folder,self.bundles,self.root/'result','synthetic')
        self.assertEqual(infer.call_count,50);self.assertEqual(report['success_count'],50)
        self.assertEqual(report['normal_miss_candidate_count'],25)
        self.assertEqual(report['normal_fraction_total'],.5);self.assertEqual(report['anomaly_count'],25)
        self.assertTrue(report['reference_score_matches']);self.assertTrue(report['completed_and_checked'])
