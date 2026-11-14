"""Synthetic multi-RPM condition evaluation; no actual RPM support inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests import test_dae_ui as fixture
from tests.test_dae import mat_bytes
from scripts.evaluate_dae_rpm_conditions import evaluate,aggregate,interval
from vibration.dae_inference import SavedDAEBundleInference,SavedDAEInference
from vibration.dae_bundle import sha256_file

class DAEEvaluationTests(unittest.TestCase):
    setUp=fixture.DAEUITests.setUp
    convert=fixture.DAEUITests.convert

    def case(self,root,rpm=900,date='20260912',condition='Normal1',time='120000',seed=1):
        p=Path(root)/date/f'{date}_{rpm}RPM'/condition/'Vib'/f'{date}_{time}_Vib.mat';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(mat_bytes(np.random.default_rng(seed).normal(size=(6,65536))))
        return p

    def test_all_rpm_date_condition_paths_calculate_and_public_still_blocked(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';i=0
            for rpm in (900,1200,1800):
                for date in ('20260912','20260916'):
                    for condition in ('Normal1','Normal2','IR','OR','Roller','Looseness0.05','Looseness0.1','LoosenessFull','Unbalance_-1','Unbalance_-2'):
                        i+=1;self.case(root,rpm,date,condition,seed=i)
            before={p.name:sha256_file(p) for p in self.bundle.iterdir() if p.is_file()}
            report=evaluate(root,self.bundle,Path(temp)/'out',count=2,data_kind='synthetic',no_plots=True)
            self.assertEqual(report['result_statuses'],{'ok':60})
            self.assertTrue(all(p['calculation_succeeded_on_selected_unique_files'] for p in report['activation_plan']))
            self.assertTrue(all(p['activation_status']=='blocked_pending_independent_review' for p in report['activation_plan']))
            self.assertIsNone(report['declared_supported_rpms']);self.assertFalse(report['independent_test_performance_validated'])
            self.assertEqual(before,{p.name:sha256_file(p) for p in self.bundle.iterdir() if p.is_file()})
            for name in ('dae_results.csv','dae_rpm_summary.csv','dae_condition_summary.csv','dae_validation_report.json','README.md'):self.assertTrue((Path(temp)/'out'/name).is_file())
            raw=next(root.rglob('*.mat')).read_bytes()
            engine=SavedDAEBundleInference(self.bundle)
            with self.assertRaises(ValueError):engine.predict_bytes(raw,rpm=900)
            with self.assertRaises(FileExistsError):evaluate(root,self.bundle,Path(temp)/'out',data_kind='synthetic',no_plots=True)

    def test_fixed_kernel_mse_matches_without_any_public_predict_call(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';path=self.case(root)
            expected=SavedDAEInference.predict_bytes(SavedDAEBundleInference(self.bundle),path.read_bytes())
            with patch('vibration.dae_inference.SavedDAEBundleInference.predict_bytes',side_effect=AssertionError('public method must not be used for offline RPM-unconfirmed evaluation')),patch('sqlite3.connect',side_effect=AssertionError('DB forbidden')):
                report=evaluate(root,self.bundle,Path(temp)/'out',data_kind='synthetic',no_plots=True)
            self.assertEqual(report['rpm_summary'][0]['normal_mse_median'],expected['mse'])
            self.assertEqual(report['fixed_threshold'],expected['threshold'])

    def test_errors_and_duplicates_excluded_from_denominators(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';p=self.case(root)
            q=self.case(root,time='120001',seed=2);q.write_bytes(p.read_bytes())
            bad=self.case(root,condition='IR',seed=3);bad.write_bytes(b'invalid MAT')
            report=evaluate(root,self.bundle,Path(temp)/'out',all_files=True,data_kind='synthetic',no_plots=True)
            self.assertEqual(report['result_statuses'],{'error':1,'ok':1,'duplicate_content':1})
            summary=report['rpm_summary'][0];self.assertEqual(summary['normal_n'],1);self.assertEqual(summary['fault_n'],0)
            self.assertIsNone(summary['fault_alarm_fraction'])

    def test_runtime_failure_does_not_fallback_to_unchecked_loader(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';self.case(root)
            with patch('scripts.evaluate_dae_rpm_conditions.SavedDAEBundleInference',side_effect=ValueError('runtime mismatch')):
                report=evaluate(root,self.bundle,Path(temp)/'out',data_kind='synthetic',no_plots=True)
            self.assertEqual(report['status'],'blocked_bundle_validation');self.assertEqual(report['result_statuses'],{'error':1})
            self.assertTrue(all(not p['calculation_succeeded_on_selected_unique_files'] for p in report['activation_plan']))

    def test_metrics_directions_and_no_support_decision_from_good_rates(self):
        rows=[]
        for condition,family,mse,alarm in [('Normal1','Normal',.1,False),('IR','Bearing',2.,True)]:
            rows.append(dict(rpm=900,date='20260912',condition=condition,family=family,status='ok',mse=mse,threshold_alarm=alarm))
        rpms,conditions=aggregate(rows)
        self.assertEqual(rpms[0]['normal_alarm_fraction'],0);self.assertEqual(rpms[0]['fault_alarm_fraction'],1)
        self.assertEqual(rpms[0]['observed_auroc'],1);self.assertIsNone(rpms[1]['normal_alarm_fraction'])
        self.assertEqual(interval(0,0),(None,None));self.assertGreater(interval(0,10)[1],0)

    def test_sampling_is_deterministic_and_not_score_selected(self):
        from scripts.analyze_fault_signatures import discover
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data'
            for i in range(5):self.case(root,time=f'12000{i}',seed=i)
            a=discover([root],count=3);b=discover([root],count=3)
            self.assertEqual(a,b)
            selected=[Path(r['path']).name for r in a if r['status']=='selected']
            self.assertEqual(selected,['20260912_120000_Vib.mat','20260912_120002_Vib.mat','20260912_120004_Vib.mat'])

    def test_nonfinite_mat_never_gets_a_normal_or_abnormal_label(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';p=self.case(root);x=np.zeros((6,65536));x[2,10]=np.nan;p.write_bytes(mat_bytes(x))
            report=evaluate(root,self.bundle,Path(temp)/'out',data_kind='synthetic',no_plots=True)
            self.assertEqual(report['result_statuses'],{'error':1})
            import csv
            with (Path(temp)/'out/dae_results.csv').open(encoding='utf-8-sig') as f:r=next(csv.DictReader(f))
            self.assertEqual(r['offline_prediction'],'');self.assertEqual(r['mse'],'')

    def test_mse_distribution_plot_written_from_synthetic_mat(self):
        self.convert()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';self.case(root)
            report=evaluate(root,self.bundle,Path(temp)/'out',data_kind='synthetic',no_plots=False)
            self.assertTrue((Path(temp)/'out/dae_mse_900.png').is_file())
            self.assertTrue(report['public_rpm_policy_unchanged'])
