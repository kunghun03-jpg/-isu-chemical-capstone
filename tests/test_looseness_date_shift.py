"""Synthetic full-file audit tests. No actual Looseness conclusions."""
import argparse
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from tests import test_dashboard_v2 as fixtures
from scripts.analyze_looseness_date_shift import discover,run,parity,comparisons,summaries
from vibration.dashboard_v2 import analyze,configuration
from vibration.candidate_analysis import candidate_analysis


class DateShiftTests(unittest.TestCase):
    setUp=fixtures.DashboardV2Tests.setUp

    def mat(self,date='20260912',rpm=900,condition='Looseness0.05',time='120000',seed=2):
        p=self.root/'data'/date/f'{date}_{rpm}RPM'/condition/'Vib'/f'{date}_{time}_Vib.mat'
        p.parent.mkdir(parents=True,exist_ok=True)
        savemat(p,{'signals':np.random.default_rng(seed).normal(size=(6,65536)),'fs':25600})
        return p

    def args(self):
        return argparse.Namespace(data_root=str(self.root/'data'),bundles=str(self.bundles),dates=['20260912','20260916'],include_normal=False,data_kind='synthetic',reference=[],inventory_only=False,resume=False,no_plots=True,output=str(self.root/'analysis'))

    def reference(self,p,date,rpm,condition):
        raw=p.read_bytes();result,_=analyze(raw,rpm,'uploaded.mat',configuration());features,_=candidate_analysis(raw,rpm);item=result['models']['autoencoder']
        row=dict(path=str(p),resolved_path=str(p),date=date,rpm=rpm,condition=condition,status='analyzed',input_sha256=result['input']['sha256'],ae_score=item['file_score'],ae_threshold=item['file_threshold'],ae_anomaly=item['predicted_anomaly'],features=json.dumps(features['features']))
        ref=self.root/'reference.csv'
        with ref.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
        return ref

    def test_discovery_counts_metadata_and_invalid_structure(self):
        self.mat();self.mat('20260916',1200,'LoosenessFull');self.mat(condition='Normal1')
        bad=self.mat(time='130000');bad.rename(bad.with_name('20260916_130000_Vib.mat'))
        rows,excluded=discover(self.root/'data',['20260912','20260916'])
        self.assertEqual(len(rows),3);self.assertEqual(len(excluded),1)
        self.assertEqual({r['rpm'] for r in rows},{900,1200})
        self.assertEqual({r['date'] for r in rows},{'20260912','20260916'})

    def test_v2_ae_features_sha_parity_and_no_rf_db(self):
        p=self.mat();args=self.args();args.reference=[str(self.reference(p,'20260912',900,'Looseness0.05'))]
        before=p.read_bytes()
        with patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('RF prohibited')),patch('sqlite3.connect',side_effect=AssertionError('DB prohibited')):
            summary=run(args)
        self.assertTrue(summary['parity_verified']);self.assertFalse(summary['conclusions_ready']);self.assertEqual(summary['reference_matched'],1)
        self.assertEqual(summary['statuses'],{'ok':1});self.assertEqual(p.read_bytes(),before)
        rows=list(csv.DictReader(StringIO((Path(args.output)/'file_results.csv').read_text(encoding='utf-8-sig'))))
        import hashlib
        self.assertEqual(rows[0]['sha256'],hashlib.sha256(before).hexdigest())
        self.assertIn('one_x_amplitude',rows[0])
        self.assertFalse(self.db.exists())

    def test_error_and_duplicate_files_do_not_stop_other_files(self):
        a=self.mat();b=self.mat(time='120001');b.write_bytes(a.read_bytes())
        bad=self.mat('20260916',time='120002');bad.write_bytes(b'invalid MAT')
        summary=run(self.args())
        self.assertEqual(summary['statuses'],{'ok':1,'duplicate_content':1,'error':1})
        groups=summary['groups'];self.assertEqual(sum(r['read_failed_files'] for r in groups),1)
        self.assertFalse(summary['conclusions_ready'])

    def test_resume_skips_processed_and_refuses_modified_inputs(self):
        p=self.mat();args=self.args();first=run(args)
        with self.assertRaises(FileExistsError):run(args)
        args.resume=True
        with patch('scripts.analyze_looseness_date_shift.process',side_effect=AssertionError('must not rerun')):
            second=run(args)
        self.assertEqual(second['processed'],1)
        p.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'MAT changed'):run(args)

    def test_interrupt_preserves_partial_results_and_resume(self):
        self.mat();self.mat('20260916',seed=4)
        args=self.args()
        import scripts.analyze_looseness_date_shift as module
        original=module.process;calls=[]
        def interrupt(*a,**k):
            calls.append(1)
            if len(calls)>1:raise KeyboardInterrupt()
            return original(*a,**k)
        with patch.object(module,'process',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):run(args)
        report=json.loads((Path(args.output)/'analysis_report.json').read_text())
        self.assertFalse(report['completed']);self.assertEqual(report['processed'],1)
        args.resume=True;report=run(args)
        self.assertTrue(report['completed']);self.assertEqual(report['processed'],2)

    def test_reference_mismatch_blocks_conclusions_and_normal_opt_in(self):
        p=self.mat();self.mat(condition='Normal1',seed=5);args=self.args();ref=self.reference(p,'20260912',900,'Looseness0.05')
        rows=list(csv.DictReader(StringIO(ref.read_text())));rows[0]['ae_score']='999'
        with ref.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        args.reference=[str(ref)];summary=run(args)
        self.assertEqual(summary['discovered_normal'],1);self.assertEqual(summary['selected'],1)
        self.assertEqual(len(summary['reference_mismatches']),1);self.assertFalse(summary['conclusions_ready'])

    def test_small_denominator_keeps_absolute_differences_not_infinite_ratio(self):
        a={'date':'20260912','rpm':900,'condition':'LoosenessFull','success_files':1,'ae_anomaly_rate':0}
        b={**a,'date':'20260916','ae_anomaly_rate':1}
        from scripts.analyze_looseness_date_shift import FEATURES
        for key in ('ae_score',*FEATURES):
            for stat in ('mean','median'):a[key+'_'+stat]=0;b[key+'_'+stat]=1
        row=next(r for r in comparisons([a,b],['20260912','20260916']) if r['rpm']==900 and r['condition']=='LoosenessFull')
        self.assertIsNone(row['rms_ac_median_ratio_b_over_a']);self.assertEqual(row['rms_ac_median_difference'],1)

    def test_inventory_all_18_groups_without_loading_mat_or_models(self):
        from scripts.analyze_looseness_date_shift import LOOSE,RPMS
        for date in ('20260912','20260916'):
            for rpm in RPMS:
                for condition in LOOSE:
                    p=self.root/'data'/date/f'{date}_{rpm}RPM'/condition/'Vib'/f'{date}_120000_Vib.mat'
                    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'not loaded in inventory')
        args=self.args();args.inventory_only=True
        with patch('scripts.analyze_looseness_date_shift.SavedModelInference',side_effect=AssertionError('inventory must not load models')):
            result=run(args)
        self.assertEqual(result['discovered'],18)
        self.assertFalse((Path(args.output)/'file_results.csv').exists())

    def test_normal_control_explicit_opt_in_and_separate_comparison(self):
        self.mat(seed=1);self.mat('20260916',seed=2)
        self.mat(condition='Normal1',seed=3);self.mat('20260916',condition='Normal1',seed=4)
        args=self.args();args.include_normal=True;report=run(args)
        self.assertEqual(report['processed'],4);self.assertEqual(report['discovered_normal'],2)
        self.assertGreater(report['normal_seconds_estimate'],0)
        control=list(csv.DictReader(StringIO((Path(args.output)/'normal_control_comparison.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual(len(control),1);self.assertEqual(control[0]['normal_control'],'Normal1')
        self.assertNotIn('accuracy',control[0])

    def test_plots_and_segment_summary_are_written_from_synthetic_mat(self):
        self.mat();args=self.args();args.no_plots=False;run(args)
        out=Path(args.output)
        self.assertTrue(list(out.glob('rms_ac_*.png')))
        segments=list(csv.DictReader(StringIO((out/'segment_summary.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual(len(segments),8);self.assertEqual(float(segments[-1]['end_seconds']),2.56)
