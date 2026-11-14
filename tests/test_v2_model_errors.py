"""Synthetic observations only; no real MAT or performance improvement claim."""
import argparse
import csv
from io import StringIO
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.analyze_v2_model_errors import load,summarize,confusion,run
from scripts.discover_looseness_cases import discover
from scripts.validate_v2_integration import load_cases,compare


class ModelErrorsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def csv(self,rows,name='integration.csv'):
        p=self.root/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
        return p

    def row(self,condition='Unbalance_-1',rpm=1800,date='20260919',score=.375726,threshold=.445470,raw=''):
        return dict(path=f'/data/{date}_{rpm}_{condition}.mat',condition=condition,rpm=rpm,date=date,status='analyzed',data_kind='synthetic',
            ae_score=score,ae_threshold=threshold,ae_anomaly=score>threshold,rf_executed=bool(raw),rf_raw_prediction=raw,
            final_candidate='결함 후보 불명확',abstained=bool(raw),reasons='[]',features=json.dumps({'rms_ac':1,'one_x_amplitude':2,'two_x_amplitude':.1}))

    def test_miss_and_false_positive_denominators_and_raw_confusion(self):
        p=self.csv([self.row(),self.row(rpm=900,score=1,raw='Looseness'),self.row('Normal1',score=1,raw='Bearing')])
        rows,bad=load(p,'integration');self.assertFalse(bad)
        groups=summarize(rows)
        self.assertEqual(next(r for r in groups if r['rpm']==1800 and r['condition']=='Unbalance_-1')['missed_fault_rate'],1)
        self.assertEqual(next(r for r in groups if r['condition']=='Normal1')['false_positive_rate'],1)
        cm=confusion(rows);self.assertTrue(any(r['truth']=='Unbalance' and r['raw_prediction']=='Looseness' for r in cm))
        self.assertEqual(sum(r['count'] for r in cm),2)

    def test_invalid_nan_gate_and_unknown_conditions_are_excluded(self):
        rows=[self.row(score=float('nan')),self.row(condition='unknown'),self.row(raw='Bearing')]
        good,bad=load(self.csv(rows),'integration');self.assertFalse(good);self.assertEqual(len(bad),3)

    def test_sources_dates_and_rpm_remain_separate(self):
        a,_=load(self.csv([self.row(date='20260912'),self.row(date='20260919')]),'integration')
        b,_=load(self.csv([self.row()],name='other.csv'),'integration')
        self.assertEqual(len(summarize(a+b)),3)

    def test_offline_raw_does_not_invent_final_label(self):
        p=self.csv([dict(path='/file.mat',rpm=900,date='20260919',condition='OR',raw_prediction='Bearing',accepted='False',reasons='policy')])
        rows,bad=load(p,'offline');self.assertFalse(bad)
        self.assertEqual(rows[0]['final_candidate'],'');self.assertTrue(rows[0]['abstained'])
        self.assertNotIn('ae_score',rows[0])

    def test_looseness_discovery_strict_metadata_and_no_predictions(self):
        valid=self.root/'20260919/20260919_1200RPM/Looseness0.05/Vib/20260919_120000_Vib.mat'
        valid.parent.mkdir(parents=True);valid.write_bytes(b'not read by discovery')
        invalid=valid.with_name('20260912_120000_Vib.mat');invalid.write_bytes(b'')
        rows,bad=discover([self.root]);self.assertEqual(len(rows),1);self.assertEqual(len(bad),1)
        self.assertEqual(rows[0]['raw_prediction'],'')
        p=self.csv(rows);self.assertEqual(load_cases(p,self.root,False)[0]['condition'],'Looseness0.05')
        with self.assertRaises(ValueError):load_cases(p,self.root)
        self.assertEqual(compare(True,'Looseness',''),'no_offline_reference')
        self.assertEqual(compare(False,None,''),'skipped_ae_normal')

    def test_csv_report_generates_outputs_without_reading_mat(self):
        p=self.csv([self.row(),self.row('Normal2',score=.1)])
        original=p.read_bytes();out=self.root/'report'
        report=run([p],[],out)
        for name in ('condition_coverage.csv','ae_error_analysis.csv','rf_error_analysis.csv','rpm_condition_summary.csv','MODEL_ERROR_ANALYSIS.md'):
            self.assertTrue((out/name).is_file())
        self.assertEqual(p.read_bytes(),original);self.assertEqual(report['changed_inputs'],[])
        self.assertTrue(list(out.glob('*.png')))
        with self.assertRaises(FileExistsError):run([p],[],out)
        coverage=list(csv.DictReader(StringIO((out/'condition_coverage.csv').read_text(encoding='utf-8-sig'))))
        self.assertTrue(any(r['condition']=='LoosenessFull' and r['files']=='0' for r in coverage))


class LoosenessIntegrationTests(unittest.TestCase):
    from tests import test_dashboard_v2 as fixtures
    setUp=fixtures.DashboardV2Tests.setUp

    def test_standalone_additional_cases_run_without_fake_reference(self):
        from scripts.validate_v2_integration import run as validate
        from tests import test_v2_integration as existing
        args=existing.IntegrationTests.args(self)
        args.additional_cases=args.test_predictions;args.test_predictions=None
        path=Path(args.additional_cases)
        rows=list(csv.DictReader(StringIO(path.read_text())))
        rows[0].update(raw_prediction='',condition='LoosenessFull')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        with patch('vibration.candidate_analysis.CandidateEngine') as engine:
            engine.return_value.predict.return_value=existing.IntegrationTests.fake(self)
            result=validate(args)
        self.assertEqual(result['comparison_counts'],{'no_offline_reference':1})
        self.assertIsNone(result['reference_csv'])
        self.assertEqual(result['conditions'],{'LoosenessFull':1})
