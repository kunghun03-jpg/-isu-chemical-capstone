"""Synthetic integration; RF fitting is prohibited."""
import argparse
import csv
import json
from io import StringIO
import os
from pathlib import Path
from unittest.mock import patch
import unittest
from tests import test_dashboard_v2 as fixtures
from scripts.validate_v2_integration import run, compare, load_cases
from vibration.candidate_analysis import UNCLEAR


class IntegrationTests(unittest.TestCase):
    setUp=fixtures.DashboardV2Tests.setUp

    def args(self):
        csv_path=self.root/'test_predictions.csv'
        with csv_path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['path','rpm','raw_prediction','condition','date']);writer.writeheader()
            writer.writerow(dict(path=str(self.path),rpm=900,raw_prediction='Bearing',condition='OR',date='20260919'))
        return argparse.Namespace(test_predictions=str(csv_path),data_root=str(self.root),ae_bundles=str(self.bundles),
            rf_bundle=str(self.root/'missing_rf'),output=str(self.root/'report'),data_kind='synthetic')

    def fake(self):
        return dict(kind='RandomForest',model_version='synthetic-existing-output',model_sha256='test',feature_schema='test',
                    raw_prediction='Bearing',accepted=False,reasons=['자동 후보명 표시 비활성화'],labels_enabled=False,
                    scores_uncalibrated={'Bearing':.6,'Normal':.2,'Looseness':.1,'Unbalance':.1})

    def test_comparison_semantics(self):
        self.assertEqual(compare(False,None,'Unbalance'),'skipped_ae_normal')
        self.assertEqual(compare(True,None,'Unbalance'),'rf_unavailable')
        self.assertEqual(compare(True,'Bearing','Unbalance'),'mismatch')
        self.assertEqual(compare(True,'Unbalance','Unbalance'),'match')

    def test_same_web_path_abstention_save_restart_and_no_filename_features(self):
        args=self.args(); original=self.path.read_bytes()
        with patch.dict(os.environ,{'ISU_CHEMICAL_CANDIDATE_LABELS':'0'}), \
             patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('no training')), \
             patch('vibration.candidate_analysis.CandidateEngine') as engine:
            engine.return_value.predict.return_value=self.fake()
            summary=run(args)
            self.assertEqual(summary['comparison_counts'],{'match':1})
            self.assertTrue(summary['passed'])
            self.assertEqual(set(engine.return_value.predict.call_args.args[0]),set(__import__('vibration.candidate_engine',fromlist=['FEATURES']).FEATURES))
            self.assertEqual(engine.return_value.predict.call_args.args[1],900)
        rows=list(csv.DictReader(StringIO((Path(args.output)/'integration_results.csv').read_text(encoding='utf-8-sig'))))
        self.assertEqual(rows[0]['final_candidate'],UNCLEAR)
        self.assertEqual(rows[0]['db_status'],'passed')
        self.assertTrue(json.loads(rows[0]['db_verification'])['restarted'])
        self.assertEqual(self.path.read_bytes(),original)
        self.assertFalse(self.db.exists())
        with self.assertRaises(FileExistsError):run(args)

    def test_missing_model_retains_ae_and_saves_abstention(self):
        args=self.args(); summary=run(args)
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['comparison_counts'],{'rf_unavailable':1})
        rows=list(csv.DictReader(StringIO((Path(args.output)/'integration_results.csv').read_text(encoding='utf-8-sig'))))
        self.assertNotEqual(rows[0]['ae_score'],'')
        self.assertEqual(rows[0]['db_status'],'passed')
        self.assertIn('오류',rows[0]['reasons'])

    def test_missing_mat_reports_analysis_error_without_guess(self):
        args=self.args();self.path.unlink(); summary=run(args)
        self.assertEqual(summary['comparison_counts'],{'analysis_error':1})
        self.assertFalse(summary['passed'])

    def test_normal_skips_rf_not_mismatch(self):
        args=self.args()
        from vibration.dashboard_v2 import analyze
        def normal(*a,**k):
            result,versions=analyze(*a,**k)
            item=result['models']['autoencoder'];item.update(file_threshold=item['file_score']+1,predicted_anomaly=0)
            return result,versions
        with patch('scripts.validate_v2_integration.analyze',side_effect=normal),patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('must skip')):
            summary=run(args)
        self.assertEqual(summary['comparison_counts'],{'skipped_ae_normal':1})
        self.assertTrue(summary['passed'])

    def test_csv_contract_rejects_duplicates(self):
        args=self.args()
        p=Path(args.test_predictions);s=p.read_text();p.write_text(s+s.splitlines()[1]+'\n')
        with self.assertRaisesRegex(ValueError,'중복'):load_cases(p,self.root)

    @unittest.skipUnless(os.environ.get('ISU_TEST_CANDIDATE_BUNDLE'), 'existing synthetic RF artifact not supplied')
    def test_existing_artifact_offline_direct_and_web_path_match_without_fit(self):
        from vibration.candidate_engine import CandidateEngine
        from vibration.candidate_analysis import candidate_analysis
        args=self.args();args.rf_bundle=os.environ['ISU_TEST_CANDIDATE_BUNDLE']
        report,_=candidate_analysis(self.raw,900)
        with patch('sklearn.ensemble.RandomForestClassifier.fit',side_effect=AssertionError('no training')):
            offline=CandidateEngine(args.rf_bundle).predict(report['features'],900,labels_enabled=False)['raw_prediction']
            p=Path(args.test_predictions)
            rows=list(csv.DictReader(StringIO(p.read_text())))
            rows[0]['raw_prediction']=offline
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            summary=run(args)
        self.assertEqual(summary['comparison_counts'],{'match':1})
        self.assertTrue(summary['passed'])

    def test_priority_selection_changes_order_not_reference_prediction(self):
        from scripts.validate_v2_integration import PRIORITY
        args=self.args();p=Path(args.test_predictions)
        with p.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','rpm','raw_prediction','condition','date'])
            w.writerow(dict(path=PRIORITY,rpm=1800,raw_prediction='Normal',condition='Unbalance_-1',date='20260919'))
        rows=load_cases(p,self.root)
        self.assertEqual(rows[0]['path'],PRIORITY)
        self.assertEqual(rows[0]['raw_prediction'],'Normal')

    def test_repository_output_rejected_before_writing(self):
        from scripts.validate_v2_integration import ROOT
        args=self.args();args.output=str(ROOT/'integration-do-not-create')
        with self.assertRaisesRegex(ValueError,'저장소 밖'):run(args)
        self.assertFalse(Path(args.output).exists())
