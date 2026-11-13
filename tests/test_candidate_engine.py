"""Synthetic artifact training/inference and gating, not real fault performance validation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.analyze_fault_signatures import write_csv
from scripts.train_candidate_engine import run, split_rows
from tests.test_fault_classifier import fixture
from tests import test_dashboard_v2 as ui_fixtures
from vibration.candidate_analysis import analyze_with_engine, candidate_analysis, SKIPPED, UNCLEAR, STAGE_VERSION
from vibration.candidate_engine import CandidateEngine, FEATURES, feature_matrix, decide
from vibration.candidate_history import read_candidate
from vibration.dashboard_v2 import analyze, configuration
from vibration.history import HistoryStore


def clean_fixture(folder):
    rows, report = fixture(folder)
    for row in rows:
        v = np.ones(9)
        if row['family'] == 'Bearing':
            v[[7, 8]] = 11
        elif row['family'] == 'Looseness':
            v[[2, 4]] = 11
        elif row['family'] == 'Unbalance':
            v[3] = 11
        row.update(dict(zip(FEATURES, v.tolist())))
    write_csv(folder / 'file_features.csv', rows)
    return rows, report


class EngineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.eda, self.bundle = self.root / 'eda', self.root / 'engine'
        self.rows, self.eda_report = clean_fixture(self.eda)
        self.report = run(self.eda, self.bundle, ['20260912'], ['20260916'], ['20260919'])
        self.engine = CandidateEngine(self.bundle)

    def features(self, label='Bearing'):
        row = next(r for r in self.rows if r['family'] == label)
        return {f: row[f] for f in FEATURES}

    def test_features_order_and_imputation_round_trip(self):
        features = self.features()
        a = self.engine.predict(features, 900, True)
        b = self.engine.predict(dict(reversed(list(features.items()))), 900, True)
        self.assertEqual(a, b)
        self.assertEqual(a['raw_prediction'], 'Bearing')
        self.assertTrue(a['accepted'])
        with self.assertRaises(ValueError):
            feature_matrix([{**features, 'path': 'IR.mat'}])
        features['envelope_rms_ac'] = None
        x = feature_matrix([features])
        imputed = self.engine.pipeline.named_steps['imputer'].transform(x)
        self.assertEqual(imputed[0, 8], self.report['training_imputer_statistics'][8])
        response = self.engine.predict(features, 900, True)
        self.assertFalse(response['accepted'])
        self.assertTrue(any('결측' in r for r in response['reasons']))

    def test_missing_corrupt_version_schema_and_manifest_errors(self):
        with self.assertRaises(FileNotFoundError):
            CandidateEngine(self.root / 'absent')
        manifest_path = self.bundle / 'manifest.json'
        original = manifest_path.read_text()
        for field in ('versions', 'features'):
            m = json.loads(original)
            if field == 'versions':
                m[field]['numpy'] = 'wrong'
            else:
                m[field] = list(reversed(m[field]))
            manifest_path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                CandidateEngine(self.bundle)
        manifest_path.write_text(original)
        with (self.bundle / 'model.joblib').open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaises(ValueError):
            CandidateEngine(self.bundle)

    def test_default_off_normal_conflict_sparse_policy_and_low_margin(self):
        off = self.engine.predict(self.features(), 900)
        self.assertFalse(off['accepted'])
        self.assertFalse(off['labels_enabled'])
        normal = self.engine.predict(self.features('Normal'), 900, True)
        self.assertTrue(any('판정 불일치' in r for r in normal['reasons']))
        # Only four validation Unbalance samples per RPM: policy must remain unavailable.
        sparse = self.engine.predict(self.features('Unbalance'), 900, True)
        self.assertFalse(sparse['accepted'])
        self.assertTrue(any('표본' in r for r in sparse['reasons']))
        accepted, reasons, _ = decide('Bearing', 0., feature_matrix([self.features()])[0], 900,
                                      self.engine.manifest['training_bounds'], self.engine.manifest['policy'], [900, 1200], True)
        self.assertFalse(accepted)
        self.assertTrue(any('점수 차이' in r for r in reasons))
        outside = self.features()
        outside['rms_ac'] = 1e4
        self.assertFalse(self.engine.predict(outside, 900, True)['accepted'])
        self.assertFalse(self.engine.predict(self.features(), 1800, True)['accepted'])

    def test_dates_disjoint_and_test_changes_do_not_fit_model_or_policy(self):
        self.assertTrue(self.report['inputs_unchanged'])
        from scripts.evaluate_fault_classifier import load_eda
        rows, _, _ = load_eda(self.eda)
        with self.assertRaises(ValueError):
            split_rows(rows, ['20260912'], ['20260912'], ['20260919'])
        for row in self.rows:
            if row['date'] == '20260919':
                row.update({f: 1000. for f in FEATURES})
        write_csv(self.eda / 'file_features.csv', self.rows)
        report2 = run(self.eda, self.root / 'other', ['20260912'], ['20260916'], ['20260919'])
        engine2 = CandidateEngine(self.root / 'other')
        self.assertEqual(self.engine.manifest['policy'], engine2.manifest['policy'])
        self.assertEqual(self.engine.manifest['training_bounds'], engine2.manifest['training_bounds'])
        self.assertEqual(self.report['training_imputer_statistics'], report2['training_imputer_statistics'])
        np.testing.assert_array_equal(self.engine.pipeline.predict_proba(feature_matrix([self.features()])),
                                      engine2.pipeline.predict_proba(feature_matrix([self.features()])))

    def test_report_selective_metrics_and_no_overwrite(self):
        summary = self.report['test_metrics'][0]
        self.assertGreater(summary['abstention_rate'], 0)
        self.assertEqual(summary['selective_accuracy'], 1.)  # Deliberately separable synthetic vectors only.
        self.assertTrue((self.bundle / 'split_files.csv').is_file())
        with self.assertRaises(FileExistsError):
            run(self.eda, self.bundle, ['20260912'], ['20260916'], ['20260919'])


class StageTests(unittest.TestCase):
    setUp = ui_fixtures.DashboardV2Tests.setUp
    app = ui_fixtures.DashboardV2Tests.app
    upload = ui_fixtures.DashboardV2Tests.upload

    def ae(self):
        return analyze(self.raw, 900, 'measurement.mat', configuration())

    def test_normal_skips_engine_abnormal_calls_engine_and_conflict(self):
        result, _ = self.ae()
        normal = deepcopy(result)
        normal['models']['autoencoder'].update(file_threshold=1e10, predicted_anomaly=0)
        with patch('vibration.candidate_analysis.CandidateEngine', side_effect=AssertionError('must skip')):
            report, _ = analyze_with_engine(self.raw, 900, normal)
        self.assertEqual(report['candidate'], SKIPPED)
        self.assertEqual(report['status'], 'skipped_normal')
        fake = {'kind': 'RandomForest', 'model_version': 'synthetic', 'raw_prediction': 'Bearing',
                'accepted': True, 'labels_enabled': True, 'reasons': [], 'policy_rule': {'margin_min': .5},
                'score_margin': .9, 'scores_uncalibrated': {'Bearing': .95, 'Normal': .05}}
        with patch.dict(os.environ, {'ISU_CHEMICAL_CANDIDATE_BUNDLE': str(self.root / 'mock'), 'ISU_CHEMICAL_CANDIDATE_LABELS': '1'}), \
                patch('vibration.candidate_analysis.CandidateEngine') as engine:
            engine.return_value.predict.return_value = fake
            before = deepcopy(result)
            report, _ = analyze_with_engine(self.raw, 900, result)
            engine.return_value.predict.assert_called_once()
            self.assertEqual(report['candidate'], '베어링 결함 의심')
            self.assertEqual(result, before)
        store = HistoryStore(self.db)
        store.save(result, self.ae()[1], candidate_report=report)
        self.assertEqual(read_candidate(store, 1), report)

    def test_missing_model_is_visible_and_old_report_remains_readable(self):
        result, versions = self.ae()
        legacy, _ = candidate_analysis(self.raw, 900)
        store = HistoryStore(self.db)
        store.save(result, versions, candidate_report=legacy)
        self.assertEqual(read_candidate(store, 1), legacy)
        with patch.dict(os.environ, {'ISU_CHEMICAL_CANDIDATE_BUNDLE': str(self.root / 'absent')}):
            report, _ = analyze_with_engine(self.raw, 900, result)
        self.assertEqual(report['candidate'], UNCLEAR)
        self.assertTrue(any('FileNotFoundError' in r for r in report['reasons']))
        store.save(result, versions, candidate_report=report)
        self.assertEqual(read_candidate(store, 1)['analyzer_version'], STAGE_VERSION)
        self.assertEqual(store.detail(1)['models'][0]['score'], result['models']['autoencoder']['file_score'])

    def test_normal_web_skips_classifier_and_persists_skip(self):
        from vibration.inference_bundle import manifest_id
        manifest_path = self.bundles / '900rpm/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['models']['autoencoder']['file_threshold'] = 1e10  # Temporary synthetic bundle only.
        manifest['bundle_id'] = manifest_id(manifest)
        manifest_path.write_text(json.dumps(manifest))
        with self.upload(), patch('vibration.candidate_analysis.CandidateEngine', side_effect=AssertionError('must skip')):
            app = self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_candidate']['candidate'], SKIPPED)
            app.button(key='v2_save').click().run()
            self.assertFalse(app.exception)
        self.assertEqual(read_candidate(HistoryStore(self.db), 1)['status'], 'skipped_normal')

    def test_web_ae_unchanged_and_engine_failure_does_not_drop_result(self):
        expected, _ = self.ae()
        with self.upload(), patch.dict(os.environ, {'ISU_CHEMICAL_CANDIDATE_BUNDLE': str(self.root / 'absent')}):
            app = self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result']['models'], expected['models'])
            self.assertEqual(app.session_state['v2_candidate']['status'], 'abstained')
            self.assertTrue(any('FileNotFoundError' in i.value for i in app.info))
            app.button(key='v2_save').click().run()
            self.assertFalse(app.exception)

    def test_real_synthetic_mat_to_saved_pipeline_not_a_mock_classifier(self):
        result, versions = self.ae()
        observation, _ = candidate_analysis(self.raw, 900)
        folder = self.root / 'candidate-training'
        csv_rows, _ = fixture(folder)
        scales = {'Normal': .5, 'Bearing': 1., 'Looseness': 1.5, 'Unbalance': 2.}
        for row in csv_rows:
            row.update({f: observation['features'][f] * scales[row['family']] for f in FEATURES})
        write_csv(folder / 'file_features.csv', csv_rows)
        bundle = self.root / 'real-pipeline'
        run(folder, bundle, ['20260912'], ['20260916'], ['20260919'])
        with patch.dict(os.environ, {'ISU_CHEMICAL_CANDIDATE_BUNDLE': str(bundle), 'ISU_CHEMICAL_CANDIDATE_LABELS': '1'}):
            report, _ = analyze_with_engine(self.raw, 900, result)
        self.assertEqual(report['engine']['raw_prediction'], 'Bearing')
        self.assertEqual(report['candidate'], '베어링 결함 의심')
        self.assertIsNone(report['confidence'])
        store = HistoryStore(self.db)
        store.save(result, versions, candidate_report=report)
        self.assertEqual(read_candidate(store, 1), report)


if __name__ == '__main__':
    unittest.main()
