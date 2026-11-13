"""Synthetic calculation compatibility only, not real fault detection validation."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from scripts.verify_ae_v2_parity import TARGET, locate_mat, verify
from tests.test_web_env import synthetic_bundles
from vibration.dashboard_v2 import DashboardConfig
from vibration.inference_bundle import manifest_id


class ParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        bundles = self.root / 'bundles'
        synthetic_bundles(bundles)
        folder = bundles / '900rpm'
        manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(source_experiment='synthetic', experiment_mode='synthetic',
                        limitations=['Synthetic fixture'],
                        reference_file_metrics={name: {} for name in manifest['models']})
        manifest['bundle_id'] = manifest_id(manifest)
        manifest_path.write_text(json.dumps(manifest))
        self.source = self.root / 'source'
        self.source.mkdir()
        shutil.copyfile(folder / 'autoencoder.pt', self.source / 'autoencoder_900rpm.pt')
        shutil.copyfile(folder / 'normalization.npz', self.source / 'normalization_900rpm.npz')
        self.csv = self.source / 'evaluation_file_900rpm.csv'
        self.csv.write_text('file,threshold\nother.mat,0\n')
        self.mat = self.root / 'data/20260912_900RPM/IR/Vib' / TARGET
        self.mat.parent.mkdir(parents=True)
        self.signals = np.random.default_rng(42).normal(size=(6, 65536))
        savemat(self.mat, {'signals': self.signals, 'fs': 25600.})
        self.config = DashboardConfig('autoencoder', bundles)

    def run_verify(self, **kwargs):
        return verify(self.mat, 900, self.source, self.config, **kwargs)

    def test_actual_v2_path_matches_independent_equations_without_db(self):
        trace = sys.gettrace()
        with patch('sqlite3.connect', side_effect=AssertionError('No DB access')):
            report = self.run_verify(output=self.root / 'report')
        self.assertIs(sys.gettrace(), trace)
        self.assertEqual(report['status'], 'passed_calculation_parity')
        self.assertTrue(report['inputs_unchanged'])
        self.assertTrue(report['weights_equal'])
        self.assertTrue(all(r['passed'] for r in report['stage_comparisons']))
        self.assertEqual(report['archived_evaluation_rows_with_same_basename'], [])
        self.assertFalse(report['performance_validated'])
        self.assertTrue((self.root / 'report/report.json').is_file())
        with self.assertRaises(FileExistsError):
            self.run_verify(output=self.root / 'report')

    def test_changed_original_statistics_identify_first_stage(self):
        np.savez(self.source / 'normalization_900rpm.npz',
                 mean=np.ones((1, 1, 3201)), std=np.ones((1, 1, 3201)))
        report = self.run_verify()
        self.assertEqual(report['status'], 'mismatch_requires_review')
        self.assertEqual(report['first_mismatch_stage'], 'train_mean')
        self.assertIn('normalized', report['mismatch_stages'])

    def test_threshold_difference_is_not_hidden(self):
        self.csv.write_text('file,threshold\nother.mat,100\n')
        report = self.run_verify()
        self.assertIn('stored_file_threshold', report['mismatch_stages'])
        self.assertFalse(report['decision_equal'])

    def test_nonfinite_mat_rejected_without_prediction(self):
        self.signals[2, 0] = np.nan
        savemat(self.mat, {'signals': self.signals, 'fs': 25600.})
        with self.assertRaises(ValueError):
            self.run_verify()

    def test_web_failure_restores_trace(self):
        trace = sys.gettrace()
        with patch('vibration.inference.SavedModelInference.predict_bytes', side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):
                self.run_verify()
        self.assertIs(sys.gettrace(), trace)

    def test_missing_ambiguous_and_wrong_rpm_rejected(self):
        with self.assertRaises(FileNotFoundError):
            locate_mat(self.root / 'missing.mat')
        self.assertEqual(locate_mat(roots=[self.root]), self.mat)
        shutil.copyfile(self.mat, self.root / TARGET)
        with self.assertRaises(ValueError):
            locate_mat(roots=[self.root])
        with self.assertRaises(ValueError):
            verify(self.mat, 1200, self.source, self.config)


if __name__ == '__main__':
    unittest.main()
