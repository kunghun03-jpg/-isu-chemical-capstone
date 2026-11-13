"""Synthetic date-shift and harmonic experiment checks, not real performance validation."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from scripts.analyze_fault_signatures import analyze_signal, write_csv
from scripts.analyze_feature_date_shift import analyze, contrast, run as shift_run
from scripts.compare_fault_feature_sets import load_extension, paired_deltas, run as compare_run
from scripts.evaluate_fault_classifier import FEATURES, RF_CONFIG, evaluate_rows, load_eda
from scripts.extract_harmonic_features import EXTRA_FEATURES, RATIO_POLICY, extended_features, run as extract_run
from tests.test_fault_classifier import fixture


def fake_extension(folder, rows, source):
    """Synthetic manifest fixture; no claim that these were computed from actual MATs."""
    folder.mkdir()
    records, csv_rows = [], []
    for row in rows:
        values = {f: None if np.isnan(v) else v for f, v in zip(FEATURES, row['values'])}
        values.update({f: float(j + 1) for j, f in enumerate(EXTRA_FEATURES)})
        records.append({'path': row['path'], 'status': 'extracted', 'mat_sha256_actual': row['mat_sha256_recorded'], 'values': values})
        csv_rows.append({**{k: row[k] for k in ('path', 'rpm', 'date', 'condition', 'label')},
                         'mat_sha256': row['mat_sha256_recorded'], **values})
    csv_path = folder / 'extended_file_features.csv'
    write_csv(csv_path, csv_rows)
    report = {'completed': True, 'inputs_unchanged': True, 'status': 'harmonic_extraction_complete',
              'eda_input_sha256': source['input_sha256'], 'baseline_features': list(FEATURES),
              'extra_features': list(EXTRA_FEATURES), 'ratio_policy': RATIO_POLICY, 'files': records,
              'extended_csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    (folder / 'harmonic_extraction.json').write_text(json.dumps(report))


class ExperimentsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.eda = self.root / 'eda'
        self.csv, self.manifest = fixture(self.eda, dates=('20260912', '20260919'), rpms=(900, 1200))

    def test_overlap_identical_separated_and_missing(self):
        self.assertEqual(contrast([1, 1], [1, 1])['histogram_overlap_10_bins'], 1)
        separate = contrast([0, 0], [10, 10])
        self.assertEqual(separate['histogram_overlap_10_bins'], 0)
        self.assertEqual(separate['cliffs_delta_b_minus_a'], 1)
        self.assertEqual(separate['wasserstein_raw'], 10)
        self.assertEqual(contrast([np.nan], [1])['status'], 'missing_observations')

    def test_date_change_larger_than_fault_contrast(self):
        rows, _, _ = load_eda(self.eda)
        for row in rows:
            row['values'] = [float(row['date'] == '20260919') * 100 + (1 if row['label'] == 'Looseness' else 0)] * len(FEATURES)
        overlap, shifts, sensitivity = analyze(rows)
        self.assertTrue(any(r['normal'] == 'Normal2' for r in overlap))
        self.assertTrue(any(r['condition'] == 'Normal1' for r in shifts))
        selected = next(r for r in sensitivity if r['looseness'] == 'Looseness0.1')
        self.assertTrue(selected['date_distance_larger'])
        self.assertGreater(selected['date_to_fault_distance_ratio'], 10)

    def test_harmonic_amplitudes_ratios_and_energy_fractions(self):
        fs, n = 25600., 65536
        t = np.arange(n) / fs
        # At 1200RPM these are the nearest FFT bins to orders 1..5, all inside ±1Hz.
        x = sum(a * np.sin(2 * np.pi * round(order * 20 / (fs / n)) * (fs / n) * t)
                for order, a in enumerate((2, 1, .5, .25, .125), 1))
        result = analyze_signal(x, fs, 1200)
        extra, audit = extended_features(result, 1200, 1.)
        self.assertAlmostEqual(extra['three_x_amplitude'], .5, places=4)
        self.assertAlmostEqual(extra['two_to_one_ratio'], .5, places=4)
        self.assertAlmostEqual(extra['harmonic_2to5_amplitude_sum'], 1.875, places=4)
        self.assertAlmostEqual(extra['harmonic_sum_over_1x'], .9375, places=4)
        self.assertAlmostEqual(sum(extra[f'energy_{b}_fraction'] for b in ('low', 'mid', 'high')), 1.)
        self.assertTrue(audit['one_x_ratios_defined'])

    def test_near_zero_denominators_missing_not_infinite(self):
        result = analyze_signal(np.zeros(65536), 25600., 900)
        extra, audit = extended_features(result, 900, 1.)
        self.assertFalse(audit['one_x_ratios_defined'])
        self.assertIsNone(extra['two_to_one_ratio'])
        self.assertIsNone(extra['energy_low_fraction'])
        self.assertEqual(extra['harmonic_2to5_amplitude_sum'], 0)
        # Deterministic per-file floor suppresses a tiny 1X even if higher harmonics exist.
        result['amplitude'][np.argmin(abs(result['frequency'] - 30))] = 1
        extra, _ = extended_features(result, 900, 1.)
        self.assertIsNone(extra['two_to_one_ratio'])

    def test_nonfinite_harmonic_values_rejected(self):
        result = analyze_signal(np.zeros(65536), 25600., 900)
        result['amplitude'][np.argmin(abs(result['frequency'] - 30))] = np.inf
        with self.assertRaises(ValueError):
            extended_features(result, 900, 1.)

    def test_missing_MAT_never_fabricates_extensions(self):
        with patch('sqlite3.connect', side_effect=AssertionError('No DB')):
            report = extract_run(self.eda, self.root / 'missing')
        self.assertEqual(report['status'], 'no_MAT_features_extracted')
        self.assertEqual(report['extracted_files'], 0)
        self.assertTrue(all('reason' in r for r in report['files']))

    def test_actual_synthetic_MAT_hash_baseline_checks_and_read_only(self):
        # Build two true MAT fixtures and replace only their synthetic CSV source records.
        data = self.root / 'data'
        rows, entries = [], []
        for i, condition in enumerate(('Normal1', 'Looseness0.1')):
            old = next(r for r in self.csv if r['rpm'] == 900 and r['date'] == '20260912' and r['condition'] == condition)
            row = dict(old)
            path = data / '20260912_900RPM' / condition / 'Vib/20260912_120000_Vib.mat'
            path.parent.mkdir(parents=True)
            t = np.arange(65536) / 25600.
            signal = np.sin(2 * np.pi * 15.234375 * t) + (i + 1) * .1 * np.sin(2 * np.pi * 45 * t)
            signals = np.zeros((6, 65536))
            signals[2] = signal
            savemat(path, {'signals': signals, 'fs': 25600.})
            computed = analyze_signal(signal, 25600., 900)['features']
            row.update(path=str(path), **computed)
            rows.append(row)
            record = dict(next(r for r in self.manifest['files'] if r['path'] == old['path']))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            record.update(path=str(path), sha256_before=sha, sha256_after=sha)
            entries.append(record)
        write_csv(self.eda / 'file_features.csv', rows)
        self.manifest['files'] = entries
        (self.eda / 'analysis.json').write_text(json.dumps(self.manifest))
        before = {p: p.read_bytes() for p in data.rglob('*.mat')}
        result = extract_run(self.eda, self.root / 'real-synthetic')
        self.assertEqual(result['extracted_files'], 2)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        baseline, _, source = load_eda(self.eda)
        common, extended, _, _ = load_extension(self.root / 'real-synthetic', baseline, source)
        self.assertEqual(len(common), 2)
        self.assertEqual(len(extended[0]['values']), len(FEATURES) + len(EXTRA_FEATURES))
        # Same source hash but tampered old feature must exclude this MAT, not overwrite the baseline.
        rows[0]['rms_ac'] += 10
        write_csv(self.eda / 'file_features.csv', rows)
        result = extract_run(self.eda, self.root / 'mismatch')
        self.assertEqual(result['extracted_files'], 1)
        self.assertIn('frozen EDA', result['files'][0]['reason'])

    def test_default_parameterization_reproduces_original_baseline(self):
        rows, _, _ = load_eda(self.eda)
        self.assertEqual(evaluate_rows(rows), evaluate_rows(rows, FEATURES))

    def test_extension_holdout_values_and_labels_do_not_change_training(self):
        rows, _, source = load_eda(self.eda)
        ext = self.root / 'extension'
        fake_extension(ext, rows, source)
        _, extended, _, _ = load_extension(ext, rows, source)
        features = (*FEATURES, *EXTRA_FEATURES)
        first = evaluate_rows(extended, features)
        altered = deepcopy(extended)
        for row in altered:
            if row['date'] == '20260919':
                row['values'] = [1e6] * len(features)
                row['label'] = 'Unbalance'
        second = evaluate_rows(altered, features)
        identity = 'unseen_rpm_lodo_900_20260919'
        for index in (2, 3, 4):
            self.assertEqual([r for r in first[index] if r['fold_id'] == identity],
                             [r for r in second[index] if r['fold_id'] == identity])
        for fold in first[1]:
            self.assertNotIn(fold['test_date'], fold['train_dates'])
            if fold['protocol'] == 'unseen_rpm_lodo':
                self.assertNotIn(fold['test_rpm'], fold['train_rpms'])

    def test_paired_run_same_files_negative_deltas_kept_and_reference_reproduced(self):
        rows, _, source = load_eda(self.eda)
        ext, reference = self.root / 'extension', self.root / 'reference'
        fake_extension(ext, rows[:-1], source)
        predictions = evaluate_rows(rows)[0]
        reference.mkdir()
        write_csv(reference / 'predictions.csv', predictions)
        (reference / 'evaluation.json').write_text(json.dumps({'completed': True, 'inputs_unchanged': True,
            'rf_config': RF_CONFIG, 'input_features': list(FEATURES), 'provenance': source}))
        before = {p: p.read_bytes() for p in [*self.eda.iterdir(), *ext.iterdir(), *reference.iterdir()]}
        with patch('sqlite3.connect', side_effect=AssertionError('No DB')):
            report = compare_run(self.eda, ext, self.root / 'comparison', reference)
        self.assertEqual(report['common_files'], len(rows) - 1)
        self.assertEqual(report['reference_reproduction']['status'], 'matched')
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue(list((self.root / 'comparison').rglob('*.png')))
        baseline = report['baseline_common_metrics']
        worse = deepcopy(baseline)
        for row in worse:
            row['macro_f1_fixed_four_classes'] = -1  # Unit probe verifies signed deltas, never clips decreases.
        self.assertTrue(all(r['delta'] < 0 for r in paired_deltas(baseline, worse) if r['metric'] == 'macro_f1'))

    def test_extension_tampering_and_output_overwrite_rejected(self):
        rows, _, source = load_eda(self.eda)
        ext = self.root / 'extension'
        fake_extension(ext, rows, source)
        with (ext / 'extended_file_features.csv').open('a') as stream:
            stream.write('tampered\n')
        with self.assertRaises(ValueError):
            load_extension(ext, rows, source)
        self.root.joinpath('exists').mkdir()
        with self.assertRaises(FileExistsError):
            shift_run(self.eda, self.root / 'exists')

    def test_date_shift_artifacts_csv_only(self):
        report = shift_run(self.eda, self.root / 'shift')
        self.assertTrue(report['completed'])
        self.assertTrue((self.root / 'shift/date_vs_fault_distance.csv').is_file())
        self.assertTrue(list((self.root / 'shift').glob('*.png')))


if __name__ == '__main__':
    unittest.main()
