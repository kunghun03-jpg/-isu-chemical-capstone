"""Synthetic EDA correctness/safety, never evidence about actual fault signatures."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from scripts.analyze_fault_signatures import (
    CONDITIONS, ROOT, analyze_signal, discover, metadata, run, validate_output,
)

FS, N = 25600., 65536
TIME = np.arange(N) / FS


class MathTests(unittest.TestCase):
    def test_time_features_order_peaks_and_fft_resolution(self):
        # Exact FFT bins, close to nominal 15/30Hz orders at 900RPM.
        f1, f2 = 39 * FS / N, 77 * FS / N
        x = 7 + 3 * np.sin(2 * np.pi * f1 * TIME) + 1.5 * np.sin(2 * np.pi * f2 * TIME)
        result = analyze_signal(x, FS, 900)
        features = result['features']
        self.assertAlmostEqual(features['rms_ac'], np.sqrt((9 + 2.25) / 2), places=10)
        self.assertAlmostEqual(features['mean_raw'], 7, places=10)
        self.assertAlmostEqual(features['rms_raw'] ** 2, 49 + (9 + 2.25) / 2, places=8)
        self.assertAlmostEqual(features['crest_factor_ac'], features['peak_ac'] / features['rms_ac'])
        self.assertEqual(features['fft_resolution_hz'], .390625)
        self.assertEqual(features['one_x_peak_hz'], f1)
        self.assertEqual(features['two_x_peak_hz'], f2)
        self.assertAlmostEqual(features['one_x_amplitude'], 3, places=4)
        self.assertAlmostEqual(features['two_x_amplitude'], 1.5, places=4)
        self.assertEqual(result['peaks'][0]['frequency_hz'], f1)

    def test_psd_band_mean_square(self):
        x = sum(a * np.sin(2 * np.pi * f * TIME) for a, f in ((2, 78.125), (4, 1000), (6, 5000)))
        features = analyze_signal(x, FS, 1200)['features']
        for name, expected in (('low', 2), ('mid', 8), ('high', 18)):
            self.assertAlmostEqual(features['energy_' + name], expected, places=5)

    def test_envelope_recovers_modulation_without_fault_label(self):
        x = (1 + .6 * np.sin(2 * np.pi * 93.75 * TIME)) * np.sin(2 * np.pi * 4000 * TIME)
        result = analyze_signal(x, FS, 1800)
        peak = result['envelope_peaks'][0]
        self.assertLess(abs(peak['frequency_hz'] - 93.75), result['features']['envelope_resolution_hz'])
        self.assertGreater(peak['amplitude'], .5)
        self.assertGreater(result['features']['carrier_power_fraction'], .99)
        self.assertNotIn('diagnosis', result)

    def test_zero_signal_and_invalid_configuration(self):
        result = analyze_signal(np.zeros(N), FS, 900)
        self.assertIsNone(result['features']['crest_factor_ac'])
        self.assertIsNone(result['features']['carrier_power_fraction'])
        self.assertEqual(result['peaks'], [])
        for kwargs in ({'envelope_band': (2000, 13000)}, {'order_tolerance': .1},
                       {'order_tolerance': 8}, {'trim_seconds': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_signal(np.zeros(N), FS, 900, **kwargs)

    def test_nonfinite_and_overflow_rejected(self):
        for value in (np.nan, np.inf, 1e308):
            with self.subTest(value=value), self.assertRaises((ValueError, FloatingPointError)):
                analyze_signal(np.full(N, value), FS, 900)


class FileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()

    def mat(self, condition='Normal1', date='20260912', rpm=900, time='120000', invalid=None):
        path = self.data / date / f'{date}_{rpm}RPM' / condition / 'Vib' / f'{date}_{time}_Vib.mat'
        path.parent.mkdir(parents=True, exist_ok=True)
        # CH1 has a much larger signal to catch accidental channel changes.
        signals = np.zeros((6, N))
        signals[0] = 1000 * np.sin(2 * np.pi * 100 * TIME)
        signals[2] = 2 * np.sin(2 * np.pi * 15.234375 * TIME)
        if invalid == 'nan':
            signals[0, 0] = np.nan  # existing loader checks all six channels
        if invalid == 'overflow':
            signals[2, 0] = 1e101
        savemat(path, {'signals': signals, 'fs': 12800. if invalid == 'fs' else FS})
        return path

    def test_exact_metadata_and_ambiguous_labels(self):
        for c in CONDITIONS:
            meta = metadata(self.mat(c))
            self.assertEqual(meta['condition'], c)
            self.assertEqual(meta['date'], '20260912')
            self.assertIsNone(meta['run_id'])
        for path in ('20260912_900RPM/Normal/Vib/20260912_120000_Vib.mat',
                     '20260912_900RPM/Normal1/Vib/20260919_120000_Vib.mat',
                     '20260912_2940RPM/IR/Vib/20260912_120000_Vib.mat',
                     '20260912_900RPM/IR/Acc/20260912_120000_Vib.mat'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                metadata(path)

    def test_sampling_is_stratified_and_deterministic(self):
        paths = [self.mat(time=f'12000{i}') for i in range(5)]
        other = self.mat('IR', date='20260919', rpm=1800)
        records = discover([self.data, self.data], 3)
        selected = [r['path'] for r in records if r['status'] == 'selected']
        self.assertEqual(set(selected), {str(paths[i]) for i in (0, 2, 4)} | {str(other)})
        self.assertEqual(len(records), 6)
        self.assertEqual(records, discover([self.data], 3))

    def test_all_rpms_and_conditions_have_separate_selection_strata(self):
        for rpm in (900, 1200, 1800):
            for condition in CONDITIONS:
                folder = self.data / f'20260912_{rpm}RPM' / condition / 'Vib'
                folder.mkdir(parents=True)
                for time in ('120000', '130000'):
                    (folder / f'20260912_{time}_Vib.mat').touch()
        records = discover([self.data], 1)
        self.assertEqual(sum(r['status'] == 'selected' for r in records), 30)
        self.assertEqual(sum(r['status'] == 'not_selected' for r in records), 30)
        self.assertEqual({(r['rpm'], r['condition']) for r in records if r['status'] == 'selected'},
                         {(rpm, condition) for rpm in (900, 1200, 1800) for condition in CONDITIONS})

    def test_output_is_fresh_external_and_separate(self):
        for output in (self.data / 'result', self.data, self.root, ROOT / 'output-test'):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                validate_output(output, [self.data])
        checkout = self.root / 'other-checkout'
        (checkout / '.git').mkdir(parents=True)
        (checkout / '.git/HEAD').write_text('ref: refs/heads/test\n')
        with self.assertRaises(ValueError):
            validate_output(checkout / 'results', [self.data])
        self.assertEqual(validate_output(self.root / 'reports/new', [self.data]), self.root / 'reports/new')

    def test_end_to_end_csv_png_json_and_preservation(self):
        good = [self.mat('Normal1'), self.mat('Normal2'), self.mat('IR'),
                self.mat('Normal1', date='20260919', rpm=1200)]
        bad = [self.mat('OR', invalid='nan'), self.mat('Roller', invalid='overflow'),
               self.mat('Looseness0.1', invalid='fs')]
        snapshots = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in good + bad}
        output = self.root / 'results'
        with patch('sqlite3.connect', side_effect=AssertionError('DB forbidden')):
            report = run([self.data], output, files_per_stratum=1)
        self.assertEqual(report['analyzed_files'], 4)
        self.assertEqual(report['counts']['excluded'], 3)
        self.assertTrue(report['completed'])
        self.assertEqual(snapshots, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots})
        with (output / 'file_features.csv').open(encoding='utf-8-sig') as stream:
            features = list(csv.DictReader(stream))
        self.assertEqual(len(features), 4)
        self.assertTrue(all(abs(float(r['rms_ac']) - np.sqrt(2)) < 1e-6 for r in features))
        with (output / 'condition_summary.csv').open(encoding='utf-8-sig') as stream:
            summary = list(csv.DictReader(stream))
        normal = next(r for r in summary if r['condition'] == 'NormalCombined' and r['rpm'] == '900')
        self.assertEqual(normal['n'], '2')
        self.assertEqual((normal['normal1_n'], normal['normal2_n']), ('1', '1'))
        self.assertTrue(all((r['rpm'], r['date']) in [('900', '20260912'), ('1200', '20260919')] for r in summary))
        saved = json.loads((output / 'analysis.json').read_text())
        self.assertFalse(any('diagnosis' in r for r in saved['files']))
        self.assertTrue(all(r['reason'] for r in saved['files'] if r['status'] == 'excluded'))
        self.assertTrue(list(output.rglob('*.png')))
        self.assertTrue((output / '900rpm/20260912/NormalCombined/spectrum_distribution.csv').is_file())
        self.assertTrue((output / 'envelope_filter_response.csv').is_file())
        with self.assertRaises(FileExistsError):
            run([self.data], output)

    def test_empty_or_corrupt_data_does_not_claim_analysis(self):
        path = self.mat()
        path.write_bytes(b'not a MAT')
        report = run([self.data], self.root / 'empty-report')
        self.assertEqual(report['analyzed_files'], 0)
        self.assertEqual(report['status'], 'no_valid_measurements_not_analyzed')
        self.assertEqual(report['counts']['excluded'], 1)
        with self.assertRaises(FileNotFoundError):
            discover([self.root / 'missing'])


if __name__ == '__main__':
    unittest.main()
