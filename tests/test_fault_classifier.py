"""Synthetic CSV/JSON provenance and date-disjoint learning, not real diagnosis performance."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.analyze_fault_signatures import BANDS, CONDITIONS, metadata, write_csv
from scripts.evaluate_fault_classifier import (
    CLASSES, FEATURES, LearnedRules, evaluate_rows, fit_models, fold_specs, load_eda, metrics, run,
)


def fixture(folder, dates=('20260912', '20260916', '20260919'), rpms=(900, 1200)):
    """CSV-only fixture: no MAT, trained model or actual experimental observation."""
    folder.mkdir(parents=True)
    rows, records = [], []
    rng = np.random.default_rng(42)
    for date in dates:
        for rpm in rpms:
            for condition in CONDITIONS:
                for repeat in range(2):
                    path = f'/synthetic-data/{date}/{date}_{rpm}RPM/{condition}/Vib/{date}_12000{repeat}_Vib.mat'
                    meta = metadata(path)
                    values = np.ones(len(FEATURES)) + rng.uniform(0, .05, len(FEATURES))
                    if meta['family'] == 'Bearing':
                        values[7:9] += 10
                    elif meta['family'] == 'Looseness':
                        values[2] += 10
                        values[4] += 10
                    elif meta['family'] == 'Unbalance':
                        values[3] += 10
                    sha = hashlib.sha256(path.encode()).hexdigest()
                    row = {'path': path, 'rpm': rpm, 'date': date, 'condition': condition,
                           'family': meta['family'], 'fs': 25600., 'samples': 65536, 'duration_seconds': 2.56,
                           'fft_resolution_hz': .390625, 'one_x_hz': rpm / 60, 'two_x_hz': rpm / 30,
                           'order_search_tolerance_hz': 1., 'envelope_resolution_hz': 25600 / (65536 - 5120),
                           **dict(zip(FEATURES, values))}
                    rows.append(row)
                    records.append({**{k: row[k] for k in ('path', 'rpm', 'date', 'condition', 'family', 'fs', 'samples', 'duration_seconds')},
                                    'status': 'analyzed', 'channels': 6, 'analyzed_channel': 3,
                                    'sha256_before': sha, 'sha256_after': sha})
    write_csv(folder / 'file_features.csv', rows)
    report = {'completed': True, 'status': 'exploratory_analysis_written',
              'config': {'channel': 'CH3 (index 2)', 'bands_hz': BANDS,
                         'envelope_carrier_hz': [2000, 8000], 'order_search_tolerance_hz': 1.,
                         'envelope_trim_seconds_each_edge': .1}, 'files': records}
    (folder / 'analysis.json').write_text(json.dumps(report))
    return rows, report


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.eda = self.root / 'eda'
        self.csv, self.json = fixture(self.eda)

    def save(self):
        write_csv(self.eda / 'file_features.csv', self.csv)
        (self.eda / 'analysis.json').write_text(json.dumps(self.json))

    def test_load_csv_without_mat_and_keep_exact_conditions(self):
        rows, excluded, provenance = load_eda(self.eda)
        self.assertEqual(len(rows), 120)
        self.assertEqual(excluded, [])
        self.assertEqual(set(r['label'] for r in rows), set(CLASSES))
        self.assertEqual(set(r['condition'] for r in rows), set(CONDITIONS))
        self.assertTrue(all(len(r['values']) == len(FEATURES) for r in rows))
        self.assertIn('not recomputed', provenance['provenance_limit'])

    def test_reject_wrong_date_inf_hash_and_ambiguous_condition(self):
        self.csv[0]['date'] = '20260919'
        self.csv[1]['rms_ac'] = float('inf')
        self.json['files'][2]['sha256_after'] = '0' * 64
        self.csv[3]['condition'] = 'Normal'
        self.save()
        rows, excluded, _ = load_eda(self.eda)
        self.assertEqual(len(excluded), 4)
        self.assertEqual(len(rows), 116)
        self.assertTrue(all(r['reason'] for r in excluded))

    def test_duplicate_paths_and_cross_date_contents_excluded(self):
        self.csv.append(deepcopy(self.csv[0]))
        self.json['files'][20]['sha256_before'] = self.json['files'][60]['sha256_before']
        self.json['files'][20]['sha256_after'] = self.json['files'][60]['sha256_after']
        self.save()
        _, excluded, _ = load_eda(self.eda)
        self.assertEqual(len(excluded), 4)
        self.assertTrue(any('duplicate recorded MAT SHA256' in r['reason'] for r in excluded))

    def test_invalid_schema_and_feature_configuration(self):
        self.csv[0]['one_x_hz'] = 100
        self.save()
        self.assertEqual(len(load_eda(self.eda)[1]), 1)
        self.json['config']['channel'] = 'CH1'
        self.save()
        with self.assertRaises(ValueError):
            load_eda(self.eda)

    def test_date_and_rpm_disjoint_fold_specs(self):
        rows, _, _ = load_eda(self.eda)
        for protocol, rpm, date, train, test in fold_specs(rows):
            self.assertFalse(set(train) & set(test))
            self.assertTrue(all(rows[i]['date'] != date for i in train))
            self.assertTrue(all(rows[i]['date'] == date and rows[i]['rpm'] == rpm for i in test))
            if protocol == 'unseen_rpm_lodo':
                self.assertTrue(all(rows[i]['rpm'] != rpm for i in train))

    def test_training_only_median_and_all_missing_fallback(self):
        x = np.array([[np.nan, np.nan, *([1.] * 7)], [2, np.nan, *([2.] * 7)], [4, np.nan, *([3.] * 7)]])
        imputer, _, _ = fit_models(x, np.array(['Normal', 'Bearing', 'Bearing']))
        self.assertEqual(imputer.statistics_[0], 3)
        self.assertEqual(imputer.statistics_[1], 0)
        held = imputer.transform([[np.nan, 1000000, *([1.] * 7)]])
        self.assertEqual(held[0, 0], 3)
        self.assertEqual(imputer.statistics_[0], 3)
        self.assertTrue(np.isfinite(held).all())

    def test_holdout_changes_do_not_change_training_artifacts(self):
        small = self.root / 'small'
        fixture(small, dates=('20260912', '20260919'), rpms=(900,))
        rows, _, _ = load_eda(small)
        first = evaluate_rows(rows)
        changed = deepcopy(rows)
        for r in changed:
            if r['date'] == '20260919':
                r['values'] = [1e6] * len(FEATURES)
                r['label'] = 'Unbalance'
        second = evaluate_rows(changed)
        fold_id = 'per_rpm_lodo_900_20260919'
        for index in (2, 3, 4):  # rules, RF training importance and imputer
            a = [r for r in first[index] if r['fold_id'] == fold_id]
            b = [r for r in second[index] if r['fold_id'] == fold_id]
            self.assertEqual(a, b)

    def test_learned_rules_are_short_thresholds_with_no_probabilities(self):
        x = np.array([[1.] * len(FEATURES), [2.] * len(FEATURES), [8.] * len(FEATURES), [9.] * len(FEATURES)])
        rules = LearnedRules().fit(x, ['Normal', 'Normal', 'Bearing', 'Bearing'])
        self.assertEqual(len(rules.rules), 1)
        self.assertEqual(rules.rules[0]['feature'], FEATURES[0])
        labels, evidence = rules.predict(np.array([[1.] * len(FEATURES), [10.] * len(FEATURES)]))
        self.assertEqual(labels.tolist(), ['Normal', 'Bearing'])
        self.assertIn('rms_ac', evidence[1])

    def test_metrics_missing_classes_are_explicit(self):
        result = metrics(['Normal', 'Normal', 'Bearing'], ['Normal', 'Bearing', 'Bearing'])
        self.assertAlmostEqual(result['balanced_accuracy_present_classes'], .75)
        self.assertEqual(result['absent_test_classes'], ['Looseness', 'Unbalance'])
        self.assertEqual(np.asarray(result['confusion_matrix']).shape, (4, 4))
        self.assertEqual(result['classes']['Looseness']['support'], 0)

    def test_full_csv_only_run_common_tests_artifacts_and_read_only(self):
        before = {p: p.read_bytes() for p in self.eda.iterdir()}
        output = self.root / 'output'
        with patch('sqlite3.connect', side_effect=AssertionError('No DB')):
            report = run(self.eda, output)
        self.assertEqual(report['evaluated_folds'], 12)
        self.assertTrue(report['inputs_unchanged'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with (output / 'predictions.csv').open(encoding='utf-8-sig') as stream:
            predictions = list(csv.DictReader(stream))
        for protocol in ('per_rpm_lodo', 'unseen_rpm_lodo'):
            a = {r['path'] for r in predictions if r['protocol'] == protocol and r['model'] == 'rules'}
            b = {r['path'] for r in predictions if r['protocol'] == protocol and r['model'] == 'random_forest'}
            self.assertEqual(a, b)
            self.assertEqual(len(a), 120)
        self.assertEqual(report['input_features'], list(FEATURES))
        self.assertFalse(set(FEATURES) & {'path', 'date', 'rpm', 'condition', 'label'})
        self.assertTrue((output / 'condition_errors.csv').is_file())
        self.assertTrue(list(output.glob('*.png')))
        self.assertEqual(len(report['results'][0]['classes']), 4)
        with (output / 'feature_distributions.csv').open(encoding='utf-8-sig') as stream:
            self.assertTrue(any(r['group_type'] == 'condition' and r['condition_or_family'] == 'Normal2'
                                for r in csv.DictReader(stream)))
        with self.assertRaises(FileExistsError):
            run(self.eda, output)

    def test_insufficient_dates_skip_both_methods(self):
        small = self.root / 'one-date'
        fixture(small, dates=('20260912',), rpms=(900,))
        with patch('scripts.evaluate_fault_classifier.plots'):
            report = run(small, self.root / 'small-output')
        self.assertEqual(report['status'], 'insufficient_data_no_evaluation')
        self.assertEqual(report['evaluated_folds'], 0)
        self.assertTrue(all(f['reason'] for f in report['folds']))

    def test_missing_training_class_is_disclosed_not_silently_removed_from_test(self):
        rows, _, _ = load_eda(self.eda)
        rows = [r for r in rows if not (r['label'] == 'Unbalance' and r['date'] != '20260919')]
        predictions, folds, *_ = evaluate_rows(rows)
        target = 'per_rpm_lodo_900_20260919'
        fold = next(f for f in folds if f['fold_id'] == target)
        self.assertTrue(any('training missing classes' in s and 'Unbalance' in s for s in fold['limitations']))
        for model in ('rules', 'random_forest'):
            held = [p for p in predictions if p['fold_id'] == target and p['model'] == model]
            self.assertTrue(any(p['truth'] == 'Unbalance' for p in held))
            self.assertFalse(any(p['predicted'] == 'Unbalance' for p in held))


if __name__ == '__main__':
    unittest.main()
