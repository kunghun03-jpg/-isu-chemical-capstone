"""Date-disjoint offline comparison of learned simple rules and a fixed random forest."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from scripts.analyze_fault_signatures import BANDS, metadata, validate_output, write_csv

CLASSES = ('Normal', 'Bearing', 'Looseness', 'Unbalance')
FEATURES = ('rms_ac', 'peak_ac', 'crest_factor_ac', 'one_x_amplitude', 'two_x_amplitude',
            'energy_low', 'energy_mid', 'energy_high', 'envelope_rms_ac')
PROTOCOLS = ('per_rpm_lodo', 'unseen_rpm_lodo')
RF_CONFIG = {'n_estimators': 200, 'max_depth': 5, 'min_samples_leaf': 2,
             'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_path(value):
    path = PurePosixPath(str(value).replace('\\', '/'))
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute unambiguous MAT path required')
    return str(path)


def load_eda(folder):
    """Validate recorded provenance without opening or requiring the original MATs."""
    folder = Path(folder).expanduser().resolve()
    source = folder / 'file_features.csv'
    manifest = folder / 'analysis.json'
    hashes = {str(p): digest(p) for p in (source, manifest)}
    report = json.loads(manifest.read_text(encoding='utf-8'))
    if report.get('completed') is not True or report.get('status') != 'exploratory_analysis_written':
        raise ValueError('EDA report must describe a completed analysis with measurements')
    config = report['config']
    if config.get('channel') != 'CH3 (index 2)' or config.get('bands_hz') != {k: list(v) for k, v in BANDS.items()}:
        raise ValueError('EDA channel or energy bands differ from the supported feature definition')
    carrier = config.get('envelope_carrier_hz', [])
    if len(carrier) != 2 or not 0 < carrier[0] < carrier[1] < 12800:
        raise ValueError('Invalid recorded envelope band')
    tolerance = config.get('order_search_tolerance_hz', 0)
    trim = config.get('envelope_trim_seconds_each_edge', 0)
    if not .390625 <= tolerance < 7.5 or not 0 < 2 * round(trim * 25600) < 65504:
        raise ValueError('Invalid recorded FFT order/trim settings')
    entries = {}
    for entry in report['files']:
        path = canonical_path(entry['path'])
        if path in entries:
            raise ValueError('duplicate paths in analysis.json')
        entries[path] = entry
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError('Duplicate CSV column names')
        required = {'path', 'rpm', 'date', 'condition', 'family', 'fs', 'samples', 'duration_seconds',
                    'fft_resolution_hz', 'one_x_hz', 'two_x_hz', 'order_search_tolerance_hz',
                    'envelope_resolution_hz', *FEATURES}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    path_counts = Counter()
    for row in rows:
        try:
            path_counts[canonical_path(row['path'])] += 1
        except ValueError:
            pass
    valid, excluded = [], []
    for number, row in enumerate(rows, 2):
        rejection = {'csv_line': number, 'path': row.get('path'), 'reason': None}
        try:
            path = canonical_path(row['path'])
            if path_counts[path] != 1:
                raise ValueError('duplicate CSV MAT path; all copies excluded')
            meta = metadata(path)
            entry = entries.get(path)
            if entry is None or entry.get('status') != 'analyzed':
                raise ValueError('MAT not recorded as analyzed in analysis.json')
            for key in ('rpm', 'date', 'condition', 'family'):
                value = int(row[key]) if key == 'rpm' else row[key]
                if value != meta[key] or entry.get(key) != meta[key]:
                    raise ValueError(f'CSV/path/JSON {key} mismatch')
            for key, expected in (('fs', 25600.), ('samples', 65536), ('duration_seconds', 2.56)):
                if float(row[key]) != expected or entry.get(key) != expected:
                    raise ValueError(f'CSV/JSON {key} mismatch')
            if entry.get('channels') != 6 or entry.get('analyzed_channel') != 3:
                raise ValueError('JSON channel metadata mismatch')
            for key, expected in (('fft_resolution_hz', .390625), ('one_x_hz', meta['rpm'] / 60),
                                  ('two_x_hz', meta['rpm'] / 30), ('order_search_tolerance_hz', tolerance),
                                  ('envelope_resolution_hz', 25600 / (65536 - 2 * round(trim * 25600)))):
                if float(row[key]) != expected:
                    raise ValueError(f'CSV feature configuration mismatch: {key}')
            checksum = entry.get('sha256_before', '')
            if not re.fullmatch(r'[0-9a-f]{64}', checksum) or checksum != entry.get('sha256_after'):
                raise ValueError('missing/inconsistent original MAT SHA256 record')
            values, missing = [], []
            for feature in FEATURES:
                text = row[feature].strip()
                value = np.nan if text.lower() in ('', 'nan') else float(text)
                if np.isinf(value) or (np.isfinite(value) and not 0 <= value <= np.finfo(np.float32).max):
                    raise ValueError(f'invalid nonnegative feature: {feature}')
                if np.isnan(value):
                    missing.append(feature)
                values.append(value)
            if len(missing) == len(FEATURES):
                raise ValueError('all features missing')
            valid.append({'path': path, **meta, 'label': meta['family'], 'mat_sha256_recorded': checksum,
                          'values': values, 'missing_features': missing})
        except (ValueError, KeyError, TypeError) as error:
            rejection['reason'] = str(error)
            excluded.append(rejection)
    # Equal content at distinct paths can otherwise leak across dates or RPMs.
    content_counts = Counter(r['mat_sha256_recorded'] for r in valid)
    unique = []
    for row in valid:
        if content_counts[row['mat_sha256_recorded']] > 1:
            excluded.append({'path': row['path'], 'reason': 'duplicate recorded MAT SHA256; all copies excluded'})
        else:
            unique.append(row)
    csv_paths = set(path_counts)
    omissions = [p for p, entry in entries.items() if entry.get('status') == 'analyzed' and p not in csv_paths]
    return unique, excluded, {'input_sha256': hashes, 'eda_config': config,
                              'analyzed_JSON_paths_missing_from_CSV': omissions,
                              'provenance_limit': 'Recorded path/metadata/SHA256 consistency only. MATs are not reread and feature values are not recomputed.'}


class LearnedRules:
    """One training-selected threshold per non-Normal class, deterministic conflict priority."""

    def __init__(self, features=FEATURES):
        self.features = tuple(features)

    def fit(self, x, labels):
        labels = np.asarray(labels)
        self.rules = []
        for label in CLASSES[1:]:
            positive = labels == label
            if not positive.any() or positive.all():
                continue
            best, best_key = None, None
            for column, feature in enumerate(self.features):
                values = np.unique(x[:, column])
                # Include boundaries so constant features can be recognized as uninformative.
                thresholds = np.unique(np.r_[values[0], values[:-1] + np.diff(values) / 2, values[-1]])
                for direction in ('>', '<='):
                    for threshold in thresholds:
                        hit = x[:, column] > threshold
                        if direction == '<=':
                            hit = ~hit
                        sensitivity = float(hit[positive].mean())
                        specificity = float((~hit[~positive]).mean())
                        quality = (sensitivity + specificity) / 2
                        key = (quality, -column, direction == '>', -float(threshold))
                        if best_key is None or key > best_key:
                            best_key = key
                            best = {'label': label, 'feature': feature, 'column': column,
                                    'direction': direction, 'threshold': float(threshold),
                                    'training_binary_balanced_accuracy': quality,
                                    'training_positive_n': int(positive.sum()),
                                    'training_negative_n': int((~positive).sum())}
            if best['training_binary_balanced_accuracy'] > .5:
                self.rules.append(best)
        # Only training quality determines conflicts; class order settles ties.
        self.rules.sort(key=lambda r: (-r['training_binary_balanced_accuracy'], CLASSES.index(r['label'])))
        return self

    def predict(self, x):
        labels, evidence = [], []
        for values in x:
            active = [r for r in self.rules if (values[r['column']] > r['threshold']) == (r['direction'] == '>')]
            labels.append(active[0]['label'] if active else 'Normal')
            evidence.append('; '.join(f"{r['label']}: {r['feature']}={values[r['column']]:.8g} {r['direction']} {r['threshold']:.8g}" for r in active)
                            or 'No learned fault condition active -> Normal')
        return np.asarray(labels), evidence


def fit_models(x_train, y_train, features=FEATURES):
    # No scaling is needed for axis-threshold rules or RF. Both see exactly the same imputed columns.
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x = imputer.fit_transform(x_train)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite training preprocessing result')
    rules = LearnedRules(features).fit(x, y_train)
    forest = RandomForestClassifier(**RF_CONFIG).fit(x, y_train)
    return imputer, rules, forest


def fold_specs(rows):
    dates = sorted({r['date'] for r in rows})
    rpms = sorted({r['rpm'] for r in rows})
    for protocol in PROTOCOLS:
        for rpm in rpms:
            for date in dates:
                test = [i for i, r in enumerate(rows) if r['rpm'] == rpm and r['date'] == date]
                if not test:
                    continue
                train = [i for i, r in enumerate(rows) if r['date'] != date and
                         ((r['rpm'] == rpm) if protocol == 'per_rpm_lodo' else (r['rpm'] != rpm))]
                yield protocol, rpm, date, train, test


def metrics(truth, predicted):
    matrix = confusion_matrix(truth, predicted, labels=CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=CLASSES, zero_division=0)
    present = support > 0
    return {'n': len(truth), 'macro_f1_fixed_four_classes': float(f1.mean()),
            'balanced_accuracy_present_classes': float(recall[present].mean()),
            'macro_recall_fixed_four_classes': float(recall.mean()),
            'absent_test_classes': [c for c, n in zip(CLASSES, support) if n == 0],
            'confusion_matrix': matrix.tolist(),
            'classes': {c: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(n)}
                        for c, p, r, f, n in zip(CLASSES, precision, recall, f1, support)}}


def distributions(rows, output):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['rpm'], row['date'], 'condition', row['condition'])].append(row)
        groups[(row['rpm'], row['date'], 'family', row['label'])].append(row)
    summary = []
    for (rpm, date, kind, label), group in sorted(groups.items()):
        normal = [r for r in rows if r['rpm'] == rpm and r['date'] == date and r['label'] == 'Normal']
        for j, feature in enumerate(FEATURES):
            v = np.asarray([r['values'][j] for r in group])
            v = v[np.isfinite(v)]
            n = np.asarray([r['values'][j] for r in normal])
            n = n[np.isfinite(n)]
            median = float(np.median(v)) if v.size else None
            reference = float(np.median(n)) if n.size else None
            summary.append({'rpm': rpm, 'date': date, 'group_type': kind, 'condition_or_family': label,
                            'feature': feature, 'n_files': len(group), 'n_observed': int(v.size),
                            'missing_n': len(group) - int(v.size), 'mean': float(v.mean()) if v.size else None,
                            'median': median, 'p25': float(np.percentile(v, 25)) if v.size else None,
                            'p75': float(np.percentile(v, 75)) if v.size else None,
                            'normal_n_observed': int(n.size), 'normal_median': reference,
                            'median_minus_normal': median - reference if median is not None and reference is not None else None,
                            'median_ratio_to_normal': (lambda ratio: ratio if np.isfinite(ratio) else None)(median / reference)
                            if median is not None and reference is not None and reference > 0 else None})
    write_csv(output / 'feature_distributions.csv', summary)
    return summary


def evaluate_rows(rows, features=FEATURES):
    predictions, folds, thresholds, importance, medians = [], [], [], [], []
    x_all = np.asarray([r['values'] for r in rows], dtype=float).reshape(-1, len(features))
    y_all = np.asarray([r['label'] for r in rows])
    for protocol, rpm, date, train, test in fold_specs(rows):
        identifier = f'{protocol}_{rpm}_{date}'
        fold = {'fold_id': identifier, 'protocol': protocol, 'test_rpm': rpm, 'test_date': date,
                'train_dates': sorted({rows[i]['date'] for i in train}),
                'train_rpms': sorted({rows[i]['rpm'] for i in train}),
                'train_paths': [rows[i]['path'] for i in train], 'test_paths': [rows[i]['path'] for i in test],
                'training_class_counts': dict(Counter(y_all[train].tolist())),
                'test_class_counts': dict(Counter(y_all[test].tolist())), 'status': 'skipped', 'limitations': []}
        assert date not in fold['train_dates']
        assert not set(fold['train_paths']) & set(fold['test_paths'])
        if protocol == 'unseen_rpm_lodo':
            assert rpm not in fold['train_rpms']
        folds.append(fold)
        if not train or len(set(y_all[train])) < 2 or 'Normal' not in y_all[train]:
            fold['reason'] = 'Need other-date training files with Normal and at least one fault class; both methods skipped'
            continue
        for role, counts in (('training', fold['training_class_counts']), ('test', fold['test_class_counts'])):
            missing = [c for c in CLASSES if counts.get(c, 0) == 0]
            if missing:
                fold['limitations'].append(f'{role} missing classes: {missing}')
            if any(n < 5 for n in counts.values()):
                fold['limitations'].append(f'{role} class counts <5; unstable descriptive estimates')
        if len(fold['train_dates']) < 2:
            fold['limitations'].append('Only one training date; weak evidence of generalization')
        # Fit once on training dates; no test labels enter preprocessing or model fitting.
        imputer, rules, forest = fit_models(x_all[train], y_all[train], features)
        x_test = imputer.transform(x_all[test])
        if not np.isfinite(x_test).all():
            raise ValueError(f'{identifier}: nonfinite test preprocessing; no partial comparison')
        rule_labels, evidence = rules.predict(x_test)
        rf_labels, votes = forest.predict(x_test), forest.predict_proba(x_test)
        for feature, median in zip(features, imputer.statistics_):
            medians.append({'fold_id': identifier, 'feature': feature, 'training_median_or_zero': float(median),
                            'all_training_values_missing': bool(np.isnan(x_all[train, features.index(feature)]).all())})
        for priority, rule in enumerate(rules.rules, 1):
            thresholds.append({'fold_id': identifier, 'priority': priority, **rule})
        for feature, value in zip(features, forest.feature_importances_):
            importance.append({'fold_id': identifier, 'protocol': protocol, 'test_rpm': rpm, 'test_date': date,
                               'feature': feature, 'impurity_importance': float(value)})
        for j, i in enumerate(test):
            row = rows[i]
            for method, labels in (('rules', rule_labels), ('random_forest', rf_labels)):
                item = {'fold_id': identifier, 'protocol': protocol, 'model': method,
                        'path': row['path'], 'rpm': rpm, 'date': date, 'condition': row['condition'],
                        'truth': row['label'], 'predicted': str(labels[j]), 'correct': bool(labels[j] == row['label']),
                        'missing_features': ';'.join(row['missing_features']),
                        'evidence': evidence[j] if method == 'rules' else 'Fixed RF; vote fractions are not calibrated probabilities'}
                if method == 'random_forest':
                    for label, value in zip(forest.classes_, votes[j]):
                        item['uncalibrated_vote_fraction_' + label] = float(value)
                predictions.append(item)
        fold['status'] = 'evaluated'
    return predictions, folds, thresholds, importance, medians


def aggregate_predictions(predictions):
    groups = defaultdict(list)
    for row in predictions:
        for scope, rpm, date in (('overall', '', ''), ('rpm', row['rpm'], ''),
                                 ('date', '', row['date']), ('rpm_date', row['rpm'], row['date'])):
            groups[(row['protocol'], row['model'], scope, rpm, date)].append(row)
    output = []
    for (protocol, model, scope, rpm, date), group in groups.items():
        result = metrics([r['truth'] for r in group], [r['predicted'] for r in group])
        output.append({'protocol': protocol, 'model': model, 'scope': scope, 'rpm': rpm, 'date': date, **result})
    return output


def plots(output, summaries, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for result in results:
        if result['scope'] != 'overall':
            continue
        matrix = np.asarray(result['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(matrix, cmap='Blues')
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
        ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='Actual', title=f"{result['protocol']} | {result['model']}\nDate-disjoint preliminary evaluation")
        fig.tight_layout()
        fig.savefig(output / f"confusion_{result['protocol']}_{result['model']}.png", dpi=130)
        plt.close(fig)
    for rpm in sorted({s['rpm'] for s in summaries}):
        fig, axes = plt.subplots(5, 2, figsize=(14, 16))
        dates = sorted({s['date'] for s in summaries if s['rpm'] == rpm})
        for feature, ax in zip(FEATURES, axes.flat):
            for condition in sorted({s['condition_or_family'] for s in summaries if s['group_type'] == 'condition'}):
                values = {s['date']: s['median'] for s in summaries if s['rpm'] == rpm and s['group_type'] == 'condition'
                          and s['feature'] == feature and s['condition_or_family'] == condition}
                ax.plot(dates, [values.get(d, np.nan) if values.get(d) is not None else np.nan for d in dates], '.-', label=condition)
            ax.set_title(feature)
            ax.tick_params(axis='x', rotation=30)
        axes.flat[0].legend(fontsize=7, ncol=2)
        axes.flat[-1].set_visible(False)
        fig.suptitle(f'{rpm}RPM | Within-date condition medians; descriptive only, not classifier input')
        fig.tight_layout()
        fig.savefig(output / f'feature_date_shift_{rpm}rpm.png', dpi=130)
        plt.close(fig)


def run(eda_dir, output):
    eda_dir = Path(eda_dir).expanduser().resolve()
    output = validate_output(output, [eda_dir])
    rows, excluded, provenance = load_eda(eda_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = {'completed': False, 'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'input': str(eda_dir), 'validated_files': len(rows), 'excluded_files': len(excluded),
              'input_features': list(FEATURES), 'classes': CLASSES, 'protocols': PROTOCOLS,
              'rf_config': RF_CONFIG, 'provenance': provenance,
              'software': {'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__},
              'learning': {'imputer': 'training-only median; entirely missing training columns use fixed zero',
                           'scaling': 'none; axis thresholds and RF do not require it',
                           'feature_selection': 'no common filter; both receive nine fixed columns; rules select one per class using training only',
                           'rules': 'one class-vs-rest training-selected feature/direction/threshold per fault class; maximize binary balanced accuracy; require >0.5',
                           'rule_conflict': 'higher training binary balanced accuracy, then Bearing/Looseness/Unbalance order; no active rule -> Normal'},
              'limitations': ['독립성 미확인 예비 평가: dates are disjoint but Run independence is not verified.',
                              'Original MATs are not opened; CSV features are not independently recomputed.',
                              'Rule learning and RF use labeled faults; these are offline candidate classifiers, not the normal-only AE.',
                              'No AE gating: this evaluates four-class candidates on all valid EDA files, not end-to-end AE detection plus diagnosis.',
                              'RF vote fractions are not calibrated probabilities. Rules have no candidate probability.',
                              'Feature importance is exploratory impurity importance, not causal or physical proof.',
                              'No hyperparameter or feature selection based on held-out dates.',
                              'Descriptive full-data summaries must not be used to retune this evaluation.',
                              'Unbalance_-1 is reported separately; its physical severity is not verified from the label alone.']}
    try:
        summaries = distributions(rows, output)
        predictions, folds, thresholds, importance, medians = evaluate_rows(rows)
        results = aggregate_predictions(predictions)
        write_csv(output / 'validated_files.csv', [{k: v for k, v in r.items() if k != 'values'} for r in rows])
        write_csv(output / 'excluded_files.csv', excluded)
        write_csv(output / 'predictions.csv', predictions)
        write_csv(output / 'misclassified_files.csv', [p for p in predictions if not p['correct']])
        write_csv(output / 'rule_thresholds.csv', thresholds)
        write_csv(output / 'training_imputation.csv', medians)
        write_csv(output / 'rf_feature_importance.csv', importance)
        metric_rows, class_rows, matrix_rows = [], [], []
        for result in results:
            base = {k: result[k] for k in ('protocol', 'model', 'scope', 'rpm', 'date')}
            metric_rows.append({**base, **{k: v for k, v in result.items() if k not in ('classes', 'confusion_matrix', *base)}})
            for label, values in result['classes'].items():
                class_rows.append({**base, 'class': label, **values})
            for i, actual in enumerate(CLASSES):
                for j, predicted in enumerate(CLASSES):
                    matrix_rows.append({**base, 'actual': actual, 'predicted': predicted, 'count': result['confusion_matrix'][i][j]})
        write_csv(output / 'metrics.csv', metric_rows)
        write_csv(output / 'class_metrics.csv', class_rows)
        write_csv(output / 'confusion_matrices.csv', matrix_rows)
        errors = defaultdict(list)
        for item in predictions:
            errors[(item['protocol'], item['model'], item['rpm'], item['date'], item['condition'])].append(item)
        write_csv(output / 'condition_errors.csv',
                  [{'protocol': p, 'model': m, 'rpm': rpm, 'date': d, 'condition': c, 'n': len(group),
                    'errors': sum(not r['correct'] for r in group),
                    'error_rate': float(np.mean([not r['correct'] for r in group])),
                    **{'predicted_' + label: sum(r['predicted'] == label for r in group) for label in CLASSES}}
                   for (p, m, rpm, d, c), group in errors.items()])
        plots(output, summaries, results)
        report.update(folds=folds, results=results,
                      coverage_by_protocol={protocol: {'eligible_files': len(rows),
                          'evaluated_files': len({p['path'] for p in predictions if p['protocol'] == protocol and p['model'] == 'rules'}),
                          'note': 'Overall metrics use evaluated files only; inspect skipped folds before comparing.'}
                                            for protocol in PROTOCOLS},
                      evaluated_folds=sum(f['status'] == 'evaluated' for f in folds),
                      skipped_folds=sum(f['status'] == 'skipped' for f in folds),
                      status='preliminary_date_disjoint_evaluation' if predictions else 'insufficient_data_no_evaluation',
                      completed=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['inputs_unchanged'] = all(Path(p).is_file() and digest(p) == sha for p, sha in provenance['input_sha256'].items())
        if not report['inputs_unchanged']:
            report.update(completed=False, status='failed', error='EDA inputs changed during execution; results invalid')
        (output / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eda-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path.home() / 'fault-classifier-evaluation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        report = run(args.eda_dir, output)
    except Exception as error:
        print(f'NOT EVALUATED: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ('status', 'validated_files', 'excluded_files', 'evaluated_folds', 'skipped_folds')}, ensure_ascii=False))
    print(f'Output: {output}')
    for result in report['results']:
        if result['scope'] == 'overall':
            print(f"{result['protocol']} | {result['model']} | n={result['n']} | "
                  f"macro F1={result['macro_f1_fixed_four_classes']:.4f} | "
                  f"balanced accuracy={result['balanced_accuracy_present_classes']:.4f}")
    if report['status'] == 'failed':
        return 1
    return 0 if report['completed'] and report['evaluated_folds'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
