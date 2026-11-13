"""Paired baseline/extended feature evaluation using the original date-disjoint learner."""
import argparse
from collections import Counter, defaultdict
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

from scripts.analyze_fault_signatures import write_csv
from scripts.evaluate_fault_classifier import (
    CLASSES, FEATURES, RF_CONFIG, aggregate_predictions, canonical_path, digest, evaluate_rows, load_eda, plots,
)
from scripts.extract_harmonic_features import EXTRA_FEATURES
from scripts.fault_feature_common import default_output, new_report, write_json


def load_extension(folder, rows, source):
    folder = Path(folder).expanduser().resolve()
    manifest_path, csv_path = folder / 'harmonic_extraction.json', folder / 'extended_file_features.csv'
    manifest = json.loads(manifest_path.read_text())
    if (not manifest.get('completed') or not manifest.get('inputs_unchanged') or
            manifest.get('status') != 'harmonic_extraction_complete' or
            manifest.get('eda_input_sha256') != source['input_sha256'] or
            manifest.get('extra_features') != list(EXTRA_FEATURES) or
            manifest.get('baseline_features') != list(FEATURES) or
            digest(csv_path) != manifest.get('extended_csv_sha256')):
        raise ValueError('Extension provenance/config/status/CSV hash mismatch')
    original = {r['path']: r for r in rows}
    extracted = [r for r in manifest['files'] if r['status'] == 'extracted']
    records = {canonical_path(r['path']): r for r in extracted}
    if len(records) != len(extracted):
        raise ValueError('Duplicate extracted paths in manifest')
    values = {}
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError('Duplicate extension columns')
        for row in reader:
            path = canonical_path(row['path'])
            if path in values or path not in original or path not in records:
                raise ValueError('Duplicate/unrecognized extension MAT path')
            base, record = original[path], records[path]
            for key in ('rpm', 'date', 'condition', 'label'):
                if (int(row[key]) if key == 'rpm' else row[key]) != base[key]:
                    raise ValueError('Extension metadata mismatch')
            if row['mat_sha256'] != base['mat_sha256_recorded'] or record.get('mat_sha256_actual') != row['mat_sha256']:
                raise ValueError('Extension MAT SHA256 mismatch')
            parsed = []
            for feature in (*FEATURES, *EXTRA_FEATURES):
                value = None if row[feature] == '' else float(row[feature])
                if value is not None and (not np.isfinite(value) or not 0 <= value <= np.finfo(np.float32).max):
                    raise ValueError('Invalid extension feature')
                if value != record['values'][feature]:
                    raise ValueError('Extension CSV and JSON feature values differ')
                parsed.append(np.nan if value is None else value)
            if not np.array_equal(parsed[:len(FEATURES)], base['values'], equal_nan=True):
                raise ValueError('Frozen baseline values changed')
            values[path] = parsed
    if set(values) != set(records):
        raise ValueError('CSV/extraction success lists differ')
    common = [deepcopy(r) for r in rows if r['path'] in values]
    extended = deepcopy(common)
    for row in extended:
        row['values'] = values[row['path']]
        row['missing_features'] = [f for f, value in zip((*FEATURES, *EXTRA_FEATURES), row['values']) if np.isnan(value)]
    return common, extended, manifest, {str(p): digest(p) for p in (manifest_path, csv_path)}


def save_evaluation(folder, rows, features):
    folder.mkdir()
    predictions, folds, rules, importance, imputation = evaluate_rows(rows, features)
    results = aggregate_predictions(predictions)
    for name, records in (('predictions', predictions), ('rule_thresholds', rules),
                          ('rf_feature_importance', importance), ('training_imputation', imputation)):
        write_csv(folder / (name + '.csv'), records)
    metrics, classes, matrices = [], [], []
    for result in results:
        base = {k: result[k] for k in ('protocol', 'model', 'scope', 'rpm', 'date')}
        metrics.append({**base, **{k: v for k, v in result.items() if k not in (*base, 'classes', 'confusion_matrix')}})
        classes.extend({**base, 'class': c, **v} for c, v in result['classes'].items())
        matrices.extend({**base, 'actual': a, 'predicted': b, 'count': result['confusion_matrix'][i][j]}
                        for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES))
    write_csv(folder / 'metrics.csv', metrics)
    write_csv(folder / 'class_metrics.csv', classes)
    write_csv(folder / 'confusion_matrices.csv', matrices)
    write_json(folder / 'folds.json', folds)
    plots(folder, [], results)
    return predictions, folds, results


def paired_deltas(baseline, extended):
    key = lambda r: (r['protocol'], r['model'], r['scope'], r['rpm'], r['date'])
    old, new = {key(r): r for r in baseline}, {key(r): r for r in extended}
    if old.keys() != new.keys():
        raise ValueError('Metric scopes differ; not a paired comparison')
    rows = []
    for k, before in old.items():
        after = new[k]
        changes = {'macro_f1': (before['macro_f1_fixed_four_classes'], after['macro_f1_fixed_four_classes']),
                   'balanced_accuracy': (before['balanced_accuracy_present_classes'], after['balanced_accuracy_present_classes'])}
        for label in CLASSES:
            for metric in ('precision', 'recall', 'f1'):
                changes[label + '_' + metric] = (before['classes'][label][metric], after['classes'][label][metric])
        for name, (a, b) in changes.items():
            rows.append({'protocol': k[0], 'model': k[1], 'scope': k[2], 'rpm': k[3], 'date': k[4],
                         'metric': name, 'n': before['n'], 'baseline': a, 'extended': b, 'delta': b - a})
    return rows


def compare_reference(folder, predictions, source):
    folder = Path(folder).expanduser().resolve()
    report_path, csv_path = folder / 'evaluation.json', folder / 'predictions.csv'
    report = json.loads(report_path.read_text())
    if (not report.get('completed') or not report.get('inputs_unchanged') or
            report.get('rf_config') != RF_CONFIG or report.get('input_features') != list(FEATURES) or
            report.get('provenance', {}).get('input_sha256') != source['input_sha256']):
        raise ValueError('Archived evaluation settings or EDA input identity differ; cannot claim reproduction')
    key = lambda r: (r['protocol'], r['model'], r['fold_id'], r['path'])
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        archived = list(csv.DictReader(stream))
    before, after = {key(r): r for r in archived}, {key(r): r for r in predictions}
    if len(before) != len(archived):
        raise ValueError('Duplicate prediction identities in archived evaluation')
    differences = [{'identity': str(k), 'archived_truth': before.get(k, {}).get('truth'),
                    'new_truth': after.get(k, {}).get('truth'), 'archived_prediction': before.get(k, {}).get('predicted'),
                    'new_prediction': after.get(k, {}).get('predicted')}
                   for k in sorted(before.keys() | after.keys())
                   if k not in before or k not in after or (before[k]['truth'], before[k]['predicted']) != (after[k]['truth'], after[k]['predicted'])]
    return {'status': 'matched' if not differences else 'mismatch', 'differences': differences}, {str(p): digest(p) for p in (report_path, csv_path)}


def delta_plots(output, deltas):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('macro_f1', 'balanced_accuracy', 'Normal_recall', 'Looseness_recall', 'Unbalance_recall')
    for protocol in sorted({r['protocol'] for r in deltas}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for model, ax in zip(('rules', 'random_forest'), axes):
            values = {r['metric']: r for r in deltas if r['scope'] == 'overall' and r['protocol'] == protocol and r['model'] == model}
            x = np.arange(len(names))
            ax.bar(x - .2, [values[n]['baseline'] for n in names], .4, label='Common baseline')
            ax.bar(x + .2, [values[n]['extended'] for n in names], .4, label='Common extended')
            ax.set(xticks=x, xticklabels=names, ylim=(0, 1), title=model)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(fontsize=8)
        fig.suptitle(protocol + ' | Paired exploratory evaluation; retain decreases')
        fig.tight_layout()
        fig.savefig(output / f'paired_metrics_{protocol}.png', dpi=130)
        plt.close(fig)


def run(eda_dir, extension_dir, output, reference=None):
    rows, excluded, source = load_eda(eda_dir)
    common, extended, extraction, hashes = load_extension(extension_dir, rows, source)
    hashes.update(source['input_sha256'])
    protected = [eda_dir, extension_dir]
    if reference:
        reference = Path(reference).expanduser().resolve()
        hashes.update({str(reference / name): digest(reference / name) for name in ('evaluation.json', 'predictions.csv')})
        protected.append(reference)
    with new_report(output, protected, hashes, 'comparison.json', rf_config=RF_CONFIG,
                    baseline_features=list(FEATURES), extended_features=[*FEATURES, *EXTRA_FEATURES],
                    provenance=source, ratio_policy=extraction['ratio_policy']) as (out, report):
        write_csv(out / 'eda_exclusions.csv', excluded)
        write_csv(out / 'common_cohort.csv', [{k: r[k] for k in ('path', 'rpm', 'date', 'condition', 'label')} for r in common])
        common_paths = {r['path'] for r in common}
        reasons = {r['path']: r.get('reason', 'not extracted') for r in extraction['files'] if r['status'] != 'extracted'}
        write_csv(out / 'not_in_common_cohort.csv', [{'path': r['path'], 'rpm': r['rpm'], 'date': r['date'],
                                                    'condition': r['condition'], 'reason': reasons.get(r['path'], 'not in extraction cohort')}
                                                   for r in rows if r['path'] not in common_paths])
        full, full_folds, full_metrics = save_evaluation(out / 'baseline_all', rows, FEATURES)
        old, old_folds, old_metrics = save_evaluation(out / 'baseline_common', common, FEATURES)
        new, new_folds, new_metrics = save_evaluation(out / 'extended_common', extended, (*FEATURES, *EXTRA_FEATURES))
        if old_folds != new_folds:
            raise ValueError('Baseline/extension fold identity or status differs')
        key = lambda r: (r['protocol'], r['model'], r['fold_id'], r['path'])
        previous = {key(r): r for r in old}
        if previous.keys() != {key(r) for r in new}:
            raise ValueError('Baseline/extension test identities differ')
        pairs = []
        for row in new:
            before = previous[key(row)]
            if row['truth'] != before['truth']:
                raise ValueError('Paired labels differ')
            pairs.append({**{k: row[k] for k in ('protocol', 'model', 'fold_id', 'path', 'rpm', 'date', 'condition', 'truth')},
                          'baseline_prediction': before['predicted'], 'extended_prediction': row['predicted'],
                          'baseline_correct': before['correct'], 'extended_correct': row['correct'],
                          'transition': 'improved' if row['correct'] and not before['correct'] else
                                        'worsened' if before['correct'] and not row['correct'] else
                                        'still_wrong' if not row['correct'] else 'still_correct'})
        write_csv(out / 'paired_predictions.csv', pairs)
        write_csv(out / 'failure_cases.csv', [r for r in pairs if r['transition'] != 'still_correct'])
        condition_groups = defaultdict(list)
        for row in pairs:
            condition_groups[(row['protocol'], row['model'], row['rpm'], row['date'], row['condition'])].append(row)
        write_csv(out / 'condition_transitions.csv',
                  [{'protocol': p, 'model': m, 'rpm': rpm, 'date': date, 'condition': c, 'n': len(group),
                    **dict(Counter(r['transition'] for r in group))}
                   for (p, m, rpm, date, c), group in condition_groups.items()])
        deltas = paired_deltas(old_metrics, new_metrics)
        write_csv(out / 'metric_deltas.csv', deltas)
        write_csv(out / 'overall_summary.csv', [r for r in deltas if r['scope'] == 'overall'])
        delta_plots(out, deltas)
        reference_result = {'status': 'not_provided; baseline_all recomputed, archived reproduction unverified'}
        if reference:
            reference_result, _ = compare_reference(reference, full, source)
            write_csv(out / 'reference_prediction_differences.csv', reference_result['differences'])
        report.update(status='paired_exploratory_evaluation' if old else 'insufficient_data_no_paired_evaluation',
                      validated_eda_files=len(rows), common_files=len(common), paired_predictions=len(pairs),
                      reference_reproduction=reference_result,
                      evaluated_folds=sum(f['status'] == 'evaluated' for f in old_folds),
                      skipped_folds=sum(f['status'] == 'skipped' for f in old_folds),
                      baseline_all_metrics=full_metrics, baseline_common_metrics=old_metrics,
                      extended_common_metrics=new_metrics,
                      notes=['Only common baseline vs common extension deltas measure feature changes.',
                             'Baseline_all vs baseline_common reflects cohort changes, not feature improvements.',
                             'RF configuration unchanged, including default sqrt max_features; its effective candidate count depends on feature count.',
                             'No AE gating; candidate classification is not full-system performance. RF votes are uncalibrated.'])
        if reference_result['status'] == 'mismatch':
            report['status'] = 'reference_baseline_mismatch_review_required'
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-dir', type=Path, required=True)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--reference-evaluation', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        out = args.output or default_output('comparison')
        report = run(args.eda_dir, args.extension_dir, out, args.reference_evaluation)
        print(f"{report['status']}; common={report['common_files']}/{report['validated_eda_files']}; output={out}")
        return 0 if report['completed'] and report['status'] == 'paired_exploratory_evaluation' else 2
    except Exception as error:
        print(f'NOT COMPARED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
