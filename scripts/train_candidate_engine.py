"""Train once on declared dates, select abstention on validation, evaluate untouched test."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import sklearn
import scipy

from scripts.analyze_fault_signatures import write_csv
from scripts.evaluate_fault_classifier import CLASSES, FEATURES, RF_CONFIG, load_eda, metrics
from scripts.fault_feature_common import new_report, write_json
from vibration.candidate_engine import CONFIG, SCHEMA, LABELS, derive_policy, decide, raw_predictions, sha256, train_pipeline, training_bounds


def split_rows(rows, train_dates, validation_dates, test_dates):
    groups = [set(train_dates), set(validation_dates), set(test_dates)]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Train/Validation/Test 날짜는 비어 있지 않고 서로 겹치면 안 됩니다.')
    for group in groups:
        for date in group:
            datetime.strptime(date, '%Y%m%d')
            if not any(r['date'] == date for r in rows):
                raise ValueError(f'요청 날짜의 검증된 파일이 없습니다: {date}')
    return [[r for r in rows if r['date'] in group] for group in groups]


def evaluate(pipeline, rows, bounds, policy, supported):
    x = np.asarray([r['values'] for r in rows])
    truth = [r['label'] for r in rows]
    predicted, votes, margins = raw_predictions(pipeline, x)
    records = []
    for i, row in enumerate(rows):
        accepted, reasons, _ = decide(str(predicted[i]), float(margins[i]), x[i], row['rpm'], bounds, policy, supported, True)
        records.append({'path': row['path'], 'date': row['date'], 'rpm': row['rpm'], 'condition': row['condition'],
                        'truth': row['label'], 'raw_prediction': str(predicted[i]), 'accepted': accepted,
                        'score_margin': float(margins[i]), 'correct': row['label'] == predicted[i], 'reasons': '; '.join(reasons)})
    summaries = []
    for rpm in [None, *sorted({r['rpm'] for r in rows})]:
        group = [r for r in records if rpm is None or r['rpm'] == rpm]
        accepted = [r for r in group if r['accepted']]
        summaries.append({'rpm': rpm, 'n': len(group), 'accepted_n': len(accepted),
                          'abstention_rate': 1 - len(accepted) / len(group),
                          'selective_accuracy': float(np.mean([r['correct'] for r in accepted])) if accepted else None,
                          'raw_four_class_metrics': metrics([r['truth'] for r in group], [r['raw_prediction'] for r in group]),
                          'by_class': {label: {'n': sum(r['truth'] == label for r in group),
                                              'accepted_n': sum(r['truth'] == label for r in accepted),
                                              'correct_accepted_n': sum(r['truth'] == label and r['correct'] for r in accepted)} for label in CLASSES}})
    return records, summaries


def run(eda_dir, output, train_dates, validation_dates, test_dates):
    rows, excluded, source = load_eda(eda_dir)
    c = source['eda_config']
    if c['envelope_carrier_hz'] != CONFIG['envelope_carrier_hz'] or c['order_search_tolerance_hz'] != 1. or c['envelope_trim_seconds_each_edge'] != .1:
        raise ValueError('웹과 같은 기본 EDA 설정의 CSV가 필요합니다.')
    train, validation, test = split_rows(rows, train_dates, validation_dates, test_dates)
    if any(sum(r['label'] == label for r in train) < 2 for label in CLASSES):
        raise ValueError('Train에는 네 클래스가 각각 최소 2개 필요합니다.')
    with new_report(output, [eda_dir], source['input_sha256'], 'training_report.json', provenance=source) as (out, report):
        x = np.asarray([r['values'] for r in train])
        pipeline = train_pipeline(x, [r['label'] for r in train])
        bounds = training_bounds(x)
        supported = sorted({r['rpm'] for r in train})
        policy = derive_policy(pipeline, np.asarray([r['values'] for r in validation]),
                               [r['label'] for r in validation], [r['rpm'] for r in validation], bounds, supported)
        # Policy is now frozen. Test participates only in predictions/metrics below.
        val_records, val_metrics = evaluate(pipeline, validation, bounds, policy, supported)
        test_records, test_metrics = evaluate(pipeline, test, bounds, policy, supported)
        joblib.dump(pipeline, out / 'model.joblib', compress=3)
        version = 'candidate_rf_eda9_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        manifest = {'schema_version': 1, 'model_version': version, 'feature_schema': SCHEMA,
                    'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                    'features': list(FEATURES), 'feature_config': CONFIG, 'classes': list(CLASSES), 'rf_config': RF_CONFIG,
                    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
                    'model_sha256': sha256(out / 'model.joblib'), 'training_rpms': supported, 'training_bounds': bounds,
                    'policy': policy, 'split_dates': {'train': sorted(set(train_dates)), 'validation': sorted(set(validation_dates)), 'test': sorted(set(test_dates))},
                    'source_eda_sha256': source['input_sha256'], 'auto_labels_default': False,
                    'limitations': report['limitations'] + ['Policy selection uses validation; apparent validation precision is optimistic.',
                        'This model was not refit on validation/test. Previous LODO scores are not this model performance.',
                        'Candidate evaluation uses all EDA files, not an AE-filtered subset; end-to-end gated performance is unverified.']}
        write_csv(out / 'validation_predictions.csv', val_records)
        write_csv(out / 'test_predictions.csv', test_records)
        write_csv(out / 'excluded_eda_rows.csv', excluded)
        splits = []
        for role, group in (('train', train), ('validation', validation), ('test', test)):
            splits.extend({'split': role, **{k: r[k] for k in ('path', 'date', 'rpm', 'condition', 'label', 'mat_sha256_recorded')}} for r in group)
        write_csv(out / 'split_files.csv', splits)
        manifest.update(split_files_sha256=sha256(out / 'split_files.csv'),
                        split_counts={role: len(group) for role, group in (('train', train), ('validation', validation), ('test', test))},
                        class_counts={role: dict(Counter(r['label'] for r in group)) for role, group in (('train', train), ('validation', validation), ('test', test))})
        write_json(out / 'manifest.json', manifest)
        write_csv(out / 'selective_metrics.csv', [{'split': role, **{k: v for k, v in m.items() if k not in ('raw_four_class_metrics', 'by_class')}}
                  for role, summary in (('validation', val_metrics), ('test', test_metrics)) for m in summary])
        write_csv(out / 'class_metrics.csv', [{'split': role, 'rpm': m['rpm'], 'class': label, **v}
                  for role, summary in (('validation', val_metrics), ('test', test_metrics)) for m in summary
                  for label, v in m['raw_four_class_metrics']['classes'].items()])
        write_csv(out / 'raw_confusion.csv', [{'split': role, 'rpm': m['rpm'], 'actual': actual, 'predicted': predicted,
                    'n': m['raw_four_class_metrics']['confusion_matrix'][i][j]}
                  for role, summary in (('validation', val_metrics), ('test', test_metrics)) for m in summary
                  for i, actual in enumerate(CLASSES) for j, predicted in enumerate(CLASSES)])
        write_csv(out / 'policy_confusion.csv', [{'split': role, 'actual': actual, 'displayed': displayed,
                    'n': sum(r['truth'] == actual and (r['raw_prediction'] if r['accepted'] else 'ABSTAIN') == displayed for r in records)}
                  for role, records in (('validation', val_records), ('test', test_records))
                  for actual in CLASSES for displayed in (*CLASSES, 'ABSTAIN')])
        used = {r['path'] for r in splits}
        write_csv(out / 'unused_eda_files.csv', [{'path': r['path'], 'date': r['date']} for r in rows if r['path'] not in used])
        report.update(status='trained_evaluated_no_refit', model_version=version, manifest_sha256=sha256(out / 'manifest.json'),
                      split_counts={role: len(group) for role, group in (('train', train), ('validation', validation), ('test', test))},
                      class_counts={role: dict(Counter(r['label'] for r in group)) for role, group in (('train', train), ('validation', validation), ('test', test))},
                      validation_metrics=val_metrics, test_metrics=test_metrics, policy=policy,
                      features=list(FEATURES), training_imputer_statistics=pipeline.named_steps['imputer'].statistics_.tolist(),
                      interpretation='Selective metrics simulate label enablement with a frozen policy; web defaults to labels disabled until reviewed.')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for role in ('train', 'validation', 'test'):
        p.add_argument('--' + role + '-dates', nargs='+', required=True)
    a = p.parse_args()
    try:
        report = run(a.eda_dir, a.output, a.train_dates, a.validation_dates, a.test_dates)
        print(f"{report['status']}; bundle={a.output}; validation policies={len(report['policy']['rules'])}")
        return 0 if report['completed'] else 1
    except Exception as error:
        print(f'NOT TRAINED: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
