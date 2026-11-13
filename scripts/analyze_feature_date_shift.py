"""Descriptive date shifts vs Normal/Looseness contrasts, without fitting a classifier."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from scripts.analyze_fault_signatures import write_csv
from scripts.evaluate_fault_classifier import FEATURES, distributions, load_eda, plots
from scripts.fault_feature_common import default_output, finite_ratio, new_report


def contrast(a, b):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    result = {'n_a': int(a.size), 'n_b': int(b.size), 'small_sample': bool(min(a.size, b.size) < 5)}
    if not a.size or not b.size:
        return {**result, 'status': 'missing_observations'}
    low, high = min(a.min(), b.min()), max(a.max(), b.max())
    if low == high:
        overlap = 1.
    else:
        edges = np.linspace(low, high, 11)
        overlap = float(np.minimum(np.histogram(a, edges)[0] / a.size,
                                   np.histogram(b, edges)[0] / b.size).sum())
    superiority = float(((b[:, None] > a).sum() + .5 * (b[:, None] == a).sum()) / (a.size * b.size))
    return {**result, 'status': 'descriptive_only', 'median_a': float(np.median(a)), 'median_b': float(np.median(b)),
            'wasserstein_raw': float(wasserstein_distance(a, b)), 'ks_distance': float(ks_2samp(a, b).statistic),
            'histogram_overlap_10_bins': overlap, 'probability_b_above_a_with_half_ties': superiority,
            'cliffs_delta_b_minus_a': 2 * superiority - 1}


def analyze(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['rpm'], row['date'], row['condition'])].append(row)
        if row['label'] in ('Normal', 'Looseness'):
            groups[(row['rpm'], row['date'], row['label'] + 'Combined')].append(row)
    overlap, shifts, sensitivity = [], [], []
    rpms = sorted({r['rpm'] for r in rows})
    dates = sorted({r['date'] for r in rows})
    loose = ('Looseness0.05', 'Looseness0.1', 'LoosenessFull', 'LoosenessCombined')
    for rpm in rpms:
        for j, feature in enumerate(FEATURES):
            for date in dates:
                for normal in ('Normal1', 'Normal2', 'NormalCombined'):
                    for condition in loose:
                        a, b = groups.get((rpm, date, normal), []), groups.get((rpm, date, condition), [])
                        if a and b:
                            overlap.append({'rpm': rpm, 'date': date, 'normal': normal, 'looseness': condition,
                                            'feature': feature, **contrast([r['values'][j] for r in a], [r['values'][j] for r in b])})
            for condition in sorted({k[2] for k in groups if k[0] == rpm}):
                for first, second in combinations(dates, 2):
                    a, b = groups.get((rpm, first, condition), []), groups.get((rpm, second, condition), [])
                    if a and b:
                        shifts.append({'rpm': rpm, 'condition': condition, 'date_a': first, 'date_b': second,
                                       'feature': feature, **contrast([r['values'][j] for r in a], [r['values'][j] for r in b])})
            for condition in loose:
                d = [r['wasserstein_raw'] for r in shifts if r['rpm'] == rpm and r['feature'] == feature
                     and r['condition'] in ('Normal1', 'Normal2', condition) and r['status'] == 'descriptive_only']
                f = [r['wasserstein_raw'] for r in overlap if r['rpm'] == rpm and r['feature'] == feature
                     and r['normal'] == 'NormalCombined' and r['looseness'] == condition and r['status'] == 'descriptive_only']
                dm, fm = float(np.median(d)) if d else None, float(np.median(f)) if f else None
                sensitivity.append({'rpm': rpm, 'feature': feature, 'looseness': condition,
                                    'date_pairs_n': len(d), 'fault_contrasts_n': len(f),
                                    'median_within_condition_date_distance': dm, 'median_same_date_fault_distance': fm,
                                    'date_to_fault_distance_ratio': finite_ratio(dm, fm),
                                    'date_distance_larger': dm > fm if dm is not None and fm is not None else None})
    return overlap, shifts, sensitivity


def overlap_plots(output, overlap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rpm in sorted({r['rpm'] for r in overlap}):
        dates = sorted({r['date'] for r in overlap if r['rpm'] == rpm})
        matrix = np.full((len(FEATURES), len(dates)), np.nan)
        for j, feature in enumerate(FEATURES):
            for k, date in enumerate(dates):
                records = [r for r in overlap if r['rpm'] == rpm and r['date'] == date and r['feature'] == feature
                           and r['normal'] == 'NormalCombined' and r['looseness'] == 'LoosenessCombined'
                           and 'histogram_overlap_10_bins' in r]
                if records:
                    matrix[j, k] = records[0]['histogram_overlap_10_bins']
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, vmin=0, vmax=1, aspect='auto', cmap='viridis')
        ax.set(xticks=range(len(dates)), xticklabels=dates, yticks=range(len(FEATURES)), yticklabels=FEATURES,
               title=f'{rpm}RPM | Normal/Looseness histogram overlap\n10 pair-specific bins; small-sample descriptive index')
        fig.colorbar(image, ax=ax, label='0: separated bins / 1: same bin proportions')
        fig.tight_layout()
        fig.savefig(output / f'normal_looseness_overlap_{rpm}rpm.png', dpi=130)
        plt.close(fig)


def run(eda_dir, output):
    rows, excluded, source = load_eda(eda_dir)
    with new_report(output, [eda_dir], source['input_sha256'], 'date_shift.json', provenance=source) as (out, report):
        summaries = distributions(rows, out)
        overlap, shifts, sensitivity = analyze(rows)
        write_csv(out / 'normal_looseness_overlap.csv', overlap)
        write_csv(out / 'within_condition_date_shift.csv', shifts)
        write_csv(out / 'date_vs_fault_distance.csv', sensitivity)
        write_csv(out / 'excluded_files.csv', excluded)
        plots(out, summaries, [])
        overlap_plots(out, overlap)
        report.update(status='descriptive_analysis_only', validated_files=len(rows), excluded_files=len(excluded),
                      overlap_rows=len(overlap), date_shift_rows=len(shifts),
                      definitions={'distance': '1D Wasserstein in raw feature units',
                                   'overlap': 'sum min proportions in 10 equal-width bins fitted to each pair; not a classifier probability',
                                   'date_vs_fault': 'median same-condition date distances for Normal1, Normal2 and the chosen Looseness subtype vs median same-date NormalCombined/subtype distance; no causal claim',
                                   'zero_fault_distance': 'ratio undefined, not infinity; inspect both raw distances'},
                      pooled_groups='NormalCombined/LoosenessCombined are file-weighted within RPM/date; also report each original condition')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        out = args.output or default_output('date-shift')
        result = run(args.eda_dir, out)
        print(f"{result['status']}; output={out}")
        return 0 if result['completed'] and result['validated_files'] else 2
    except Exception as error:
        print(f'NOT ANALYZED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
