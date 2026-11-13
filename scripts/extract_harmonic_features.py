"""Read original MATs to extend a frozen EDA cohort; never infer harmonics from peak lists."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

from scripts.analyze_fault_signatures import analyze_signal, peak_component, write_csv
from scripts.evaluate_fault_classifier import FEATURES, digest, load_eda
from scripts.fault_feature_common import default_output, new_report
from vibration.web_inference import MAX_UPLOAD_BYTES, preview_mat

EXTRA_FEATURES = ('three_x_amplitude', 'four_x_amplitude', 'five_x_amplitude',
                  'two_to_one_ratio', 'three_to_one_ratio', 'four_to_one_ratio', 'five_to_one_ratio',
                  'harmonic_2to5_amplitude_sum', 'harmonic_2to5_amplitude_squared_sum',
                  'harmonic_sum_over_1x', 'harmonic_2to5_amplitude_fraction',
                  'energy_low_fraction', 'energy_mid_fraction', 'energy_high_fraction')
RATIO_POLICY = {'absolute_amplitude_floor': 1e-12, 'relative_to_largest_order_floor': 1e-6,
                'energy_total_floor': 1e-24, 'undefined': 'missing, never zero or infinity'}


def extended_features(result, rpm, tolerance):
    amplitudes, audit = [], {}
    for order in range(1, 6):
        f, a = peak_component(result['frequency'], result['amplitude'], order * rpm / 60., tolerance)
        amplitudes.append(a)
        audit[f'order_{order}_measured_peak_hz'] = f
    a = np.asarray(amplitudes, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Nonfinite/negative order amplitudes')
    floor = max(RATIO_POLICY['absolute_amplitude_floor'], float(a.max()) * RATIO_POLICY['relative_to_largest_order_floor'])
    valid = a[0] > floor
    values = dict(zip(EXTRA_FEATURES[:3], a[2:].tolist()))
    for key, value in zip(EXTRA_FEATURES[3:7], a[1:]):
        values[key] = float(value / a[0]) if valid else None
    total = float(a[1:].sum())
    values.update(harmonic_2to5_amplitude_sum=total,
                  harmonic_2to5_amplitude_squared_sum=float(np.sum(a[1:] ** 2)),
                  harmonic_sum_over_1x=total / float(a[0]) if valid else None,
                  harmonic_2to5_amplitude_fraction=total / float(a.sum()) if a.sum() > floor else None)
    energies = np.asarray([result['features']['energy_' + name] for name in ('low', 'mid', 'high')])
    if not np.isfinite(energies).all() or (energies < 0).any():
        raise ValueError('Nonfinite/negative band energies')
    energy_total = float(energies.sum())
    for name, value in zip(('low', 'mid', 'high'), energies):
        values['energy_' + name + '_fraction'] = float(value / energy_total) if energy_total > RATIO_POLICY['energy_total_floor'] else None
    if any(v is not None and (not np.isfinite(v) or not 0 <= v <= np.finfo(np.float32).max) for v in values.values()):
        raise ValueError('Nonfinite/out-of-range harmonic feature')
    return values, {**audit, 'one_x_ratio_denominator_floor': floor, 'one_x_ratios_defined': bool(valid)}


def run(eda_dir, output):
    rows, excluded, source = load_eda(eda_dir)
    config = source['eda_config']
    hashes = dict(source['input_sha256'])
    with new_report(output, [eda_dir], hashes, 'harmonic_extraction.json',
                    eda_input_sha256=source['input_sha256'], extra_features=list(EXTRA_FEATURES),
                    ratio_policy=RATIO_POLICY, baseline_features=list(FEATURES),
                    nominal_orders='Folder RPM / 60 multiplied by 1..5; not tachometer/order tracking',
                    eda_config=config, files=[]) as (out, report):
        values = []
        for row in rows:
            path = Path(row['path'])
            record = {'path': str(path), 'rpm': row['rpm'], 'date': row['date'], 'condition': row['condition'],
                      'status': 'excluded', 'mat_sha256_recorded': row['mat_sha256_recorded']}
            report['files'].append(record)
            try:
                if path.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError('MAT exceeds original loader size limit')
                raw = path.read_bytes()
                checksum = hashlib.sha256(raw).hexdigest()
                hashes[str(path)] = checksum
                record['mat_sha256_actual'] = checksum
                if checksum != row['mat_sha256_recorded']:
                    raise ValueError('Original MAT SHA256 differs from EDA record')
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    loaded = preview_mat(raw)
                    result = analyze_signal(loaded['signal'], loaded['fs'], row['rpm'],
                                            config['envelope_carrier_hz'], config['order_search_tolerance_hz'],
                                            config['envelope_trim_seconds_each_edge'])
                    baseline = [np.nan if result['features'][f] is None else result['features'][f] for f in FEATURES]
                    if not np.allclose(baseline, row['values'], rtol=1e-6, atol=1e-9, equal_nan=True):
                        record['baseline_recomputed'] = [None if np.isnan(v) else float(v) for v in baseline]
                        raise ValueError('Recomputed original nine features differ from frozen EDA')
                    extra, audit = extended_features(result, row['rpm'], config['order_search_tolerance_hz'])
                if digest(path) != checksum:
                    raise ValueError('MAT changed during extraction')
                # Preserve original baseline numbers, even if re-computation differs by last-bit tolerance.
                base_values = {f: None if np.isnan(v) else float(v) for f, v in zip(FEATURES, row['values'])}
                csv_row = {'path': row['path'], 'rpm': row['rpm'], 'date': row['date'], 'condition': row['condition'],
                           'label': row['label'], 'mat_sha256': checksum, **base_values, **extra}
                values.append(csv_row)
                record.update(status='extracted', baseline_matches=True, values={**base_values, **extra}, **audit)
            except Exception as error:
                record['reason'] = f'{type(error).__name__}: {error}'
        write_csv(out / 'extended_file_features.csv', values,
                  ['path', 'rpm', 'date', 'condition', 'label', 'mat_sha256', *FEATURES, *EXTRA_FEATURES])
        write_csv(out / 'extraction_exclusions.csv', [r for r in report['files'] if r['status'] != 'extracted'])
        write_csv(out / 'eda_validation_exclusions.csv', excluded)
        report.update(status='harmonic_extraction_complete' if values else 'no_MAT_features_extracted',
                      validated_eda_files=len(rows), extracted_files=len(values), excluded_files=len(rows) - len(values),
                      undefined_one_x_ratio_files=sum(r['status'] == 'extracted' and not r['one_x_ratios_defined'] for r in report['files']),
                      extended_csv_sha256=digest(out / 'extended_file_features.csv'))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    try:
        out = args.output or default_output('harmonics')
        report = run(args.eda_dir, out)
        print(f"{report['status']}; extracted={report['extracted_files']}; excluded={report['excluded_files']}; output={out}")
        return 0 if report['completed'] and report['extracted_files'] else 2
    except Exception as error:
        print(f'NOT EXTRACTED: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
