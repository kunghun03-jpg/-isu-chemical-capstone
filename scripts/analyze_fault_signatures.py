"""Read-only, date/RPM-stratified exploratory vibration analysis; no diagnosis."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import struct
import sys
import zlib

import numpy as np
import scipy
from scipy.io import whosmat
from scipy.io.matlab import MatReadError
from scipy.signal import butter, find_peaks, hilbert, periodogram, sosfiltfilt, sosfreqz

from vibration.web_inference import MAX_UPLOAD_BYTES, preview_mat

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('Normal1', 'Normal2', 'IR', 'OR', 'Roller', 'Looseness0.05',
              'Looseness0.1', 'LoosenessFull', 'Unbalance_-1', 'Unbalance_-2')
NORMAL = ('Normal1', 'Normal2')
BEARING = ('IR', 'OR', 'Roller')
RPMS = (900, 1200, 1800)
BANDS = {'low': (0., 200.), 'mid': (200., 2000.), 'high': (2000., 10000.)}
FEATURES = ('rms_ac', 'peak_ac', 'crest_factor_ac', 'one_x_amplitude', 'two_x_amplitude',
            'energy_low', 'energy_mid', 'energy_high', 'envelope_rms_ac')


def metadata(path):
    """Require exact known condition/Vib folders and mutually consistent dates/RPM."""
    path = Path(path)
    rpm_parts = [re.fullmatch(r'(\d{8})_(\d+)RPM', part, re.I) for part in path.parts]
    rpm_parts = [m for m in rpm_parts if m]
    if len(rpm_parts) != 1:
        raise ValueError('missing_or_ambiguous_date_RPM_folder')
    date, rpm = rpm_parts[0].group(1), int(rpm_parts[0].group(2))
    datetime.strptime(date, '%Y%m%d')
    if rpm not in RPMS:
        raise ValueError('unsupported_RPM')
    if path.parent.name != 'Vib' or path.parent.parent.name not in CONDITIONS:
        raise ValueError('not_exact_target_condition_Vib_folder')
    if sum(part in CONDITIONS for part in path.parts) != 1:
        raise ValueError('ambiguous_condition_folders')
    match = re.fullmatch(r'(\d{8})_(\d{6})_Vib\.mat', path.name, re.I)
    if not match:
        raise ValueError('unrecognized_measurement_filename')
    datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S')
    dates = {p for p in path.parts if re.fullmatch(r'\d{8}', p)} | {match.group(1), date}
    if len(dates) != 1:
        raise ValueError('conflicting_dates')
    return {'rpm': rpm, 'date': date, 'condition': path.parent.parent.name,
            'family': ('Normal' if path.parent.parent.name in NORMAL else
                       'Bearing' if path.parent.parent.name in BEARING else
                       'Looseness' if path.parent.parent.name.startswith('Looseness') else 'Unbalance'),
            'run_id': None, 'independence': 'Run independence unconfirmed'}


def discover(roots, count=3):
    if count < 1:
        raise ValueError('files-per-stratum must be positive')
    records, groups, seen = [], defaultdict(list), set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'Data root does not exist: {root}')
        for entry in sorted(root.rglob('*')):
            if entry.suffix.lower() != '.mat' or not entry.is_file():
                continue
            path = entry.resolve()
            if path in seen:
                continue
            seen.add(path)
            record = {'path': str(path), 'status': 'excluded', 'reason': None}
            try:
                # Metadata uses the discovered path; resolving a symlink must not change labels.
                record.update(metadata(entry))
                if metadata(path) != metadata(entry):
                    raise ValueError('symlink_metadata_conflict')
                record['status'] = 'not_selected'
                record['reason'] = 'deterministic_sampling_limit'
                groups[(record['rpm'], record['date'], record['condition'])].append(record)
            except ValueError as error:
                record['reason'] = str(error)
            records.append(record)
    for group in groups.values():
        group.sort(key=lambda row: row['path'])
        indices = np.linspace(0, len(group) - 1, min(count, len(group)), dtype=int)
        for i in indices:
            group[i].update(status='selected', reason=None)
    return records


def fft_amplitude(signal, fs):
    x = np.asarray(signal, dtype=np.float64)
    window = np.hanning(x.size)
    amplitude = np.abs(np.fft.rfft((x - x.mean()) * window)) / window.sum()
    amplitude[1:-1 if x.size % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(x.size, 1 / fs), amplitude


def peak_component(frequency, amplitude, target, tolerance):
    mask = np.abs(frequency - target) <= tolerance
    if not mask.any():
        raise ValueError('No FFT bin in order search window')
    index = np.flatnonzero(mask)[np.argmax(amplitude[mask])]
    return float(frequency[index]), float(amplitude[index])


def top_peaks(frequency, amplitude, maximum=10, upper=10000.):
    indices, _ = find_peaks(amplitude)
    indices = indices[(frequency[indices] > 0) & (frequency[indices] <= upper)]
    indices = sorted(indices, key=lambda i: (-amplitude[i], frequency[i]))[:maximum]
    return [{'rank': rank + 1, 'frequency_hz': float(frequency[i]), 'amplitude': float(amplitude[i])}
            for rank, i in enumerate(indices)]


def finite_arrays(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError('nonfinite_intermediate_or_feature')


def analyze_signal(signal, fs, rpm, envelope_band=(2000., 8000.), order_tolerance=1., trim_seconds=.1):
    """EDA math only: whole-record FFT/PSD, never the AE's segment score."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 32 or fs <= 0 or rpm not in RPMS:
        raise ValueError('invalid_signal_fs_or_RPM')
    finite_arrays(x, fs, envelope_band, order_tolerance, trim_seconds)
    low, high = envelope_band
    if not (0 < low < high < fs / 2):
        raise ValueError('Envelope carrier band must be strictly inside (0, Nyquist)')
    if order_tolerance < fs / x.size or order_tolerance >= rpm / 120:
        raise ValueError('Order tolerance must cover >=1 FFT bin and keep 1X/2X windows disjoint')
    trim = int(round(trim_seconds * fs))
    if trim < 1 or 2 * trim >= x.size - 32:
        raise ValueError('invalid_envelope_edge_trim')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        centered = x - x.mean()
        rms = float(np.sqrt(np.mean(centered ** 2)))
        peak = float(np.max(np.abs(centered)))
        frequency, amplitude = fft_amplitude(x, fs)
        psd_frequency, psd = periodogram(centered, fs=fs, window=np.hanning(x.size),
                                        detrend=False, scaling='density')
        df = fs / x.size
        features = {'mean_raw': float(x.mean()), 'rms_raw': float(np.sqrt(np.mean(x ** 2))),
                    'peak_raw': float(np.max(np.abs(x))),
                    'rms_ac': rms, 'peak_ac': peak,
                    'crest_factor_ac': peak / rms if rms > 0 else None,
                    'fft_resolution_hz': df, 'one_x_hz': rpm / 60., 'two_x_hz': rpm / 30.,
                    'order_search_tolerance_hz': order_tolerance,
                    'order_search_effective_halfwidth_hz': order_tolerance + df / 2}
        for label, multiple in (('one_x', 1), ('two_x', 2)):
            f, a = peak_component(frequency, amplitude, multiple * rpm / 60., order_tolerance)
            features[label + '_peak_hz'], features[label + '_amplitude'] = f, a
        for name, (a, b) in BANDS.items():
            mask = (psd_frequency >= a) & ((psd_frequency <= b) if name == 'high' else (psd_frequency < b))
            features['energy_' + name] = float(np.sum(psd[mask]) * df)
        sos = butter(4, (low, high), btype='bandpass', fs=fs, output='sos')
        filtered = sosfiltfilt(sos, centered)
        # Hilbert on full filtered record; trim afterwards to reduce edge artifacts.
        envelope = np.abs(hilbert(filtered))[trim:-trim]
        efrequency, eamplitude = fft_amplitude(envelope, fs)
        features['envelope_rms_ac'] = float(np.sqrt(np.mean((envelope - envelope.mean()) ** 2)))
        carrier = (psd_frequency >= low) & (psd_frequency <= high)
        total_power = float(np.sum(psd) * df)
        features['carrier_power_fraction'] = float(np.sum(psd[carrier]) * df / total_power) if total_power > 0 else None
        features['envelope_resolution_hz'] = fs / envelope.size
        response_f, response = sosfreqz(sos, worN=512, fs=fs)
        # filtfilt's amplitude response is |H| squared.
        finite_arrays(centered, frequency, amplitude, psd, filtered, envelope, efrequency, eamplitude,
                      response, [v for v in features.values() if v is not None])
    mask, emask = frequency <= 10000, efrequency <= 1000
    return {'features': features, 'signal': x, 'frequency': frequency[mask], 'amplitude': amplitude[mask],
            'psd': psd[mask], 'envelope_frequency': efrequency[emask], 'envelope_amplitude': eamplitude[emask],
            'peaks': top_peaks(frequency, amplitude),
            'envelope_peaks': top_peaks(efrequency, eamplitude, upper=1000),
            'filter_frequency': response_f, 'filter_amplitude_response': np.abs(response) ** 2}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_output(path, roots):
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Fresh output directory required: {output}')
    if output.is_relative_to(ROOT) or any((p / '.git/HEAD').is_file() or (p / '.git').is_file()
                                          for p in (output, *output.parents)):
        raise ValueError('Output must be outside any Git checkout')
    for root in roots:
        root = Path(root).expanduser().resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Output and data roots must be separate directory trees')
    return output


def plot_groups(output, groups, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for (rpm, date), strata in sorted(groups.items()):
        folder = output / f'{rpm}rpm' / date
        folder.mkdir(parents=True)
        combined = [item for condition in NORMAL for item in strata.get(condition, [])]
        display = dict(strata)
        if combined:
            display['NormalCombined'] = combined
        comparisons = []
        for condition, items in display.items():
            target = folder / condition
            target.mkdir()
            # Representative is the selected successfully analyzed median-RMS file.
            ordered = sorted(items, key=lambda item: (item['features']['rms_ac'], item['path']))
            representative = ordered[(len(ordered) - 1) // 2]
            frequency = representative['frequency']
            amplitudes = np.stack([i['amplitude'] for i in items])
            spectra = {'mean': amplitudes.mean(axis=0), 'p10': np.percentile(amplitudes, 10, axis=0),
                       'median': np.median(amplitudes, axis=0), 'p90': np.percentile(amplitudes, 90, axis=0)}
            envelope = np.stack([i['envelope_amplitude'] for i in items])
            envelope_mean = envelope.mean(axis=0)
            envelope_p10, envelope_p90 = np.percentile(envelope, [10, 90], axis=0)
            write_csv(target / 'spectrum_distribution.csv',
                      [{'frequency_hz': float(f), **{k: float(v[j]) for k, v in spectra.items()},
                        'mean_psd': float(v)} for j, (f, v) in
                       enumerate(zip(frequency, np.mean([i['psd'] for i in items], axis=0)))])
            ef = representative['envelope_frequency']
            write_csv(target / 'envelope_distribution.csv',
                      [{'frequency_hz': float(f), 'mean': float(envelope_mean[j]),
                        'p10': float(envelope_p10[j]),
                        'p90': float(envelope_p90[j])} for j, f in enumerate(ef)])
            summary = {'rpm': rpm, 'date': date, 'condition': condition, 'n': len(items),
                       'normal1_n': sum(i['condition'] == 'Normal1' for i in items),
                       'normal2_n': sum(i['condition'] == 'Normal2' for i in items),
                       'representative_file': representative['path']}
            for feature in FEATURES:
                values = [i['features'][feature] for i in items if i['features'][feature] is not None]
                for statistic, fn in (('mean', np.mean), ('median', np.median),
                                      ('p10', lambda x: np.percentile(x, 10)),
                                      ('p90', lambda x: np.percentile(x, 90))):
                    summary[feature + '_' + statistic] = float(fn(values)) if values else None
            comparisons.append(summary)
            fig, axes = plt.subplots(3, 1, figsize=(11, 9))
            x, fs = representative['signal'], representative['fs']
            axes[0].plot(np.arange(x.size) / fs, x, linewidth=.5)
            axes[0].set(xlabel='Time (s)', ylabel='Signal (raw units)',
                        title=f'Representative: {Path(representative["path"]).name}')
            axes[1].plot(frequency, spectra['mean'], label=f'Mean n={len(items)}')
            axes[1].fill_between(frequency, spectra['p10'], spectra['p90'], alpha=.25, label='P10-P90 (not CI)')
            axes[1].set(xlabel='Frequency (Hz)', ylabel='Peak amplitude (raw units)')
            axes[1].legend()
            axes[2].plot(ef, envelope.mean(axis=0))
            axes[2].set(xlabel='Envelope frequency (Hz)', ylabel='Envelope amplitude (raw units)',
                        title='Exploratory envelope; carrier band not empirically validated')
            fig.suptitle(f'{rpm} RPM | {date} | {condition} | CH3; no fault diagnosis')
            fig.tight_layout()
            fig.savefig(target / 'waveform_spectra.png', dpi=130)
            plt.close(fig)
            representative['representative_for'].append(condition)
        write_csv(folder / 'condition_summary.csv', comparisons)
        rows.extend(comparisons)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        for condition, items in strata.items():
            f = items[0]['frequency']
            mean = np.mean([i['amplitude'] for i in items], axis=0)
            axes[0].plot(f, mean, label=f'{condition} (n={len(items)})', linewidth=.7)
            axes[1].plot(f, mean, label=condition, linewidth=.7)
        axes[0].set(xlim=(0, 200), xlabel='Frequency (Hz)', ylabel='Mean peak amplitude (raw units)')
        axes[1].set(xlim=(0, 10000), xlabel='Frequency (Hz)', ylabel='Mean peak amplitude (raw units)')
        axes[0].axvline(rpm / 60., color='gray', linestyle='--', label='Nominal 1X')
        axes[0].axvline(rpm / 30., color='gray', linestyle=':', label='Nominal 2X')
        axes[0].legend(fontsize=7, ncol=4)
        fig.suptitle(f'{rpm} RPM | {date} | Conditions kept separate')
        fig.tight_layout()
        fig.savefig(folder / 'condition_spectra.png', dpi=130)
        plt.close(fig)
        fig, axes = plt.subplots(4, 2, figsize=(15, 13))
        for feature, ax in zip(FEATURES, axes.flat):
            labels, values = [], []
            for condition, items in display.items():
                v = [i['features'][feature] for i in items if i['features'][feature] is not None]
                if v:
                    labels.append(f'{condition}\nn={len(v)}')
                    values.append(v)
            if values:
                ax.boxplot(values, tick_labels=labels, showfliers=True)
                for j, v in enumerate(values, 1):
                    ax.scatter(np.full(len(v), j), v, s=8, alpha=.6)
            ax.tick_params(axis='x', rotation=70, labelsize=7)
            ax.set_title(feature)
        fig.suptitle(f'{rpm} RPM | {date} | File-level features; NormalCombined duplicates Normal1/2')
        fig.tight_layout()
        fig.savefig(folder / 'feature_distributions.png', dpi=130)
        plt.close(fig)


def run(roots, output, files_per_stratum=3, envelope_band=(2000., 8000.), order_tolerance=1.):
    roots = [Path(p).expanduser().resolve() for p in roots]
    output = validate_output(output, roots)
    records = discover(roots, files_per_stratum)
    # Invalid global settings must fail before creating a report directory.
    if not (0 < envelope_band[0] < envelope_band[1] < 12800):
        raise ValueError('Envelope band must lie strictly inside the validated 12.8kHz Nyquist limit')
    if not np.isfinite(order_tolerance) or not 25600 / 65536 <= order_tolerance < 7.5:
        raise ValueError('Order search tolerance must be >=0.390625Hz and <7.5Hz')
    output.mkdir(parents=True, exist_ok=False)
    groups, feature_rows, peak_rows = defaultdict(lambda: defaultdict(list)), [], []
    report = {'analysis': 'exploratory_only_no_diagnostic_rules', 'completed': False,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'data_roots': list(map(str, roots)),
              'software': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
              'config': {'rpms': RPMS, 'conditions': CONDITIONS, 'channel': 'CH3 (index 2)',
                         'loader': 'vibration.web_inference.preview_mat; fixed (6,65536), 25600Hz',
                         'selection': 'per RPM/date/exact condition; sorted paths, evenly spaced indices including endpoints; no error replacement',
                         'files_per_stratum': files_per_stratum, 'fft': 'full-record symmetric Hann; coherent-gain corrected one-sided peak amplitude; remove DC',
                         'order_search_tolerance_hz': order_tolerance,
                         'bands_hz': BANDS, 'band_edges': 'lower inclusive, upper exclusive; high upper inclusive',
                         'energy': 'sum(one-sided Hann density PSD bins)*df; raw units squared; no physical unit conversion',
                         'envelope_carrier_hz': envelope_band, 'envelope_filter': '4th order Butterworth SOS, sosfiltfilt',
                         'envelope_trim_seconds_each_edge': .1, 'envelope_plot_range_hz': [0, 1000],
                         'normal_pooling': 'within same RPM/date only, file weighted; Normal1/2 also separate'},
              'limitations': ['독립성 미확인 탐색 분석: Run IDs are unknown; adjacent files may not be independent.',
                             'Carrier band is a configurable exploratory hypothesis, not empirically validated.',
                             '1X/2X use nominal folder RPM, not a tachometer measurement; local peaks may overlap unrelated components.',
                             'Bearing geometry and fault frequencies are unknown. No IR/OR/Roller diagnosis from envelope peaks.',
                             'No physical units or sensor calibration inferred. No acceleration-to-velocity conversion.',
                             'Small stratified sample; distributions are descriptive, not classification performance.',
                             'No automatic selection of discriminating features and no fault rules.'],
              'files': records}
    try:
        for record in records:
            if record['status'] != 'selected':
                continue
            path = Path(record['path'])
            try:
                if path.stat().st_size > MAX_UPLOAD_BYTES:
                    raise ValueError('MAT exceeds existing loader 20MiB limit')
                raw = path.read_bytes()
                record['sha256_before'] = hashlib.sha256(raw).hexdigest()
                record['mat_variables'] = [{'name': n, 'shape': list(s), 'class': c} for n, s, c in whosmat(BytesIO(raw))]
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    loaded = preview_mat(raw)
                    record.update(fs=loaded['fs'], samples=loaded['samples'], duration_seconds=loaded['duration'],
                                  channels=6, analyzed_channel=3)
                    result = analyze_signal(loaded['signal'], loaded['fs'], record['rpm'], envelope_band, order_tolerance)
                record['sha256_after'] = hashlib.sha256(path.read_bytes()).hexdigest()
                if record['sha256_before'] != record['sha256_after']:
                    raise ValueError('MAT changed externally during analysis; excluded')
                record.update(status='analyzed', reason=None)
                result.update(path=str(path), condition=record['condition'], fs=loaded['fs'], representative_for=[])
                groups[(record['rpm'], record['date'])][record['condition']].append(result)
                feature_rows.append({**{k: record[k] for k in ('path', 'rpm', 'date', 'condition', 'family', 'fs', 'samples', 'duration_seconds')},
                                     **result['features']})
                for kind, peaks in (('FFT', result['peaks']), ('envelope', result['envelope_peaks'])):
                    peak_rows.extend({'path': str(path), 'rpm': record['rpm'], 'date': record['date'],
                                      'condition': record['condition'], 'spectrum': kind, **p} for p in peaks)
            except (ValueError, KeyError, OSError, FloatingPointError, OverflowError,
                    NotImplementedError, MatReadError, TypeError, IndexError, EOFError,
                    struct.error, zlib.error) as error:
                record.update(status='excluded', reason=f'{type(error).__name__}: {error}')
                if 'sha256_before' in record and path.is_file():
                    record['sha256_after'] = hashlib.sha256(path.read_bytes()).hexdigest()
        summaries = []
        if groups:
            plot_groups(output, groups, summaries)
        for (rpm, date), strata in groups.items():
            for condition, items in strata.items():
                for item in items:
                    next(r for r in records if r['path'] == item['path'])['representative_for'] = item['representative_for']
            folder = output / f'{rpm}rpm' / date
            write_csv(folder / 'file_features.csv', [r for r in feature_rows if r['rpm'] == rpm and r['date'] == date])
        write_csv(output / 'file_features.csv', feature_rows)
        write_csv(output / 'condition_summary.csv', summaries)
        write_csv(output / 'peaks.csv', peak_rows)
        write_csv(output / 'selection_and_exclusions.csv',
                  [{k: v for k, v in r.items() if k not in ('mat_variables', 'representative_for')} for r in records])
        report['counts'] = dict(Counter(r['status'] for r in records))
        report['counts_by_rpm'] = {str(rpm): dict(Counter(r['status'] for r in records if r.get('rpm') == rpm)) for rpm in RPMS}
        report['missing_strata_by_observed_date'] = [dict(rpm=rpm, date=date, condition=c) for date in sorted({r['date'] for r in records if 'date' in r})
                                                    for rpm in RPMS for c in CONDITIONS if not groups.get((rpm, date), {}).get(c)]
        report['completed'] = True
        report['status'] = 'exploratory_analysis_written' if feature_rows else 'no_valid_measurements_not_analyzed'
        report['analyzed_files'] = len(feature_rows)
        if groups:
            first = next(iter(next(iter(groups.values())).values()))[0]
            write_csv(output / 'envelope_filter_response.csv',
                      [{'frequency_hz': float(f), 'zero_phase_amplitude_gain': float(g)}
                       for f, g in zip(first['filter_frequency'], first['filter_amplitude_response'])])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--files-per-stratum', type=int, default=3)
    parser.add_argument('--envelope-band', nargs=2, type=float, default=(2000., 8000.), metavar=('LOW_HZ', 'HIGH_HZ'))
    parser.add_argument('--order-tolerance-hz', type=float, default=1.)
    args = parser.parse_args()
    output = args.output or Path.home() / 'fault-signatures' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        report = run(args.data_root, output, args.files_per_stratum, args.envelope_band, args.order_tolerance_hz)
    except Exception as error:
        print(f'Analysis failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'output': str(output)}, ensure_ascii=False))
    return 0 if report['analyzed_files'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
