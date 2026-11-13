from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, detrend, welch

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data/20260919"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

RPMS = [900, 1200, 1800]

CLASSES = [
    "Normal1", "Normal2", "Normal3",
    "IR", "OR", "Roller",
    "Looeness0.05", "Looeness0.1", "LooenessFull",
    "Unbalance_-1", "Unbalance_-2"
]

FS = 25600
sos = butter(
    4, [1800, 2800],
    btype="bandpass",
    fs=FS,
    output="sos"
)

summary = []
peak_rows = []

for rpm in RPMS:
    print(f"\n===== {rpm}RPM 포락 분석 시작 =====")

    folder = DATA_ROOT / f"20260919_{rpm}RPM"
    rms_results = {}
    spectrum_results = {}

    for name in CLASSES:
        rms_values = []
        spectra = []
        failed = 0

        files = sorted((folder / name).rglob("*.mat"))

        for file in files:
            try:
                mat = loadmat(file)
                signals = np.asarray(mat["signals"])
                fs = float(np.asarray(mat["fs"]).squeeze())

                if signals.shape != (6, 65536) or fs != FS:
                    raise ValueError("신호 크기 또는 fs 오류")

                x = detrend(
                    np.asarray(signals[2, :], dtype=np.float64)
                )

                if not np.isfinite(x).all():
                    raise ValueError("원본 NaN/Inf")

                filtered = sosfiltfilt(sos, x)
                envelope = np.abs(hilbert(filtered))
                envelope_ac = envelope - envelope.mean()

                rms = float(np.sqrt(np.mean(envelope_ac ** 2)))

                freq, psd = welch(
                    envelope_ac,
                    fs=FS,
                    window="hann",
                    nperseg=16384,
                    noverlap=8192
                )

                asd = np.sqrt(psd)

                if not np.isfinite(rms) or not np.isfinite(asd).all():
                    raise ValueError("포락 계산 NaN/Inf")

                rms_values.append(rms)
                spectra.append(asd)

            except Exception as e:
                failed += 1
                print(f"[제외] {rpm}RPM {name}/{file.name}: {e}")

        if not rms_values:
            print(f"[실패] {rpm}RPM {name}")
            continue

        rms_values = np.asarray(rms_values)
        mean_asd = np.mean(spectra, axis=0)

        rms_results[name] = rms_values
        spectrum_results[name] = mean_asd

        summary.append([
            rpm, name, len(rms_values), failed,
            float(np.median(rms_values)),
            float(np.mean(rms_values)),
            float(np.std(rms_values))
        ])

        # RPM별 회전 주파수의 1X, 2X, 3X
        rotation_hz = rpm / 60
        peaks = []

        for harmonic in [1, 2, 3]:
            target = rotation_hz * harmonic
            mask = np.abs(freq - target) <= 2
            peaks.append(float(np.max(mean_asd[mask])))

        peak_rows.append([
            rpm, name, len(rms_values), failed, *peaks
        ])

        print(
            f"[완료] {name:15s} "
            f"성공={len(rms_values):2d}, 실패={failed}, "
            f"RMS 중앙값={np.median(rms_values):.6f}"
        )

    # RPM별 포락 RMS 박스플롯
    fig, ax = plt.subplots(figsize=(15, 6))
    names = list(rms_results.keys())

    ax.boxplot(
        [rms_results[n] for n in names],
        tick_labels=names,
        showfliers=True
    )

    ax.set_title(f"{rpm} RPM - Envelope AC RMS Distribution")
    ax.set_ylabel("Envelope AC RMS")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUT / f"{rpm}rpm_envelope_rms_boxplot.png",
        dpi=180
    )
    plt.close(fig)

    # RPM별 평균 포락 스펙트럼
    fig, ax = plt.subplots(figsize=(14, 6))
    mask = freq <= 500

    for name, spectrum in spectrum_results.items():
        ax.plot(
            freq[mask],
            spectrum[mask],
            label=name,
            linewidth=1
        )

    ax.set_title(
        f"{rpm} RPM - Mean Envelope Spectrum (1800-2800 Hz)"
    )
    ax.set_xlabel("Modulation Frequency (Hz)")
    ax.set_ylabel("Mean Envelope ASD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(
        OUT / f"{rpm}rpm_envelope_mean_spectrum.png",
        dpi=180
    )
    plt.close(fig)

# 전체 RMS 요약
summary_path = OUT / "multi_rpm_envelope_summary.csv"

with summary_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "RPM", "Class", "Success_Count", "Failed_Count",
        "Envelope_AC_RMS_Median",
        "Envelope_AC_RMS_Mean",
        "Envelope_AC_RMS_Std"
    ])
    writer.writerows(summary)

# 전체 1X/2X/3X 포락 피크 요약
peak_path = OUT / "multi_rpm_envelope_peaks.csv"

with peak_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "RPM", "Class", "Success_Count", "Failed_Count",
        "Envelope_1X_ASD",
        "Envelope_2X_ASD",
        "Envelope_3X_ASD"
    ])
    writer.writerows(peak_rows)

print("\n전체 RPM 포락 분석 완료!")
print(summary_path)
print(peak_path)
