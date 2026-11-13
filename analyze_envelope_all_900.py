from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, welch, detrend

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919/20260919_900RPM"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal1",
    "Normal2",
    "Normal3",
    "IR",
    "OR",
    "Roller",
    "Looeness0.05",
    "Looeness0.1",
    "LooenessFull",
    "Unbalance_-1",
    "Unbalance_-2",
]

LOWCUT = 1800
HIGHCUT = 2800

# 파일마다 동일한 처리 조건 적용
FS = 25600
sos = butter(
    4,
    [LOWCUT, HIGHCUT],
    btype="bandpass",
    fs=FS,
    output="sos"
)

rms_results = {}
spectrum_results = {}
summary = []

for name in CLASSES:
    folder = DATA / name
    files = sorted(folder.rglob("*.mat"))

    if not files:
        print(f"[건너뜀] {name}: MAT 파일 없음")
        continue

    rms_values = []
    spectra = []
    failed = 0

    for file in files:
        try:
            mat = loadmat(file)
            signals = np.asarray(mat["signals"])
            fs = float(np.asarray(mat["fs"]).squeeze())

            if signals.shape != (6, 65536):
                raise ValueError("신호 크기 오류")

            if fs != FS:
                raise ValueError("샘플링 주파수 오류")

            acceleration = detrend(
                np.asarray(signals[2, :], dtype=np.float64)
            )

            if not np.isfinite(acceleration).all():
                raise ValueError("비정상 신호값")

            filtered = sosfiltfilt(sos, acceleration)
            envelope = np.abs(hilbert(filtered))

            # RMS는 포락선의 평균을 제거한 AC 성분 기준
            envelope_ac = envelope - np.mean(envelope)
            rms = np.sqrt(np.mean(envelope_ac ** 2))

            freq, psd = welch(
                envelope_ac,
                fs=fs,
                window="hann",
                nperseg=16384,
                noverlap=8192
            )

            spectrum = np.sqrt(psd)

            # NaN 또는 Inf가 포함된 결과는 통계에서 제외
            if not np.isfinite(rms) or not np.isfinite(spectrum).all():
                raise ValueError("포락 분석 결과에 NaN/Inf 발생")

            rms_values.append(rms)
            spectra.append(spectrum)

        except Exception as e:
            failed += 1
            print(f"[오류] {file.name}: {e}")

    if not rms_values:
        print(f"[실패] {name}: 유효한 파일 없음")
        continue

    rms_values = np.asarray(rms_values)
    spectra = np.asarray(spectra)

    rms_results[name] = rms_values
    spectrum_results[name] = np.mean(spectra, axis=0)

    summary.append([
        name,
        len(rms_values),
        failed,
        float(np.median(rms_values)),
        float(np.mean(rms_values)),
        float(np.std(rms_values)),
    ])

    print(
        f"[완료] {name}: "
        f"{len(rms_values)}개 성공, "
        f"{failed}개 실패, "
        f"RMS 중앙값={np.median(rms_values):.6f}"
    )

if not rms_results:
    raise RuntimeError("분석에 성공한 파일이 없습니다.")

# 1. 상태별 포락 RMS 분포
fig, ax = plt.subplots(figsize=(12, 6))

names = list(rms_results.keys())
values = [rms_results[name] for name in names]

ax.boxplot(values, tick_labels=names, showfliers=True)
ax.set_title("900 RPM - Envelope AC RMS Distribution")
ax.set_ylabel("Envelope AC RMS")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUT / "900rpm_envelope_rms_boxplot.png",
    dpi=180
)
plt.close(fig)

# 2. 상태별 평균 포락 스펙트럼
fig, ax = plt.subplots(figsize=(13, 6))

mask = freq <= 500

for name, spectrum in spectrum_results.items():
    ax.plot(
        freq[mask],
        spectrum[mask],
        label=name,
        linewidth=1
    )

ax.set_title(
    "900 RPM - Mean Envelope Spectrum "
    "(1800-2800 Hz Bandpass)"
)
ax.set_xlabel("Modulation Frequency (Hz)")
ax.set_ylabel("Mean Envelope ASD")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "900rpm_envelope_mean_spectrum.png",
    dpi=180
)
plt.close(fig)

# 3. 수치 요약 CSV
csv_path = OUT / "900rpm_envelope_summary.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Class",
        "Success_Count",
        "Failed_Count",
        "Envelope_AC_RMS_Median",
        "Envelope_AC_RMS_Mean",
        "Envelope_AC_RMS_Std",
    ])
    writer.writerows(summary)

print("\n전체 파일 분석 완료!")
print(OUT / "900rpm_envelope_rms_boxplot.png")
print(OUT / "900rpm_envelope_mean_spectrum.png")
print(csv_path)
