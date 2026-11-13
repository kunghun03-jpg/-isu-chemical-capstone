from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, hilbert, detrend, welch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/20260919/20260919_900RPM"
OUT = ROOT / "outputs/feedback1_signal_analysis"
OUT.mkdir(parents=True, exist_ok=True)

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

rows = []

for name in CLASSES:
    spectra = []
    failed = 0

    for file in sorted((DATA / name).rglob("*.mat")):
        try:
            mat = loadmat(file)
            x = np.asarray(mat["signals"][2, :], dtype=float)

            if not np.isfinite(x).all():
                raise ValueError("원본 NaN/Inf")

            filtered = sosfiltfilt(sos, detrend(x))
            envelope = np.abs(hilbert(filtered))
            envelope -= envelope.mean()

            freq, psd = welch(
                envelope,
                fs=FS,
                nperseg=16384,
                noverlap=8192
            )

            asd = np.sqrt(psd)

            if not np.isfinite(asd).all():
                raise ValueError("스펙트럼 NaN/Inf")

            spectra.append(asd)

        except Exception as e:
            failed += 1
            print(f"[제외] {name}/{file.name}: {e}")

    if not spectra:
        continue

    mean_asd = np.mean(spectra, axis=0)

    peaks = []
    for target in [15, 30, 45]:
        mask = np.abs(freq - target) <= 2

        # 목표 주파수 ±2Hz에서 최대 ASD
        peaks.append(float(np.max(mean_asd[mask])))

    rows.append([name, len(spectra), failed, *peaks])

    print(
        f"{name:15s} "
        f"1X={peaks[0]:.6f} "
        f"2X={peaks[1]:.6f} "
        f"3X={peaks[2]:.6f}"
    )

path = OUT / "900rpm_envelope_1x_2x_3x.csv"

with path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Class", "Success_Count", "Failed_Count",
        "Envelope_1X_ASD", "Envelope_2X_ASD", "Envelope_3X_ASD"
    ])
    writer.writerows(rows)

print(f"\n저장 완료: {path}")
