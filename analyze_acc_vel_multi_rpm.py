from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, detrend, welch

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

hp_sos = butter(
    4, 5,
    btype="highpass",
    fs=FS,
    output="sos"
)

def get_peaks(signal, targets):
    freq, psd = welch(
        signal,
        fs=FS,
        window="hann",
        nperseg=65536,
        noverlap=32768
    )

    asd = np.sqrt(psd)
    peaks = []

    for target in targets:
        mask = np.abs(freq - target) <= 1.0
        peaks.append(float(np.max(asd[mask])))

    return peaks

rows = []

for rpm in RPMS:
    folder = DATA_ROOT / f"20260919_{rpm}RPM"
    rotation_hz = rpm / 60
    targets = [rotation_hz * i for i in [1, 2, 3]]

    print(f"\n===== {rpm}RPM 분석 시작 =====")
    print(f"1X={targets[0]:.0f}Hz, "
          f"2X={targets[1]:.0f}Hz, "
          f"3X={targets[2]:.0f}Hz")

    for name in CLASSES:
        acc_values = []
        vel_values = []
        failed = 0

        files = sorted((folder / name).rglob("*.mat"))

        for file in files:
            try:
                mat = loadmat(file)
                x = np.asarray(
                    mat["signals"][2, :],
                    dtype=np.float64
                )
                fs = float(np.asarray(mat["fs"]).squeeze())

                if x.size != 65536 or fs != FS:
                    raise ValueError("신호 길이 또는 fs 불일치")

                if not np.isfinite(x).all():
                    raise ValueError("원본 NaN/Inf")

                acc = detrend(x)
                acc_hp = sosfiltfilt(hp_sos, acc)

                vel = np.zeros_like(acc_hp)
                vel[1:] = np.cumsum(
                    (acc_hp[:-1] + acc_hp[1:]) / (2 * FS)
                )

                vel = sosfiltfilt(
                    hp_sos, detrend(vel)
                )

                acc_peaks = get_peaks(acc, targets)
                vel_peaks = get_peaks(vel, targets)

                if not np.isfinite(
                    acc_peaks + vel_peaks
                ).all():
                    raise ValueError("스펙트럼 NaN/Inf")

                acc_values.append(acc_peaks)
                vel_values.append(vel_peaks)

            except Exception as e:
                failed += 1
                print(f"[제외] {rpm}RPM {name}/{file.name}: {e}")

        if not acc_values:
            print(f"[실패] {rpm}RPM {name}")
            continue

        acc_med = np.median(acc_values, axis=0)
        vel_med = np.median(vel_values, axis=0)

        rows.append([
            rpm,
            name,
            len(acc_values),
            failed,
            *acc_med,
            *vel_med
        ])

        print(
            f"[완료] {name:15s} "
            f"성공={len(acc_values):2d}, "
            f"실패={failed}, "
            f"Acc 1X={acc_med[0]:.6f}"
        )

csv_path = OUT / "multi_rpm_acc_velocity_peaks.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "RPM", "Class",
        "Success_Count", "Failed_Count",
        "Acc_1X_ASD", "Acc_2X_ASD", "Acc_3X_ASD",
        "Vel_1X_ASD", "Vel_2X_ASD", "Vel_3X_ASD"
    ])
    writer.writerows(rows)

print(f"\n전체 분석 완료: {csv_path}")
