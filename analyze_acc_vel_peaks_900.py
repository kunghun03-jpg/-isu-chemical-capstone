from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, detrend, welch

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
TARGETS = [15, 30, 45]

# 저주파 드리프트 제거용 필터
hp_sos = butter(
    4, 5,
    btype="highpass",
    fs=FS,
    output="sos"
)

def get_peak_values(signal):
    freq, psd = welch(
        signal,
        fs=FS,
        window="hann",
        nperseg=65536,
        noverlap=32768
    )

    asd = np.sqrt(psd)

    peaks = []
    for target in TARGETS:
        mask = np.abs(freq - target) <= 1.0
        peaks.append(float(np.max(asd[mask])))

    return peaks

rows = []

for name in CLASSES:
    acc_values = []
    vel_values = []
    failed = 0

    files = sorted((DATA / name).rglob("*.mat"))

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

            # 가속도: 추세 제거
            acc = detrend(x)

            # 속도: 가속도 고역통과 후 적분
            acc_hp = sosfiltfilt(hp_sos, acc)

            # 누적 사다리꼴 적분
            vel = np.zeros_like(acc_hp)
            vel[1:] = np.cumsum(
                (acc_hp[:-1] + acc_hp[1:]) / (2 * FS)
            )

            # 적분 후 드리프트 제거
            vel = sosfiltfilt(hp_sos, detrend(vel))

            acc_peaks = get_peak_values(acc)
            vel_peaks = get_peak_values(vel)

            all_values = acc_peaks + vel_peaks

            if not np.isfinite(all_values).all():
                raise ValueError("스펙트럼 NaN/Inf")

            acc_values.append(acc_peaks)
            vel_values.append(vel_peaks)

        except Exception as e:
            failed += 1
            print(f"[제외] {name}/{file.name}: {e}")

    if not acc_values:
        print(f"[실패] {name}: 유효한 파일 없음")
        continue

    acc_median = np.median(
        np.asarray(acc_values), axis=0
    )
    vel_median = np.median(
        np.asarray(vel_values), axis=0
    )

    rows.append([
        name,
        len(acc_values),
        failed,
        *acc_median,
        *vel_median
    ])

    print(
        f"[완료] {name:15s} "
        f"파일={len(acc_values):2d}, "
        f"가속도 1X={acc_median[0]:.6f}, "
        f"속도 1X={vel_median[0]:.8f}"
    )

csv_path = OUT / "900rpm_acc_velocity_1x_2x_3x.csv"

with csv_path.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "Class",
        "Success_Count",
        "Failed_Count",
        "Acc_1X_ASD",
        "Acc_2X_ASD",
        "Acc_3X_ASD",
        "Vel_1X_ASD",
        "Vel_2X_ASD",
        "Vel_3X_ASD"
    ])
    writer.writerows(rows)

print(f"\n분석 완료: {csv_path}")
