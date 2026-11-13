# 보관된 초기 실험 코드: compare_20hz_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

conditions = ["Normal1", "Normal2", "Unbalance_-1", "Unbalance_-2"]

split_path = project_dir / "outputs_clean" / "1200rpm" / "first" / "split_1200rpm.csv"

with split_path.open(encoding="utf-8-sig", newline="") as f:
    rows = [
        row
        for row in csv.DictReader(f)
        if row["condition"] in conditions
        and row["split"] in ("test_normal", "test_anomaly")
    ]

fs = 25600.0
n = 65536
window = np.hanning(n)
freq = np.fft.rfftfreq(n, d=1 / fs)
band = (freq >= 18) & (freq <= 22)

features = defaultdict(list)
records = []

for row in rows:
    data = loadmat(project_dir / row["file"])
    signals = data["signals"].copy()

    if signals.shape != (6, n) or float(data["fs"].item()) != fs:
        raise ValueError(f"데이터 형식 확인 필요: {row['file']}")

    if not np.isfinite(signals).all() or (np.abs(signals) > 1e100).any():
        raise ValueError(f"데이터 값 확인 필요: {row['file']}")

    signals -= signals.mean(axis=1, keepdims=True)

    amplitude = np.abs(np.fft.rfft(signals * window, axis=1)) / window.sum()

    amplitude[:, 1:-1] *= 2

    # 채널마다 18~22Hz 안의 최대 진폭 추출
    peaks = amplitude[:, band].max(axis=1)
    features[row["condition"]].append(peaks)

    record = {"file": row["file"], "condition": row["condition"]}

    for ch in range(6):
        record[f"CH{ch + 1}_peak"] = float(peaks[ch])

    records.append(record)

for condition in conditions:
    if not features[condition]:
        raise ValueError(f"{condition}: 비교할 파일 없음")

    features[condition] = np.stack(features[condition])

# 파일별 특징값 저장
csv_path = output_dir / "peak_20hz_features_1200rpm.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

# 채널별 분포 그림
fig, axes = plt.subplots(3, 2, figsize=(13, 11))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
rng = np.random.default_rng(42)

for ch, ax in enumerate(axes.flat):
    for position, (condition, color) in enumerate(zip(conditions, colors), start=1):
        values = features[condition][:, ch]
        x = position + rng.uniform(-0.15, 0.15, len(values))

        ax.scatter(x, values, color=color, s=22, alpha=0.65)

        median = np.median(values)
        ax.hlines(median, position - 0.22, position + 0.22, color="black", linewidth=2)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(conditions, rotation=20)
    ax.set_title(f"CH {ch + 1}")
    ax.set_ylabel("Peak amplitude, 18–22 Hz")
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "1200RPM - Per-file peak amplitude near 20 Hz\n"
    "Each dot: one file; black line: median"
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

figure_path = output_dir / "peak_20hz_distribution_1200rpm.png"
fig.savefig(figure_path, dpi=160)

print("상태별 채널 진폭 중앙값")

for condition in conditions:
    medians = np.median(features[condition], axis=0)
    print(f"\n{condition}: {len(features[condition])}개 파일")
    print(" / ".join(f"CH{ch + 1}: {value:.6f}" for ch, value in enumerate(medians)))

print("\n특징값과 그래프 저장 완료!")
plt.show()
