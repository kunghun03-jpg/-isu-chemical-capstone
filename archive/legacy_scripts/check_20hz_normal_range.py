# 보관된 초기 실험 코드: check_20hz_normal_range. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

split_path = project_dir / "outputs_clean" / "1200rpm" / "first" / "split_1200rpm.csv"

conditions = ("Normal1", "Normal2", "Unbalance_-1", "Unbalance_-2")

with split_path.open(encoding="utf-8-sig", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["condition"] in conditions]

fs = 25600.0
n = 65536
window = np.hanning(n)
freq = np.fft.rfftfreq(n, d=1 / fs)
band = (freq >= 18) & (freq <= 22)

groups = defaultdict(list)
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
    peaks = amplitude[:, band].max(axis=1)

    # 정상은 학습·검증·평가를 따로 표시
    if row["condition"] in ("Normal1", "Normal2"):
        group = f"{row['condition']}_{row['split']}"
    else:
        group = row["condition"]

    groups[group].append(peaks)

    record = {"file": row["file"], "condition": row["condition"], "split": row["split"]}

    for ch in range(6):
        record[f"CH{ch + 1}_peak"] = float(peaks[ch])

    records.append(record)

labels = [
    "Normal1_train",
    "Normal1_validation",
    "Normal1_test_normal",
    "Normal2_train",
    "Normal2_validation",
    "Normal2_test_normal",
    "Unbalance_-1",
    "Unbalance_-2",
]

for label in labels:
    if not groups[label]:
        raise ValueError(f"{label}: 비교할 파일 없음")
    groups[label] = np.stack(groups[label])

# 추출한 특징값 저장
with (output_dir / "peak_20hz_all_splits.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

# 그룹별 중앙값 출력
for label in labels:
    values = groups[label]
    medians = np.median(values, axis=0)

    print(f"\n{label}: {len(values)}개 파일")
    print(" / ".join(f"CH{ch + 1}: {value:.6f}" for ch, value in enumerate(medians)))

# 채널별 파일 분포
fig, axes = plt.subplots(3, 2, figsize=(16, 13))
rng = np.random.default_rng(42)

colors = [
    "tab:blue",
    "tab:blue",
    "tab:blue",
    "tab:orange",
    "tab:orange",
    "tab:orange",
    "tab:green",
    "tab:red",
]

short_labels = [
    "N1 train",
    "N1 val",
    "N1 test",
    "N2 train",
    "N2 val",
    "N2 test",
    "Unb -1",
    "Unb -2",
]

for ch, ax in enumerate(axes.flat):
    for position, (label, color) in enumerate(zip(labels, colors), start=1):
        values = groups[label][:, ch]
        x = position + rng.uniform(-0.15, 0.15, len(values))

        ax.scatter(x, values, color=color, s=18, alpha=0.6)
        ax.hlines(
            np.median(values),
            position - 0.22,
            position + 0.22,
            color="black",
            linewidth=2,
        )

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_title(f"CH {ch + 1}")
    ax.set_ylabel("Peak amplitude, 18–22 Hz")
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle(
    "1200RPM - Peak amplitude near 20 Hz by data split\n"
    "Each dot: one file; black line: median"
)
fig.tight_layout(rect=(0, 0, 1, 0.94))

fig.savefig(output_dir / "peak_20hz_all_splits.png", dpi=160)

print("\n특징값과 그래프 저장 완료!")
plt.show()
