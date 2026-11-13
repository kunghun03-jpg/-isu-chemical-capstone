# 보관된 초기 실험 코드: compare_low_frequency_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

conditions = ["Normal1", "Normal2", "Unbalance_-1", "Unbalance_-2"]

# 기존 평가에 사용한 파일 목록만 선택
split_path = project_dir / "outputs_clean" / "1200rpm" / "first" / "split_1200rpm.csv"

with split_path.open(encoding="utf-8-sig", newline="") as f:
    rows = [
        row
        for row in csv.DictReader(f)
        if row["condition"] in conditions
        and row["split"] in ("test_normal", "test_anomaly")
    ]

spectra = defaultdict(list)

fs = 25600.0
n = 65536
window = np.hanning(n)
freq = np.fft.rfftfreq(n, d=1 / fs)
mask = freq <= 100

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

    spectra[row["condition"]].append(amplitude[:, mask])

# 파일별 스펙트럼의 중앙값으로 대표 곡선 만들기
representatives = {}

for condition in conditions:
    if not spectra[condition]:
        raise ValueError(f"{condition}: 비교할 파일 없음")

    representatives[condition] = np.median(np.stack(spectra[condition]), axis=0)

    print(f"{condition}: {len(spectra[condition])}개 파일 비교")

fig, axes = plt.subplots(6, 1, figsize=(13, 13), sharex=True)

for ch, ax in enumerate(axes):
    for condition in conditions:
        ax.plot(
            freq[mask], representatives[condition][ch], label=condition, linewidth=1
        )

    ax.axvline(20, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel(f"CH {ch + 1}\nAmplitude")
    ax.grid(True, alpha=0.3)

axes[0].legend()
axes[-1].set_xlabel("Frequency (Hz)")
axes[-1].set_xlim(0, 100)

fig.suptitle(
    "1200RPM - Median low-frequency spectra\n"
    "Dashed line: nominal rotation frequency (20 Hz)"
)
fig.tight_layout(rect=(0, 0, 1, 0.95))

figure_path = output_dir / "low_frequency_comparison_1200rpm.png"
fig.savefig(figure_path, dpi=160)

print("\n그래프 저장:", figure_path)
plt.show()
