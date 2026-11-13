# 보관된 초기 실험 코드: analyze_results. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "900rpm" / "first"

with (output_dir / "evaluation_900rpm.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

with (output_dir / "threshold_900rpm.json").open(encoding="utf-8") as f:
    threshold = json.load(f)["threshold"]

# 상태별 점수와 정상 파일별 결과 묶기
condition_scores = defaultdict(list)
normal_files = defaultdict(list)

for row in rows:
    score = float(row["score"])
    condition_scores[row["condition"]].append(score)

    if row["split"] == "test_normal":
        normal_files[row["file"]].append(row)

print(f"임계값: {threshold:.6f}")
print("\n정상 파일 중 오탐이 발생한 파일")

false_alarm_files = []

for file, file_rows in sorted(normal_files.items()):
    flagged = [row for row in file_rows if int(row["predicted_anomaly"]) == 1]

    if not flagged:
        continue

    scores = [float(row["score"]) for row in file_rows]
    segment_numbers = [int(row["segment_id"]) + 1 for row in flagged]

    print(f"\n파일: {file}")
    print(f"  오탐 구간: {len(flagged)}/{len(file_rows)}개")
    print(f"  구간 번호: {segment_numbers}")
    print(f"  최대 점수: {max(scores):.6f}")

    false_alarm_files.append(
        {
            "file": file,
            "condition": file_rows[0]["condition"],
            "false_alarm_segments": len(flagged),
            "total_segments": len(file_rows),
            "max_score": max(scores),
        }
    )

print(f"\n오탐이 발생한 정상 파일: {len(false_alarm_files)}개")

with (output_dir / "false_alarm_files_900rpm.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "condition",
            "false_alarm_segments",
            "total_segments",
            "max_score",
        ],
    )
    writer.writeheader()
    writer.writerows(false_alarm_files)

# 그림 1: 모든 상태의 점수 분포
labels = sorted(condition_scores)
values = [np.array(condition_scores[label]) for label in labels]

fig, ax = plt.subplots(figsize=(13, 6))
ax.boxplot(values, tick_labels=labels, showfliers=True)
ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")

# 큰 점수와 작은 점수를 함께 보기 위한 로그 눈금
ax.set_yscale("symlog", linthresh=0.1)
ax.set_ylabel("Reconstruction error (MSE, symlog scale)")
ax.set_title("900RPM - Evaluation score distributions")
ax.tick_params(axis="x", rotation=35)
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "score_distribution_900rpm.png", dpi=160)

# 그림 2: 정상과 점수가 가까웠던 불평형 상태 확대
focus_labels = ["Normal1", "Normal2", "Unbalance_-1", "Unbalance_-2"]

fig, ax = plt.subplots(figsize=(10, 6))

for position, label in enumerate(focus_labels, start=1):
    scores = np.array(condition_scores[label])
    offsets = np.linspace(-0.18, 0.18, len(scores))

    ax.scatter(position + offsets, scores, s=12, alpha=0.45)

ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")
ax.set_xticks(range(1, len(focus_labels) + 1))
ax.set_xticklabels(focus_labels)
ax.set_ylabel("Reconstruction error (MSE)")
ax.set_title("900RPM - Normal and unbalance scores")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "normal_unbalance_scores_900rpm.png", dpi=160)

print("\n오탐 목록과 그래프 2개 저장 완료!")
plt.show()
