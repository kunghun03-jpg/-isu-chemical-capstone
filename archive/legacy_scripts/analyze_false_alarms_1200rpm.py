# 보관된 초기 실험 코드: analyze_false_alarms_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import json
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

# 정상 데이터로 정한 보조 점수 기준
with (output_dir / "reference_20hz_1200rpm.json").open(encoding="utf-8") as f:
    reference = json.load(f)

channels = reference["channels"]
mean = np.array(reference["mean"], dtype=np.float64)
std = np.array(reference["std"], dtype=np.float64)
threshold = float(reference["threshold"])

# 파일별 20Hz 주변 진폭
with (output_dir / "peak_20hz_all_splits.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    feature_rows = list(csv.DictReader(f))

features = {row["file"]: row for row in feature_rows}

# 결합 평가 결과
with (output_dir / "evaluation_combined_1200rpm.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    evaluation = list(csv.DictReader(f))

normal_rows = [row for row in evaluation if row["split"] == "test_normal"]

details = []
false_alarm_count = 0

for row in normal_rows:
    if int(row["combined_alarm"]) != 1:
        continue

    false_alarm_count += 1
    feature = features[row["file"]]

    peaks = np.array([float(feature[f"CH{ch}_peak"]) for ch in channels])

    z = (peaks - mean) / std

    # 전체 점수는 채널별 기여도의 합
    contributions = z**2 / len(channels)
    score = float(contributions.sum())

    if not np.isclose(score, float(row["aux_score"])):
        raise ValueError(f"저장된 점수와 불일치: {row['file']}")

    print(f"\n파일: {Path(row['file']).name}")
    print(f"오토엔코더 점수: {float(row['ae_score']):.6f}")
    print(f"보조 점수: {score:.6f} / 임계값: {threshold:.6f}")
    print("채널별 기여도 — 큰 순서")

    for index in np.argsort(contributions)[::-1]:
        share = contributions[index] / score * 100 if score > 0 else 0

        print(
            f"  CH{channels[index]}: "
            f"진폭 {peaks[index]:.6f}, "
            f"정상 평균 대비 {z[index]:+.2f} 표준편차, "
            f"기여 비율 {share:.1f}%"
        )

        details.append(
            {
                "file": row["file"],
                "condition": row["condition"],
                "channel": channels[index],
                "peak_amplitude": float(peaks[index]),
                "train_mean": float(mean[index]),
                "train_std": float(std[index]),
                "standardized_deviation": float(z[index]),
                "score_contribution": float(contributions[index]),
                "contribution_percent": float(share),
                "aux_score": score,
                "aux_threshold": threshold,
            }
        )

print(f"\n정상 평가 파일 {len(normal_rows)}개 중 " f"결합 오탐 {false_alarm_count}개")

report_path = output_dir / "false_alarm_channel_details_1200rpm.csv"

with report_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "condition",
            "channel",
            "peak_amplitude",
            "train_mean",
            "train_std",
            "standardized_deviation",
            "score_contribution",
            "contribution_percent",
            "aux_score",
            "aux_threshold",
        ],
    )
    writer.writeheader()
    writer.writerows(details)

print("분석 결과 저장:", report_path)
