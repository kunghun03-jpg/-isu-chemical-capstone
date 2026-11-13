# 보관된 초기 실험 코드: evaluate_20hz_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import json
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

# 앞 단계에서 추출한 파일별 6채널 진폭 읽기
with (output_dir / "peak_20hz_all_splits.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

X = np.array(
    [[float(row[f"CH{ch}_peak"]) for ch in range(1, 7)] for row in rows],
    dtype=np.float64,
)

splits = np.array([row["split"] for row in rows])
conditions = np.array([row["condition"] for row in rows])

if not np.isfinite(X).all():
    raise ValueError("특징값에 NaN 또는 무한대가 있습니다.")

train_mask = splits == "train"
val_mask = splits == "validation"
test_mask = np.isin(splits, ["test_normal", "test_anomaly"])

if not train_mask.any() or not val_mask.any():
    raise ValueError("정상 학습·검증 데이터가 필요합니다.")

# 정상 학습 데이터만으로 채널별 기준 계산
mean = X[train_mask].mean(axis=0)
std = X[train_mask].std(axis=0)
std = np.maximum(std, 1e-8)

# 각 채널의 표준화된 차이를 제곱한 뒤 평균
scores = np.mean(((X - mean) / std) ** 2, axis=1)

# 정상 검증 데이터로만 임계값 설정
threshold = float(np.percentile(scores[val_mask], 95))
predictions = scores > threshold

print(f"정상 학습 파일: {train_mask.sum()}개")
print(f"정상 검증 파일: {val_mask.sum()}개")
print(f"보조 점수 임계값: {threshold:.6f}")

normal_test = splits == "test_normal"

print(f"\n정상 평가 파일 오탐률: " f"{predictions[normal_test].mean():.2%}")

print("\n상태별 파일 단위 결과")

for condition in sorted(np.unique(conditions[test_mask])):
    mask = test_mask & (conditions == condition)

    print(
        f"{condition}: {mask.sum()}개 파일, "
        f"이상 판정 비율 {predictions[mask].mean():.2%}, "
        f"점수 중앙값 {np.median(scores[mask]):.6f}"
    )

# 평가 파일별 점수와 판정 저장
results = []

for i in np.flatnonzero(test_mask):
    results.append(
        {
            "file": rows[i]["file"],
            "condition": rows[i]["condition"],
            "split": rows[i]["split"],
            "score": float(scores[i]),
            "threshold": threshold,
            "predicted_anomaly": int(predictions[i]),
        }
    )

with (output_dir / "evaluation_20hz_1200rpm.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# 이후 동일한 기준을 적용할 수 있도록 저장
with (output_dir / "reference_20hz_1200rpm.json").open("w", encoding="utf-8") as f:
    json.dump(
        {
            "band_hz": [18, 22],
            "channels": [1, 2, 3, 4, 5, 6],
            "mean": mean.tolist(),
            "std": std.tolist(),
            "threshold": threshold,
            "percentile": 95,
            "score_method": "mean_squared_standardized_deviation",
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print("\n20Hz 보조 점수 평가 완료!")
