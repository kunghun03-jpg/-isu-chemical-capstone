# 보관된 초기 실험 코드: compare_baseline. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "900rpm" / "first"


def load_data(split):
    with np.load(output_dir / f"{split}_normalized.npz") as data:
        return data["X"].copy(), data["condition"].copy()


# 정상 학습 데이터의 평균 스펙트럼
train_X, _ = load_data("train")
reference = train_X.mean(axis=0, dtype=np.float64)
del train_X


def calculate_scores(X):
    scores = []

    for start in range(0, len(X), 32):
        batch = X[start : start + 32].astype(np.float64)

        # 평균 정상 스펙트럼과의 차이
        error = np.mean((batch - reference) ** 2, axis=(1, 2))
        scores.append(error)

    scores = np.concatenate(scores)

    if not np.isfinite(scores).all():
        raise ValueError("점수에 NaN 또는 무한대 발생")

    return scores


# 이 방법도 정상 검증 데이터로 임계값 설정
val_X, _ = load_data("validation")
val_scores = calculate_scores(val_X)
threshold = float(np.percentile(val_scores, 95))
del val_X

print("비교 방법: 정상 평균 스펙트럼과의 거리")
print(f"임계값: {threshold:.6f}")

summary = []

for split in ("test_normal", "test_anomaly"):
    X, conditions = load_data(split)
    scores = calculate_scores(X)
    predicted = scores > threshold

    print(f"\n[{split}]")

    if split == "test_normal":
        print(f"정상 구간 오탐률: {predicted.mean():.2%}")
    else:
        print(f"이상 구간 탐지율: {predicted.mean():.2%}")

    for condition in sorted(np.unique(conditions)):
        mask = conditions == condition
        rate = float(predicted[mask].mean())
        median = float(np.median(scores[mask]))

        print(f"  {condition}: 이상 판정 비율 {rate:.2%}, " f"점수 중앙값 {median:.6f}")

        summary.append(
            {
                "split": split,
                "condition": str(condition),
                "segments": int(mask.sum()),
                "anomaly_rate": rate,
                "median_score": median,
                "threshold": threshold,
            }
        )

    del X

with (output_dir / "baseline_summary_900rpm.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

print("\n비교 결과 저장 완료!")
