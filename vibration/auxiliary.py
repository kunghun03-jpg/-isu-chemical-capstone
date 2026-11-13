# 회전 주파수 특징이 정상 기준에서 벗어난 정도를 보조 점수로 계산합니다.

from pathlib import Path
import csv
import json
import numpy as np


def run(project_dir, output_dir, rpm):
    with (output_dir / "rotation_features.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    # 1. 파일별 CH3 회전 주파수 최대 진폭을 배열로 모읍니다.
    X = np.array(
        [[float(row["CH3_peak"])] for row in rows],
        dtype=np.float64,
    )
    splits = np.array([row["split"] for row in rows])
    conditions = np.array([row["condition"] for row in rows])
    if not np.isfinite(X).all():
        raise ValueError("특징값에 NaN 또는 무한대가 있습니다.")

    # 2. 정상 학습·검증 데이터와 평가 데이터를 구분합니다.
    train_mask = splits == "train"
    val_mask = splits == "validation"
    test_mask = np.isin(splits, ["test_normal", "test_anomaly"])
    if not train_mask.any() or not val_mask.any():
        raise ValueError("정상 학습·검증 데이터가 필요합니다.")

    # 3. 정상 학습 데이터로 채널별 평균·표준편차를 계산합니다.
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0)
    std = np.maximum(std, 1e-08)

    # 4. 채널별 표준화 편차의 제곱 평균을 보조 점수로 사용합니다.
    scores = np.mean(((X - mean) / std) ** 2, axis=1)

    # 정상 검증 점수의 95백분위수를 임계값으로 정합니다.
    threshold = float(np.percentile(scores[val_mask], 95))
    predictions = scores > threshold
    print(f"정상 학습 파일: {train_mask.sum()}개")
    print(f"정상 검증 파일: {val_mask.sum()}개")
    print(f"보조 점수 임계값: {threshold:.6f}")
    normal_test = splits == "test_normal"
    print(f"\n정상 평가 파일 오탐률: {predictions[normal_test].mean():.2%}")
    print("\n상태별 파일 단위 결과")
    for condition in sorted(np.unique(conditions[test_mask])):
        mask = test_mask & (conditions == condition)
        print(
            f"{condition}: {mask.sum()}개 파일, 이상 판정 비율 {predictions[mask].mean():.2%}, 점수 중앙값 {np.median(scores[mask]):.6f}"
        )

    # 5. 평가 결과와 이후 결합 평가에 사용할 정상 기준을 저장합니다.
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
    with (output_dir / f"evaluation_rotation_{rpm}rpm.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / f"reference_rotation_{rpm}rpm.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "band_hz": [rpm / 60 - 2, rpm / 60 + 2],
                "channels": [3],
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
    print("\n회전 주파수 보조 점수 평가 완료!")
