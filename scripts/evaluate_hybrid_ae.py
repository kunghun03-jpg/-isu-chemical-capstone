"""저장된 추론 점수를 활용한 하이브리드 AE 탐색적 성능 평가."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from scripts.infer_hybrid_ae import HybridAEInference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

DEST = OUT / "hybrid_ae_exploratory_evaluation"

abs_path = (
    OUT / "integrated_ae_exploratory_evaluation"
    / "predictions.csv"
)

rel_path = (
    OUT / "relative_ae_exploratory_evaluation"
    / "predictions.csv"
)

absolute = pd.read_csv(abs_path)
relative = pd.read_csv(rel_path)

keys = ["file", "rpm", "condition", "ground_truth"]

required = set(keys + [
    "score", "threshold", "predicted_anomaly"
])

for frame in (absolute, relative):

    if not required.issubset(frame.columns):
        raise ValueError("필수 평가 열 누락")

    if len(frame) != 1484:
        raise ValueError("평가 파일 수 불일치")

    frame["file"] = frame["file"].map(
        lambda name: str((ROOT / name).resolve())
    )

    if frame["file"].duplicated().any():
        raise ValueError("중복 평가 파일")

    if not frame["ground_truth"].isin([0, 1]).all():
        raise ValueError("정답 레이블 오류")

    if not np.isfinite(
        frame[["score", "threshold"]].to_numpy()
    ).all():
        raise ValueError("비정상 평가 점수")

    if (
        (frame["score"] < 0).any()
        or (frame["threshold"] <= 0).any()
    ):
        raise ValueError("점수 또는 임계값 오류")

    expected = (
        frame["score"] > frame["threshold"]
    ).astype(int)

    if not np.array_equal(
        expected, frame["predicted_anomaly"]
    ):
        raise ValueError("기존 판정 결과 불일치")

if set(absolute["file"]) != set(relative["file"]):
    raise ValueError("두 모델의 평가 파일 불일치")

merged = absolute.merge(
    relative,
    on=keys,
    suffixes=("_abs", "_rel"),
    validate="one_to_one",
)

if len(merged) != 1484:
    raise ValueError("파일별 평가 조건 불일치")

if merged["ground_truth"].value_counts().to_dict() != {
    0: 301, 1: 1183
}:
    raise ValueError("정상·결함 파일 수 불일치")

if set(merged["rpm"]) != {900, 1200, 1800}:
    raise ValueError("RPM 구성 불일치")

print("===== 평가 파일 검증 완료 =====")
print("정상: 301개")
print("결함: 1183개")
print("전체: 1484개")
print("두 모델의 파일 및 평가 조건 일치")

results = []
summary = []
defects = []


def calculate_metrics(y, scores, pred):

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    return {
        "AUROC": roc_auc_score(y, scores),
        "AUPRC": average_precision_score(y, scores),
        "recall": recall,
        "fpr": fpr,
        "fnr": 1 - recall,
        "specificity": 1 - fpr,
        "balanced_accuracy": (
            recall + 1 - fpr
        ) / 2,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


for rpm in (900, 1200, 1800):

    model = HybridAEInference(rpm)

    group = merged[
        merged["rpm"] == rpm
    ].copy()

    np.testing.assert_allclose(
        group["threshold_abs"],
        model.abs_threshold,
        rtol=1e-9,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        group["threshold_rel"],
        model.rel_threshold,
        rtol=1e-9,
        atol=1e-12,
    )

    group["hybrid_score"] = (
        model.weight_abs
        * group["score_abs"] / model.abs_threshold
        + model.weight_rel
        * group["score_rel"] / model.rel_threshold
    )

    group["hybrid_threshold"] = model.threshold

    group["hybrid_prediction"] = (
        group["hybrid_score"] > model.threshold
    ).astype(int)

    if not np.isfinite(group["hybrid_score"]).all():
        raise ValueError("하이브리드 점수 오류")

    group["evaluation_status"] = "exploratory_only"

    results.append(group)

    models = {
        "integrated_ae": (
            "score_abs", "predicted_anomaly_abs"
        ),
        "relative_ae": (
            "score_rel", "predicted_anomaly_rel"
        ),
        "hybrid_ae": (
            "hybrid_score", "hybrid_prediction"
        ),
    }

    print(f"\n===== {rpm} RPM 성능 비교 =====")

    for name, (score_col, pred_col) in models.items():

        metrics = calculate_metrics(
            group["ground_truth"],
            group[score_col],
            group[pred_col],
        )

        summary.append({
            "rpm": rpm,
            "model": name,
            **metrics,
            "evaluation_status": "exploratory_only",
        })

        print(f"\n[{name}]")
        print(f"AUROC: {metrics['AUROC']:.4f}")
        print(f"AUPRC: {metrics['AUPRC']:.4f}")
        print(f"이상 탐지율: {metrics['recall']:.2%}")
        print(f"정상 오탐률: {metrics['fpr']:.2%}")
        print(f"미탐률: {metrics['fnr']:.2%}")
        print(
            f"균형 정확도: "
            f"{metrics['balanced_accuracy']:.2%}"
        )

    abnormal = group[group["ground_truth"] == 1]

    for condition, items in abnormal.groupby("condition"):

        defects.append({
            "rpm": rpm,
            "condition": condition,
            "files": len(items),
            "integrated_recall":
                items["predicted_anomaly_abs"].mean(),
            "relative_recall":
                items["predicted_anomaly_rel"].mean(),
            "hybrid_recall":
                items["hybrid_prediction"].mean(),
        })

predictions = pd.concat(results, ignore_index=True)
summary = pd.DataFrame(summary)
defects = pd.DataFrame(defects)

DEST.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    DEST / "predictions.csv", index=False
)

summary.to_csv(
    DEST / "rpm_summary.csv", index=False
)

defects.to_csv(
    DEST / "defect_summary.csv", index=False
)

print("\n===== 전체 모델 성능 비교 =====")

for name, group in summary.groupby("model"):

    tp = group["TP"].sum()
    fn = group["FN"].sum()
    fp = group["FP"].sum()
    tn = group["TN"].sum()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(f"\n[{name}]")
    print(f"RPM 평균 AUROC: {group['AUROC'].mean():.4f}")
    print(f"RPM 평균 AUPRC: {group['AUPRC'].mean():.4f}")
    print(f"전체 이상 탐지율: {recall:.2%}")
    print(f"전체 정상 오탐률: {fpr:.2%}")
    print(f"전체 미탐률: {1-recall:.2%}")
    print(f"전체 균형 정확도: {(recall+1-fpr)/2:.2%}")
    print(f"오탐: {fp}/301")
    print(f"미탐: {fn}/1183")

print("\n===== 하이브리드 성능 평가 완료 =====")
print("전체 평가 파일:", len(predictions))
print("기존 추론 점수 재사용")
print("추가 MAT 추론 없음")
print("추가 모델 학습 없음")
print("가중치 및 임계값 변경 없음")
print("기존 Test 재사용에 따른 탐색적 평가")
print("독립적인 최종 성능 검증 아님")
print("저장 위치:", DEST)
