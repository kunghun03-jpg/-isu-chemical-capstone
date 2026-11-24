"""고정된 기존 Test 데이터를 이용한 통합 AE 탐색적 평가."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from scripts.infer_integrated_ae import predict

ROOT = Path("outputs")
DEST = ROOT / "integrated_ae_exploratory_evaluation"

EXPECTED = {900: (100, 398), 1200: (99, 393), 1800: (102, 392)}

datasets = {}

# 전체 평가 대상 사전 확인
for rpm in (900, 1200, 1800):
    split_path = (
        ROOT / f"{rpm}rpm"
        / "date_prelim_20261106"
        / f"split_{rpm}rpm.csv"
    )

    with split_path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    selected = [
        r for r in rows
        if r["split"] in ("test_normal", "test_anomaly")
    ]

    normal = sum(
        r["split"] == "test_normal" for r in selected
    )
    anomaly = sum(
        r["split"] == "test_anomaly" for r in selected
    )

    if (normal, anomaly) != EXPECTED[rpm]:
        raise ValueError(
            f"{rpm} RPM 평가 파일 수 불일치: "
            f"{normal}, {anomaly}"
        )

    paths = [Path(r["file"]).resolve() for r in selected]

    if len(paths) != len(set(paths)):
        raise ValueError("중복 평가 파일 발견")

    for row, path in zip(selected, paths):
        if row["date"] != "20260919":
            raise ValueError("평가 날짜 불일치")
        if not path.is_file():
            raise FileNotFoundError(path)

    datasets[rpm] = selected

DEST.mkdir(parents=True, exist_ok=True)

results = []

for rpm, rows in datasets.items():
    print(f"\n===== {rpm} RPM 평가 시작 =====", flush=True)

    for index, row in enumerate(rows, 1):
        result = predict(row["file"], rpm)

        results.append({
            "file": row["file"],
            "rpm": rpm,
            "condition": row["condition"],
            "ground_truth": int(
                row["split"] == "test_anomaly"
            ),
            "score": result["score"],
            "threshold": result["threshold"],
            "predicted_anomaly": result["predicted_anomaly"],
            "evaluation_status": "exploratory_only",
        })

        if index % 50 == 0 or index == len(rows):
            print(
                f"{rpm} RPM: {index}/{len(rows)}",
                flush=True,
            )

frame = pd.DataFrame(results)

if len(frame) != 1484:
    raise ValueError("전체 평가 파일 수 불일치")

if not np.isfinite(
    frame[["score", "threshold"]].to_numpy()
).all():
    raise ValueError("비정상 수치 발견")

frame.to_csv(
    DEST / "predictions.csv",
    index=False,
)

summary = []
defect_summary = []

for rpm, group in frame.groupby("rpm"):
    y = group["ground_truth"].to_numpy()
    score = group["score"].to_numpy()
    pred = group["predicted_anomaly"].to_numpy()

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    summary.append({
        "rpm": rpm,
        "model": "integrated_ae",
        "normal_files": tn + fp,
        "anomaly_files": tp + fn,
        "AUROC": roc_auc_score(y, score),
        "AUPRC": average_precision_score(y, score),
        "recall": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "evaluation_status": "exploratory_only",
    })

    abnormal = group[group["ground_truth"] == 1]

    for condition, items in abnormal.groupby("condition"):
        defect_summary.append({
            "rpm": rpm,
            "condition": condition,
            "files": len(items),
            "recall": items["predicted_anomaly"].mean(),
        })

summary = pd.DataFrame(summary)
summary.to_csv(DEST / "rpm_summary.csv", index=False)

pd.DataFrame(defect_summary).to_csv(
    DEST / "defect_summary.csv",
    index=False,
)

print("\n===== 통합 AE 성능 평가 결과 =====")

for _, row in summary.iterrows():
    print(f"\n[{int(row['rpm'])} RPM]")
    print(f"AUROC: {row['AUROC']:.4f}")
    print(f"AUPRC: {row['AUPRC']:.4f}")
    print(f"이상 탐지율: {row['recall']:.1%}")
    print(f"정상 오탐률: {row['fpr']:.1%}")
    print(f"미탐률: {row['fnr']:.1%}")

print("\n===== 평가 완료 =====")
print("총 평가 파일:", len(frame))
print("저장 위치:", DEST)
print("모델 및 임계값 변경 없음")
print("기존 Test 재사용에 따른 탐색적 평가")
print("독립적인 최종 성능 검증 아님")
