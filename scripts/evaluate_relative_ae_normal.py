"""상대 스펙트럼 AE의 정상 상태 변화 탐색 평가."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_relative_ae import RelativeAEInference

ROOT = Path(__file__).resolve().parents[1]

RPM = 900

SPLIT = (
    ROOT / "outputs/900rpm/date_prelim_20261106"
    / "split_900rpm.csv"
)

DEST = (
    ROOT / "outputs/experimental_relative_ae/900rpm"
    / "normal_shift_evaluation"
)

frame = pd.read_csv(SPLIT, dtype={"date": str})

selected = frame[
    (frame["date"] == "20260916")
    & frame["condition"].isin(["Normal1", "Normal2"])
    & (frame["rpm"] == RPM)
].copy()

expected = {
    "Normal1": 50,
    "Normal2": 58,
}

actual = selected["condition"].value_counts().to_dict()

print("===== 평가 데이터 확인 =====")
print(actual)

if actual != expected:
    raise ValueError(
        f"평가 대상 불일치: {actual}"
    )

paths = [
    (ROOT / name).resolve()
    for name in selected["file"]
]

if len(paths) != len(set(paths)):
    raise ValueError("중복 파일 발견")

for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)

model = RelativeAEInference(RPM)

print(f"임계값: {model.threshold:.9f}")

results = []

for index, (_, row) in enumerate(
    selected.iterrows(), 1
):
    result = model.predict(ROOT / row["file"])

    results.append({
        "file": row["file"],
        "date": row["date"],
        "rpm": RPM,
        "condition": row["condition"],
        "score": result["score"],
        "threshold": result["threshold"],
        "predicted_anomaly": result["predicted_anomaly"],
        "evaluation_status": "exploratory_only",
    })

    if index % 20 == 0 or index == len(selected):
        print(
            f"진행: {index}/{len(selected)}",
            flush=True,
        )

predictions = pd.DataFrame(results)

if not np.isfinite(
    predictions[["score", "threshold"]].to_numpy()
).all():
    raise ValueError("비정상 평가 수치")

summary = []

for condition, group in predictions.groupby("condition"):

    total = len(group)
    fp = int(group["predicted_anomaly"].sum())

    summary.append({
        "condition": condition,
        "files": total,
        "false_positives": fp,
        "false_positive_rate": fp / total,
        "median_score": group["score"].median(),
        "threshold": model.threshold,
    })

total = len(predictions)
fp = int(predictions["predicted_anomaly"].sum())

summary.append({
    "condition": "All",
    "files": total,
    "false_positives": fp,
    "false_positive_rate": fp / total,
    "median_score": predictions["score"].median(),
    "threshold": model.threshold,
})

summary = pd.DataFrame(summary)

DEST.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    DEST / "predictions.csv", index=False
)

summary.to_csv(
    DEST / "summary.csv", index=False
)

previous = {
    "Normal1": 0 / 50,
    "Normal2": 17 / 58,
    "All": 17 / 108,
}

print("\n===== 기존 통합 AE vs 상대 스펙트럼 AE =====")

for _, row in summary.iterrows():

    condition = row["condition"]
    old = previous[condition]
    new = row["false_positive_rate"]

    print(f"\n[{condition}]")
    print(f"기존 통합 AE 오탐률: {old:.2%}")
    print(f"상대 스펙트럼 AE 오탐률: {new:.2%}")
    print(
        f"오탐 파일: "
        f"{int(row['false_positives'])}/"
        f"{int(row['files'])}"
    )
    print(f"오탐률 변화: {(new-old)*100:+.2f}%p")
    print(f"점수 중앙값: {row['median_score']:.6f}")

print("\n===== 평가 완료 =====")
print("전체 평가 파일:", len(predictions))
print("저장 위치:", DEST)
print("모델 및 임계값 변경 없음")
print("9/19 Test 사용 없음")
print("정상 상태 변화에 대한 탐색적 평가")
