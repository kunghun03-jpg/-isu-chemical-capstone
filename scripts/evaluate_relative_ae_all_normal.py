"""전체 RPM 상대 스펙트럼 AE 정상 상태 변화 탐색 평가."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_relative_ae import RelativeAEInference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experimental_relative_ae"

EXPECTED = {900: {'Normal1': 50, 'Normal2': 58}, 1200: {'Normal1': 48, 'Normal2': 46}, 1800: {'Normal1': 52, 'Normal2': 51}}

PREVIOUS_FP = {
    900: {"Normal1": 0, "Normal2": 17},
    1200: {"Normal1": 0, "Normal2": 0},
    1800: {"Normal1": 0, "Normal2": 51},
}

all_results = []

for rpm in (900, 1200, 1800):

    folder = BASE / f"{rpm}rpm"

    split = (
        ROOT / "outputs" / f"{rpm}rpm"
        / "date_prelim_20261106"
        / f"split_{rpm}rpm.csv"
    )

    frame = pd.read_csv(split, dtype={"date": str})

    selected = frame[
        (frame["date"] == "20260916")
        & frame["condition"].isin(["Normal1", "Normal2"])
        & (frame["rpm"] == rpm)
    ].copy()

    actual = selected["condition"].value_counts().to_dict()

    if actual != EXPECTED[rpm]:
        raise ValueError(
            f"{rpm} RPM 평가 대상 불일치: {actual}"
        )

    paths = [
        (ROOT / name).resolve()
        for name in selected["file"]
    ]

    if len(paths) != len(set(paths)):
        raise ValueError("중복 평가 파일 발견")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"\n===== {rpm} RPM =====", flush=True)

    model = RelativeAEInference(rpm)

    # 저장된 학습 점수와 실제 추론 점수 확인
    validation = pd.read_csv(
        folder / "validation_scores.csv"
    )

    for condition in ("Normal1", "Normal2"):

        row = validation[
            validation["condition"] == condition
        ].iloc[0]

        result = model.predict(ROOT / row["file"])

        np.testing.assert_allclose(
            result["score"],
            float(row["score"]),
            rtol=1e-5,
            atol=1e-7,
        )

        assert result["predicted_anomaly"] == int(
            row["predicted_anomaly"]
        )

    print("학습·추론 일치 검증: 통과")

    dest = folder / "normal_shift_evaluation"
    dest.mkdir(parents=True, exist_ok=True)

    if rpm == 900:

        # 이미 완료된 900 RPM 결과 재사용
        predictions = pd.read_csv(
            dest / "predictions.csv",
            dtype={"date": str},
        )

        if len(predictions) != len(selected):
            raise ValueError("900 RPM 결과 개수 불일치")

        if predictions["file"].duplicated().any():
            raise ValueError("900 RPM 중복 결과")

        if set(predictions["file"]) != set(selected["file"]):
            raise ValueError("900 RPM 평가 파일 불일치")

        if not np.allclose(
            predictions["threshold"],
            model.threshold,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("900 RPM 임계값 불일치")

        print("기존 900 RPM 평가 결과 재사용")

    else:

        results = []

        for index, (_, row) in enumerate(
            selected.iterrows(), 1
        ):

            result = model.predict(ROOT / row["file"])

            results.append({
                "file": row["file"],
                "date": row["date"],
                "rpm": rpm,
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

    expected_decisions = (
        predictions["score"] > model.threshold
    ).astype(int)

    if not np.array_equal(
        expected_decisions,
        predictions["predicted_anomaly"],
    ):
        raise ValueError("이상 판정 불일치")

    predictions.to_csv(
        dest / "predictions.csv", index=False
    )

    all_results.append(predictions)

    print(f"임계값: {model.threshold:.9f}")

    for condition, group in predictions.groupby("condition"):

        total = len(group)
        old_fp = PREVIOUS_FP[rpm][condition]
        new_fp = int(group["predicted_anomaly"].sum())

        print(f"\n[{condition}]")
        print(f"기존 통합 AE: {old_fp}/{total} ({old_fp/total:.2%})")
        print(f"상대 스펙트럼 AE: {new_fp}/{total} ({new_fp/total:.2%})")
        print(f"오탐률 변화: {(new_fp-old_fp)/total*100:+.2f}%p")

combined = pd.concat(all_results, ignore_index=True)

if len(combined) != 305:
    raise ValueError("전체 평가 파일 수 불일치")

summary = []

print("\n===== 전체 RPM 성능 비교 =====")

for rpm, group in combined.groupby("rpm"):

    total = len(group)
    old_fp = sum(PREVIOUS_FP[rpm].values())
    new_fp = int(group["predicted_anomaly"].sum())

    summary.append({
        "rpm": rpm,
        "normal_files": total,
        "integrated_ae_fp": old_fp,
        "relative_ae_fp": new_fp,
        "integrated_ae_fpr": old_fp / total,
        "relative_ae_fpr": new_fp / total,
        "fpr_change_pp": (new_fp-old_fp)/total*100,
    })

    print(f"\n[{rpm} RPM]")
    print(f"기존 통합 AE: {old_fp}/{total} ({old_fp/total:.2%})")
    print(f"상대 스펙트럼 AE: {new_fp}/{total} ({new_fp/total:.2%})")
    print(f"변화: {(new_fp-old_fp)/total*100:+.2f}%p")

total = len(combined)
old_fp = 68
new_fp = int(combined["predicted_anomaly"].sum())

print("\n[전체]")
print(f"기존 통합 AE: {old_fp}/{total} ({old_fp/total:.2%})")
print(f"상대 스펙트럼 AE: {new_fp}/{total} ({new_fp/total:.2%})")
print(f"전체 오탐률 변화: {(new_fp-old_fp)/total*100:+.2f}%p")

dest = BASE / "all_normal_shift_evaluation"
dest.mkdir(parents=True, exist_ok=True)

combined.to_csv(dest / "predictions.csv", index=False)

pd.DataFrame(summary).to_csv(
    dest / "rpm_summary.csv", index=False
)

print("\n===== 전체 정상 데이터 평가 완료 =====")
print("평가 파일:", total)
print("모델 및 임계값 변경 없음")
print("9/19 Test 사용 없음")
print("저장 위치:", dest)
