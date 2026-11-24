"""하이브리드 AE 정상 상태 변화 탐색 평가."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_hybrid_ae import HybridAEInference
from scripts.infer_integrated_ae import predict as predict_absolute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SOURCE = (
    OUT / "experimental_relative_ae"
    / "all_normal_shift_evaluation"
    / "predictions.csv"
)

DEST = (
    OUT / "experimental_hybrid_ae"
    / "normal_shift_evaluation"
)

EXPECTED = {
    900: {"Normal1": 50, "Normal2": 58},
    1200: {"Normal1": 48, "Normal2": 46},
    1800: {"Normal1": 52, "Normal2": 51},
}

PREVIOUS_FP = {
    900: {"Normal1": 0, "Normal2": 17},
    1200: {"Normal1": 0, "Normal2": 0},
    1800: {"Normal1": 0, "Normal2": 51},
}

relative = pd.read_csv(
    SOURCE, dtype={"date": str}
)

relative["file"] = relative["file"].map(
    lambda name: str((ROOT / name).resolve())
)

if len(relative) != 305:
    raise ValueError("평가 파일 수 불일치")

if relative["file"].duplicated().any():
    raise ValueError("중복 파일 발견")

if not (relative["date"] == "20260916").all():
    raise ValueError("평가 날짜 불일치")

if not relative["condition"].isin(
    ["Normal1", "Normal2"]
).all():
    raise ValueError("정상 조건 불일치")

if not np.isfinite(
    relative[["score", "threshold"]].to_numpy()
).all():
    raise ValueError("비정상 상대 AE 점수")

if (
    (relative["score"] < 0).any()
    or (relative["threshold"] <= 0).any()
):
    raise ValueError("점수 범위 오류")

if not np.array_equal(
    (relative["score"] > relative["threshold"]).astype(int),
    relative["predicted_anomaly"],
):
    raise ValueError("상대 AE 판정 불일치")

# 기존 분할 파일과 정확히 일치하는지 확인
for rpm in (900, 1200, 1800):

    split_path = (
        OUT / f"{rpm}rpm"
        / "date_prelim_20261106"
        / f"split_{rpm}rpm.csv"
    )

    split = pd.read_csv(
        split_path, dtype={"date": str}
    )

    selected = split[
        (split["date"] == "20260916")
        & (split["rpm"] == rpm)
        & split["condition"].isin(
            ["Normal1", "Normal2"]
        )
    ].copy()

    selected["file"] = selected["file"].map(
        lambda name: str((ROOT / name).resolve())
    )

    group = relative[relative["rpm"] == rpm]

    keys = ["file", "condition"]

    if (
        selected["file"].duplicated().any()
        or group["condition"].value_counts().to_dict()
        != EXPECTED[rpm]
        or set(map(tuple, selected[keys].to_numpy()))
        != set(map(tuple, group[keys].to_numpy()))
    ):
        raise ValueError(
            f"{rpm} RPM 평가 대상 불일치"
        )

    for name in group["file"]:
        if not Path(name).is_file():
            raise FileNotFoundError(name)

print("===== 평가 데이터 검증 완료 =====")
print("Normal1: 150개")
print("Normal2: 155개")
print("전체: 305개")

# 기존 통합 AE 점수 검색
cached = {}

candidate_paths = sorted(
    OUT.rglob("*.csv")
)

for path in candidate_paths:

    if (
        "experimental_relative_ae" in path.parts
        or "experimental_hybrid_ae" in path.parts
        or "integrated_ae" not in str(path)
    ):
        continue

    try:
        header = pd.read_csv(path, nrows=0)

        required = {
            "file", "rpm", "condition",
            "score", "threshold", "date",
        }

        if not required.issubset(header.columns):
            continue

        frame = pd.read_csv(
            path, dtype={"date": str}
        )

        frame = frame[
            (frame["date"] == "20260916")
            & frame["condition"].isin(
                ["Normal1", "Normal2"]
            )
        ]

        for _, row in frame.iterrows():

            key = (
                str((ROOT / row["file"]).resolve()),
                int(row["rpm"]),
                row["condition"],
            )

            # 출처가 확인된 점수만 참고하고
            # 실제 사용 전 추론 결과를 재확인
            cached[key] = (
                float(row["score"]),
                float(row["threshold"]),
            )

    except (ValueError, OSError, pd.errors.ParserError):
        continue

print("기존 통합 AE 점수 후보:", len(cached))

results = []

for rpm in (900, 1200, 1800):

    print(f"\n===== {rpm} RPM =====")

    model = HybridAEInference(rpm)

    group = relative[
        relative["rpm"] == rpm
    ].copy()

    np.testing.assert_allclose(
        group["threshold"],
        model.rel_threshold,
        rtol=1e-9,
        atol=1e-12,
    )

    # 저장 점수와 일치 여부 확인을 위해
    # 정상 조건별 1개씩 실제 상대 AE 추론
    for condition in ("Normal1", "Normal2"):

        row = group[
            group["condition"] == condition
        ].iloc[0]

        check = model.relative.predict(row["file"])

        np.testing.assert_allclose(
            check["score"],
            row["score"],
            rtol=1e-5,
            atol=1e-7,
        )

    # 통합 AE는 출처 및 모델 무결성이 확인된
    # 전용 캐시가 없으므로 실제 추론으로 확보
    for index, (_, row) in enumerate(
        group.iterrows(), 1
    ):

        absolute = predict_absolute(
            row["file"], rpm
        )

        np.testing.assert_allclose(
            absolute["threshold"],
            model.abs_threshold,
            rtol=1e-9,
            atol=1e-12,
        )

        abs_score = float(absolute["score"])
        rel_score = float(row["score"])

        score = (
            model.weight_abs
            * abs_score / model.abs_threshold
            + model.weight_rel
            * rel_score / model.rel_threshold
        )

        if not np.isfinite(score):
            raise ValueError("하이브리드 점수 오류")

        results.append({
            "file": row["file"],
            "rpm": rpm,
            "date": "20260916",
            "condition": row["condition"],
            "absolute_score": abs_score,
            "relative_score": rel_score,
            "hybrid_score": score,
            "hybrid_threshold": model.threshold,
            "integrated_prediction":
                int(abs_score > model.abs_threshold),
            "relative_prediction":
                int(row["predicted_anomaly"]),
            "hybrid_prediction":
                int(score > model.threshold),
            "evaluation_status": "exploratory_only",
        })

        if index % 20 == 0 or index == len(group):
            print(
                f"진행: {index}/{len(group)}",
                flush=True,
            )

predictions = pd.DataFrame(results)

if len(predictions) != 305:
    raise ValueError("전체 평가 결과 불일치")

summary = []

print("\n===== RPM·정상 조건별 오탐률 비교 =====")

for (rpm, condition), group in predictions.groupby(
    ["rpm", "condition"]
):

    total = len(group)

    abs_fp = int(group["integrated_prediction"].sum())
    rel_fp = int(group["relative_prediction"].sum())
    hybrid_fp = int(group["hybrid_prediction"].sum())

    if abs_fp != PREVIOUS_FP[rpm][condition]:
        raise ValueError(
            f"기존 통합 AE 결과 불일치: {rpm}, {condition}"
        )

    print(f"\n[{rpm} RPM / {condition}]")
    print(f"통합 AE: {abs_fp}/{total} ({abs_fp/total:.2%})")
    print(f"상대 AE: {rel_fp}/{total} ({rel_fp/total:.2%})")
    print(f"하이브리드 AE: {hybrid_fp}/{total} ({hybrid_fp/total:.2%})")

    summary.append({
        "rpm": rpm,
        "condition": condition,
        "files": total,
        "integrated_fp": abs_fp,
        "relative_fp": rel_fp,
        "hybrid_fp": hybrid_fp,
        "integrated_fpr": abs_fp / total,
        "relative_fpr": rel_fp / total,
        "hybrid_fpr": hybrid_fp / total,
    })

print("\n===== 전체 RPM 오탐률 비교 =====")

for rpm, group in predictions.groupby("rpm"):

    print(f"\n[{rpm} RPM]")

    for name in ("integrated", "relative", "hybrid"):

        fp = int(group[f"{name}_prediction"].sum())

        print(
            f"{name}: {fp}/{len(group)} "
            f"({fp/len(group):.2%})"
        )

print("\n[전체]")

for name in ("integrated", "relative", "hybrid"):

    fp = int(predictions[f"{name}_prediction"].sum())

    print(
        f"{name}: {fp}/305 ({fp/305:.2%})"
    )

DEST.mkdir(parents=True, exist_ok=True)

predictions.to_csv(
    DEST / "predictions.csv", index=False
)

pd.DataFrame(summary).to_csv(
    DEST / "condition_summary.csv", index=False
)

print("\n===== 하이브리드 정상 평가 완료 =====")
print("전체 평가 파일: 305개")
print("상대 AE 저장 점수 재사용")
print("추가 모델 학습 없음")
print("모델 및 임계값 변경 없음")
print("9/19 Test 사용 없음")
print("저장 위치:", DEST)
