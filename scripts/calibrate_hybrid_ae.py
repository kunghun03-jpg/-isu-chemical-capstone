"""정상 Validation 기반 하이브리드 AE 임계값 설정."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experimental_hybrid_ae"

WEIGHT_ABS = 0.5
WEIGHT_REL = 0.5

EXPECTED = {
    900: 30,
    1200: 32,
    1800: 30,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_validation(path):

    frame = pd.read_csv(path)

    required = {
        "file", "condition", "score", "threshold"
    }

    if not required.issubset(frame.columns):
        raise ValueError(f"필수 열 누락: {path}")

    if frame["file"].duplicated().any():
        raise ValueError("중복 Validation 파일")

    if not frame["condition"].isin(
        ["Normal1", "Normal2"]
    ).all():
        raise ValueError("정상 데이터 이외의 자료 발견")

    values = frame[
        ["score", "threshold"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("비정상 점수 발견")

    if (frame["score"] < 0).any():
        raise ValueError("음수 복원 오차 발견")

    thresholds = frame["threshold"].unique()

    if len(thresholds) != 1 or thresholds[0] <= 0:
        raise ValueError("임계값 오류")

    threshold = float(thresholds[0])

    np.testing.assert_allclose(
        threshold,
        np.percentile(frame["score"], 95),
        rtol=1e-6,
        atol=1e-8,
    )

    frame["file"] = frame["file"].map(
        lambda name: str((ROOT / name).resolve())
    )

    if frame["file"].duplicated().any():
        raise ValueError("동일 경로 중복")

    return frame, threshold


for rpm in (900, 1200, 1800):

    print(f"\n===== {rpm} RPM =====")

    abs_dir = (
        ROOT / "outputs/experimental_integrated_ae"
        / f"{rpm}rpm"
    )

    rel_dir = (
        ROOT / "outputs/experimental_relative_ae"
        / f"{rpm}rpm"
    )

    abs_csv = abs_dir / "validation_scores.csv"
    rel_csv = rel_dir / "validation_scores.csv"

    absolute, abs_threshold = load_validation(abs_csv)
    relative, rel_threshold = load_validation(rel_csv)

    if len(absolute) != EXPECTED[rpm]:
        raise ValueError("절대 AE Validation 개수 오류")

    if len(relative) != EXPECTED[rpm]:
        raise ValueError("상대 AE Validation 개수 오류")

    if set(absolute["file"]) != set(relative["file"]):
        raise ValueError("두 모델의 평가 파일 불일치")

    # 9월 12일 정상 Validation 분할 확인
    split = pd.read_csv(
        abs_dir / "split.csv",
        dtype={"date": str},
    )

    valid = split[
        (split["split"] == "validation")
        & (split["date"] == "20260912")
        & split["condition"].isin(["Normal1", "Normal2"])
        & (split["rpm"] == rpm)
    ].copy()

    valid["file"] = valid["file"].map(
        lambda name: str((ROOT / name).resolve())
    )

    if (
        len(valid) != EXPECTED[rpm]
        or valid["file"].duplicated().any()
        or set(valid["file"]) != set(absolute["file"])
    ):
        raise ValueError("정상 Validation 분할 불일치")

    merged = absolute.merge(
        relative,
        on="file",
        suffixes=("_abs", "_rel"),
        validate="one_to_one",
    )

    if not (
        merged["condition_abs"]
        == merged["condition_rel"]
    ).all():
        raise ValueError("정상 조건 불일치")

    condition_map = valid.set_index("file")["condition"]

    if not (
        merged["file"].map(condition_map)
        == merged["condition_abs"]
    ).all():
        raise ValueError("분할 파일 조건 불일치")

    merged["absolute_ratio"] = (
        merged["score_abs"] / abs_threshold
    )

    merged["relative_ratio"] = (
        merged["score_rel"] / rel_threshold
    )

    merged["hybrid_score"] = (
        WEIGHT_ABS * merged["absolute_ratio"]
        + WEIGHT_REL * merged["relative_ratio"]
    )

    if not np.isfinite(merged["hybrid_score"]).all():
        raise ValueError("하이브리드 점수 오류")

    threshold = float(
        np.percentile(merged["hybrid_score"], 95)
    )

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("하이브리드 임계값 오류")

    merged["hybrid_threshold"] = threshold

    merged["predicted_anomaly"] = (
        merged["hybrid_score"] > threshold
    ).astype(int)

    dest = OUT / f"{rpm}rpm"
    dest.mkdir(parents=True, exist_ok=True)

    merged.to_csv(
        dest / "validation_scores.csv",
        index=False,
    )

    artifacts = {}

    for folder in (abs_dir, rel_dir):
        for name in (
            "model.pt",
            "normalization.npz",
            "validation_scores.csv",
        ):
            path = folder / name
            artifacts[str(path.relative_to(ROOT))] = sha256(path)

    artifacts[str((abs_dir / "split.csv").relative_to(ROOT))] = (
        sha256(abs_dir / "split.csv")
    )

    metadata = {
        "rpm": rpm,
        "model": "hybrid_ae",
        "weight_absolute": WEIGHT_ABS,
        "weight_relative": WEIGHT_REL,
        "absolute_threshold": abs_threshold,
        "relative_threshold": rel_threshold,
        "hybrid_threshold": threshold,
        "threshold_method": "normal_validation_q95",
        "validation_date": "20260912",
        "validation_files": len(merged),
        "experimental_only": True,
        "approved_for_production": False,
        "physical_run_independence_verified": False,
        "artifact_sha256": artifacts,
    }

    (dest / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Validation 파일:", len(merged))
    print(f"통합 AE 임계값: {abs_threshold:.9f}")
    print(f"상대 AE 임계값: {rel_threshold:.9f}")
    print(f"하이브리드 Q95: {threshold:.9f}")

    for condition, group in merged.groupby("condition_abs"):

        fp = int(group["predicted_anomaly"].sum())
        total = len(group)

        print(
            f"{condition}: "
            f"임계값 초과={fp}/{total} "
            f"({fp/total:.2%})"
        )

    print(
        "전체 임계값 초과:",
        int(merged["predicted_anomaly"].sum()),
        "/",
        len(merged),
    )

print("\n===== 하이브리드 임계값 생성 완료 =====")
print("결합 가중치: 0.5 : 0.5")
print("추가 모델 학습 없음")
print("기존 모델 및 임계값 변경 없음")
print("9/19 Test 점수 사용 없음")
print("저장 위치:", OUT)
