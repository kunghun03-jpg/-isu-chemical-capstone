"""네 가지 AE 방식의 실험용 통합 비교."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_autoencoders import compare
from scripts.infer_hybrid_ae import HybridAEInference


def normalize_result(result, model_name):

    score = float(result["score"])
    threshold = float(result["threshold"])

    if (
        not np.isfinite([score, threshold]).all()
        or score < 0
        or threshold <= 0
    ):
        raise ValueError(f"{model_name} 점수 오류")

    abnormal = score > threshold

    return {
        "model": model_name,
        "score": score,
        "threshold": threshold,
        "score_ratio": score / threshold,
        "prediction": "Abnormal" if abnormal else "Normal",
        "predicted_anomaly": int(abnormal),
        "experimental_only": model_name != "original_ae",
    }


def compare_four(mat_path, rpm):

    rpm = int(rpm)
    path = Path(mat_path)

    if rpm not in (900, 1200, 1800):
        raise ValueError("지원하지 않는 RPM")

    if not path.is_file():
        raise FileNotFoundError(path)

    # 기존 AE + 통합 AE
    base = compare(path, rpm)

    # 기존 비교 모듈 반환 구조 확인
    original = None
    integrated = None

    for key in (
        "operating_ae",
        "original",
        "original_ae",
        "baseline",
        "baseline_ae",
        "existing",
        "existing_ae",
    ):
        if isinstance(base.get(key), dict):
            original = base[key]
            break

    for key in (
        "integrated",
        "integrated_ae",
        "experimental",
        "experimental_ae",
    ):
        if isinstance(base.get(key), dict):
            integrated = base[key]
            break

    if original is None or integrated is None:
        raise ValueError(
            "기존 비교 모듈 반환 구조 확인 필요: "
            + str(list(base.keys()))
        )

    original = normalize_result(
        original, "original_ae"
    )

    integrated = normalize_result(
        integrated, "integrated_ae"
    )

    hybrid_model = HybridAEInference(rpm)

    # 상대 AE는 한 번만 추론
    relative = normalize_result(
        hybrid_model.relative.predict(path),
        "relative_ae",
    )

    np.testing.assert_allclose(
        integrated["threshold"],
        hybrid_model.abs_threshold,
        rtol=1e-9,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        relative["threshold"],
        hybrid_model.rel_threshold,
        rtol=1e-9,
        atol=1e-12,
    )

    # 이미 계산된 점수 재사용
    hybrid_score = (
        hybrid_model.weight_abs
        * integrated["score_ratio"]
        + hybrid_model.weight_rel
        * relative["score_ratio"]
    )

    hybrid = normalize_result(
        {
            "score": hybrid_score,
            "threshold": hybrid_model.threshold,
        },
        "hybrid_ae",
    )

    return {
        "file": str(path),
        "rpm": rpm,
        "models": {
            "original_ae": original,
            "integrated_ae": integrated,
            "relative_ae": relative,
            "hybrid_ae": hybrid,
        },
        "operational_prediction": original["prediction"],
        "experimental_comparison_only": True,
        "approved_for_production": False,
        "operational_model_changed": False,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("mat_path")
    parser.add_argument("--rpm", type=int, required=True)

    args = parser.parse_args()

    print(json.dumps(
        compare_four(args.mat_path, args.rpm),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))
