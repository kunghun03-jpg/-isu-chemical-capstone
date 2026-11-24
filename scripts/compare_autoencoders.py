"""기존 AE와 통합 AE의 동일 MAT 파일 비교 (실험용)."""

import argparse
import json
from pathlib import Path

import numpy as np

from vibration.inference import SavedModelInference
from scripts.infer_integrated_ae import predict as predict_integrated


OPERATING_BUNDLES = Path(
    "outputs/inference_bundles/date_prelim_20261106"
)


def compare(path, rpm):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    bundle = OPERATING_BUNDLES / f"{rpm}rpm"

    operating = SavedModelInference(
        bundle,
        models=("autoencoder",),
        selected_artifacts_only=True,
    )

    operating_result = operating.predict_mat(path, rpm=rpm)

    if "autoencoder" not in operating_result["models"]:
        raise ValueError("기존 AE 추론 결과 누락")

    old = operating_result["models"]["autoencoder"]

    experimental = predict_integrated(path, rpm)

    old_score = float(old["file_score"])
    old_threshold = float(old["file_threshold"])
    new_score = float(experimental["score"])
    new_threshold = float(experimental["threshold"])

    values = (
        old_score,
        old_threshold,
        new_score,
        new_threshold,
    )

    if not all(np.isfinite(v) for v in values):
        raise ValueError("비유한 점수 또는 임계값")

    old_prediction = int(old["predicted_anomaly"])
    new_prediction = int(experimental["predicted_anomaly"])

    if old_prediction not in (0, 1):
        raise ValueError("기존 AE 판정 오류")

    if new_prediction not in (0, 1):
        raise ValueError("통합 AE 판정 오류")

    return {
        "file": path.name,
        "rpm": rpm,
        "comparison_status": "experimental_only",
        "operating_ae": {
            "score": old_score,
            "threshold": old_threshold,
            "predicted_anomaly": old_prediction,
            "decision": old["decision"],
        },
        "integrated_ae": {
            "score": new_score,
            "threshold": new_threshold,
            "predicted_anomaly": new_prediction,
            "decision": experimental["decision"],
            "status": "experimental_only",
            "defect_recall_verified": False,
        },
        "decisions_match": old_prediction == new_prediction,
        "production_model_modified": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument(
        "--rpm",
        type=int,
        choices=(900, 1200, 1800),
        required=True,
    )

    args = parser.parse_args()

    result = compare(args.file, args.rpm)

    print("===== 기존 AE vs 통합 AE =====")
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
