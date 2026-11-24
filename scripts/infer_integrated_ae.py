"""Normal1+Normal2 통합 AE의 실험용 MAT 추론."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from vibration.model import Autoencoder
from vibration.preprocessing import ch3_log_spectra


BUNDLE_ROOT = Path(
    "outputs/experimental_integrated_ae"
)


def predict(mat_path, rpm):
    bundle = BUNDLE_ROOT / f"{rpm}rpm"

    manifest_path = bundle / "manifest.json"

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    if (
        manifest.get("schema_version") != 1
        or manifest.get("rpm") != rpm
        or manifest.get("model") != "integrated_autoencoder"
        or manifest.get("status") != "experimental_only"
        or manifest.get("approved_for_production") is not False
        or manifest.get("defect_recall_verified") is not False
        or manifest.get("run_independence_verified") is not False
        or manifest.get("threshold_source") != "normal_validation_only"
        or manifest.get("normalization_source") != "normal_train_only"
        or manifest.get("score_direction") != "higher_is_more_anomalous"
        or manifest.get("segment_aggregation") != "mean"
    ):
        raise ValueError("실험용 번들 정책 불일치")

    # 저장된 모델 파일의 무결성 확인
    for filename, expected_hash in (
        manifest["artifacts_sha256"].items()
    ):
        path = bundle / filename

        actual_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                f"번들 파일 무결성 오류: {filename}"
            )

    threshold = float(manifest["threshold"])

    if not np.isfinite(threshold):
        raise ValueError("임계값 오류")

    checkpoint = torch.load(
        bundle / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = Autoencoder()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    with np.load(
        bundle / "normalization.npz"
    ) as stats:
        mean = stats["mean"]
        std = stats["std"]

    if (
        mean.shape != (1, 1, 3201)
        or std.shape != (1, 1, 3201)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("정규화 통계 오류")

    mat = loadmat(mat_path)

    raw = np.asarray(
        ch3_log_spectra(
            mat["signals"],
            mat["fs"],
        ),
        dtype=np.float32,
    )

    if raw.shape != (8, 1, 3201):
        raise ValueError("MAT 스펙트럼 크기 오류")

    if not np.isfinite(raw).all():
        raise ValueError("비유한 입력 스펙트럼")

    x = ((raw - mean) / std).astype(
        np.float32
    )

    if not np.isfinite(x).all():
        raise ValueError("정규화된 입력 오류")

    tensor = torch.from_numpy(x)

    with torch.inference_mode():
        reconstructed = model(tensor)

        segment_scores = (
            (reconstructed.double() - tensor.double())
            .square()
            .mean(dim=(1, 2))
            .numpy()
        )

    score = float(segment_scores.mean())

    if not np.isfinite(score):
        raise ValueError("AE 점수 오류")

    predicted = int(score > threshold)

    return {
        "model": "integrated_autoencoder",
        "status": "experimental_only",
        "rpm": rpm,
        "rpm_source": "caller_supplied",
        "file": Path(mat_path).name,
        "score": score,
        "threshold": threshold,
        "predicted_anomaly": predicted,
        "decision": (
            "Abnormal" if predicted else "Normal"
        ),
        "defect_recall_verified": False,
        "run_independence_verified": False,
        "modifies_production_model": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file", required=True
    )
    parser.add_argument(
        "--rpm",
        type=int,
        choices=(900, 1200, 1800),
        required=True,
    )

    args = parser.parse_args()

    path = Path(args.file)

    if not path.is_file():
        raise FileNotFoundError(path)

    result = predict(path, args.rpm)

    print(
        "===== 통합 AE 실험용 추론 ====="
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "기존 운영 모델 및 임계값 변경 없음"
    )


if __name__ == "__main__":
    main()
