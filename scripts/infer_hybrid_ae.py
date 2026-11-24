"""통합 AE와 상대 스펙트럼 AE를 결합한 실험용 추론."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.infer_integrated_ae import predict as predict_absolute
from scripts.infer_relative_ae import RelativeAEInference

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experimental_hybrid_ae"


class HybridAEInference:

    def __init__(self, rpm):

        self.rpm = int(rpm)

        if self.rpm not in (900, 1200, 1800):
            raise ValueError("지원하지 않는 RPM")

        folder = BASE / f"{self.rpm}rpm"

        self.metadata = json.loads(
            (folder / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        meta = self.metadata

        if (
            meta["rpm"] != self.rpm
            or meta["model"] != "hybrid_ae"
            or meta["experimental_only"] is not True
            or meta["approved_for_production"] is not False
            or meta["threshold_method"] != "normal_validation_q95"
        ):
            raise ValueError("하이브리드 모델 정책 불일치")

        self.weight_abs = float(meta["weight_absolute"])
        self.weight_rel = float(meta["weight_relative"])

        if (self.weight_abs, self.weight_rel) != (0.5, 0.5):
            raise ValueError("결합 가중치 불일치")

        self.abs_threshold = float(meta["absolute_threshold"])
        self.rel_threshold = float(meta["relative_threshold"])
        self.threshold = float(meta["hybrid_threshold"])

        thresholds = [
            self.abs_threshold,
            self.rel_threshold,
            self.threshold,
        ]

        if (
            not np.isfinite(thresholds).all()
            or min(thresholds) <= 0
        ):
            raise ValueError("임계값 오류")

        # 원본 모델 및 검증 자료 무결성 검사
        for name, expected_hash in meta["artifact_sha256"].items():

            path = (ROOT / name).resolve()

            if not path.is_relative_to(ROOT):
                raise ValueError("프로젝트 외부 참조 차단")

            actual_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

            if actual_hash != expected_hash:
                raise ValueError(f"파일 무결성 오류: {name}")

        validation = pd.read_csv(
            folder / "validation_scores.csv"
        )

        if (
            len(validation) != meta["validation_files"]
            or validation["file"].duplicated().any()
            or not validation["condition_abs"].isin(
                ["Normal1", "Normal2"]
            ).all()
        ):
            raise ValueError("Validation 자료 오류")

        expected_scores = (
            self.weight_abs
            * validation["score_abs"] / self.abs_threshold
            + self.weight_rel
            * validation["score_rel"] / self.rel_threshold
        )

        np.testing.assert_allclose(
            validation["hybrid_score"],
            expected_scores,
            rtol=1e-9,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            self.threshold,
            np.percentile(expected_scores, 95),
            rtol=1e-9,
            atol=1e-12,
        )

        self.relative = RelativeAEInference(self.rpm)

        np.testing.assert_allclose(
            self.relative.threshold,
            self.rel_threshold,
            rtol=1e-9,
            atol=1e-12,
        )

    def predict(self, mat_path):

        absolute = predict_absolute(
            mat_path, self.rpm
        )

        relative = self.relative.predict(
            mat_path
        )

        np.testing.assert_allclose(
            absolute["threshold"],
            self.abs_threshold,
            rtol=1e-9,
            atol=1e-12,
        )

        abs_score = float(absolute["score"])
        rel_score = float(relative["score"])

        if (
            not np.isfinite([abs_score, rel_score]).all()
            or min(abs_score, rel_score) < 0
        ):
            raise ValueError("복원 오차 오류")

        abs_ratio = abs_score / self.abs_threshold
        rel_ratio = rel_score / self.rel_threshold

        score = (
            self.weight_abs * abs_ratio
            + self.weight_rel * rel_ratio
        )

        if not np.isfinite(score):
            raise ValueError("하이브리드 점수 오류")

        abnormal = score > self.threshold

        return {
            "file": str(mat_path),
            "rpm": self.rpm,
            "model": "hybrid_ae",
            "absolute_score": abs_score,
            "relative_score": rel_score,
            "absolute_ratio": abs_ratio,
            "relative_ratio": rel_ratio,
            "score": score,
            "threshold": self.threshold,
            "predicted_anomaly": int(abnormal),
            "prediction": (
                "Abnormal" if abnormal else "Normal"
            ),
            "experimental_only": True,
            "approved_for_production": False,
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("mat_path")
    parser.add_argument("--rpm", type=int, required=True)

    args = parser.parse_args()

    model = HybridAEInference(args.rpm)

    print(json.dumps(
        model.predict(args.mat_path),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ))
