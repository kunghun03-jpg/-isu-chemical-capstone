"""상대 스펙트럼 AE 실험용 추론 모듈."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from vibration.model import Autoencoder
from vibration.relative_preprocessing import (
    ch3_relative_spectra,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "outputs/experimental_relative_ae"


class RelativeAEInference:

    def __init__(self, rpm):

        self.rpm = int(rpm)

        if self.rpm not in (900, 1200, 1800):
            raise ValueError("지원하지 않는 RPM")

        folder = BUNDLE_ROOT / f"{self.rpm}rpm"

        for name in (
            "model.pt",
            "normalization.npz",
            "validation_scores.csv",
        ):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)

        with np.load(
            folder / "normalization.npz",
            allow_pickle=False,
        ) as stats:
            self.mean = stats["mean"]
            self.std = stats["std"]

        if (
            self.mean.shape != (1, 1, 3201)
            or self.std.shape != self.mean.shape
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or (self.std < 1e-6).any()
        ):
            raise ValueError("정규화 통계 오류")

        validation = pd.read_csv(
            folder / "validation_scores.csv"
        )

        if (
            validation.empty
            or validation["file"].duplicated().any()
            or not validation["condition"].isin(
                ["Normal1", "Normal2"]
            ).all()
        ):
            raise ValueError("Validation 데이터 오류")

        scores = validation["score"].to_numpy(dtype=float)
        thresholds = validation["threshold"].to_numpy(dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError("Validation 점수 오류")

        if not np.isfinite(thresholds).all():
            raise ValueError("임계값 오류")

        unique = np.unique(thresholds)

        if len(unique) != 1:
            raise ValueError("임계값 불일치")

        self.threshold = float(unique[0])

        if not np.isclose(
            self.threshold,
            np.percentile(scores, 95),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError("Validation Q95 불일치")

        checkpoint = torch.load(
            folder / "model.pt",
            map_location="cpu",
            weights_only=True,
        )

        expected = (
            f"{self.rpm}rpm_relative_l2_experimental"
        )

        if checkpoint.get("experiment") != expected:
            raise ValueError("실험 모델 식별 정보 불일치")

        self.model = Autoencoder()

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()
        torch.set_num_threads(4)

    def predict(self, mat_path):

        path = Path(mat_path)

        mat = loadmat(path)

        values = ch3_relative_spectra(
            mat["signals"],
            mat["fs"],
        )

        # 학습 코드와 동일한 정규화 연산
        normalized = (
            (values - self.mean) / self.std
        ).astype(np.float32)

        if not np.isfinite(normalized).all():
            raise ValueError("정규화 결과 오류")

        x = torch.from_numpy(normalized)

        with torch.inference_mode():

            reconstructed = self.model(x)

            segment_scores = (
                (reconstructed.double() - x.double())
                .square()
                .mean(dim=(1, 2))
                .numpy()
            )

        if not np.isfinite(segment_scores).all():
            raise ValueError("복원 오차 계산 오류")

        score = float(segment_scores.mean())
        abnormal = score > self.threshold

        return {
            "file": str(path),
            "rpm": self.rpm,
            "model": "relative_ae",
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

    inference = RelativeAEInference(args.rpm)

    result = inference.predict(args.mat_path)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
