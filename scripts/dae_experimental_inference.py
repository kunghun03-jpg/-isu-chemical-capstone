"""DAE 실험용 MAT 직접 추론 및 오프라인 점수 일치 검증.

운영용 DAE 번들/지원 RPM 설정은 변경하지 않습니다.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dae_group_split_experiment_20261119"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dae_legacy"))

from model import ConvAutoencoder
from vibration.dae_preprocessing import (
    preprocess_mat_bytes,
    normalize_spectrum,
)


class ExperimentalDAE:
    def __init__(self):
        with np.load(
            OUT / "dae_normalization.npz",
            allow_pickle=False
        ) as data:
            self.mean = data["mean"].copy()
            self.std = data["std"].copy()

        with np.load(
            OUT / "dae_thresholds.npz",
            allow_pickle=False
        ) as data:
            self.thresholds = {
                rpm: float(data[f"rpm_{rpm}_q95"])
                for rpm in (900, 1200, 1800)
            }

        self.model = ConvAutoencoder().cpu()

        state = torch.load(
            OUT / "dae_group_split_best.pth",
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict_bytes(self, raw, rpm):
        if rpm not in self.thresholds:
            raise ValueError(f"지원하지 않는 실험 RPM: {rpm}")

        spectrum = preprocess_mat_bytes(raw)

        normalized = normalize_spectrum(
            spectrum, self.mean, self.std
        ).astype(np.float32)

        x = torch.from_numpy(
            normalized
        ).reshape(1, 1, 800)

        with torch.no_grad():
            reconstructed = self.model(x)
            score = float(
                torch.mean((reconstructed - x) ** 2).item()
            )

        if not math.isfinite(score):
            raise ValueError("비유한 DAE 점수")

        threshold = self.thresholds[rpm]

        return {
            "rpm": rpm,
            "score": score,
            "threshold": threshold,
            "predicted_anomaly": int(score > threshold),
            "evaluation_scope": "date_based_exploratory",
        }


def main():
    reference_path = OUT / "dae_test_scores.csv"

    with reference_path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reference = list(csv.DictReader(f))

    # 이전에 AE·OCSVM·IF로 검증했던 동일한 6개 파일
    samples = {
        900: (
            "20260919_131643_Vib.mat",
            "20260919_141357_Vib.mat",
        ),
        1200: (
            "20260919_131353_Vib.mat",
            "20260919_141659_Vib.mat",
        ),
        1800: (
            "20260919_131119_Vib.mat",
            "20260919_142140_Vib.mat",
        ),
    }

    engine = ExperimentalDAE()
    passed = 0

    print("===== DAE 실험용 직접 추론 검증 =====")

    for rpm, filenames in samples.items():
        for split, filename in zip(
            ("test_normal", "test_anomaly"),
            filenames
        ):
            matches = [
                row for row in reference
                if int(row["rpm"]) == rpm
                and row["split"] == split
                and Path(row["file"]).name == filename
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"평가 CSV의 파일 일치 개수 오류: {filename}"
                )

            saved = matches[0]
            path = ROOT / saved["file"]

            actual = engine.predict_bytes(
                path.read_bytes(), rpm
            )

            score_ok = math.isclose(
                actual["score"],
                float(saved["score"]),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )

            threshold_ok = math.isclose(
                actual["threshold"],
                float(saved["threshold"]),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )

            decision_ok = (
                actual["predicted_anomaly"]
                == int(saved["predicted_anomaly"])
            )

            ok = score_ok and threshold_ok and decision_ok

            print(f"\n[{rpm}RPM / {split}]")
            print("파일:", filename)
            print("직접 추론 MSE:", actual["score"])
            print("기존 평가 MSE:", saved["score"])
            print("저장 임계값:", actual["threshold"])
            print(
                "실험용 판정:",
                "이상" if actual["predicted_anomaly"] else "정상"
            )
            print("검증:", "일치" if ok else "불일치")

            if not ok:
                raise AssertionError(
                    f"DAE 평가 불일치: {filename}"
                )

            passed += 1

    print("\n===== 최종 결과 =====")
    print("검사 파일:", passed)
    print("불일치:", 6 - passed)
    print("DAE 실험용 직접 추론 검증 통과")


if __name__ == "__main__":
    main()
