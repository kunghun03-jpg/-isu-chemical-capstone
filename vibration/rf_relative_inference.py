
"""Portable inference for experimental relative-feature RF."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BUNDLE = (
    ROOT / "outputs/web_artifacts/rf_relative_candidate"
)

FEATURE_COLUMNS = (
    "crest_factor",
    "peak_over_rms",
    "one_x_over_rms",
    "two_x_over_rms",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "envelope_over_rms",
)

CLASSES = (
    "Bearing",
    "Looseness",
    "Unbalance",
)

REQUIRED_FEATURES = (
    "rms_ac",
    "crest_factor_ac",
    "peak_ac",
    "one_x_amplitude",
    "two_x_amplitude",
    "energy_low",
    "energy_mid",
    "energy_high",
    "envelope_rms_ac",
)


def sha256(path):

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def runtime_versions():

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "pandas": pd.__version__,
    }


def relative_feature_frame(features):
    """Convert existing CH3 features into RF ratios."""

    if not isinstance(features, dict):
        raise ValueError("RF 특징 입력 형식 오류")

    values = {}

    for key in REQUIRED_FEATURES:

        if key not in features or features[key] is None:
            raise ValueError(f"RF 특징 누락: {key}")

        value = float(features[key])

        if not np.isfinite(value) or value < 0:
            raise ValueError(f"유효하지 않은 특징: {key}")

        values[key] = value

    rms = values["rms_ac"]

    if rms < 1e-12:
        raise ValueError("유효한 RMS가 없어 비교 불가")

    safe_rms = max(abs(rms), 1e-12)

    row = {
        "crest_factor": values["crest_factor_ac"],

        "peak_over_rms":
            values["peak_ac"] / safe_rms,

        "one_x_over_rms":
            values["one_x_amplitude"] / safe_rms,

        "two_x_over_rms":
            values["two_x_amplitude"] / safe_rms,

        "low_energy_ratio":
            values["energy_low"] / safe_rms**2,

        "mid_energy_ratio":
            values["energy_mid"] / safe_rms**2,

        "high_energy_ratio":
            values["energy_high"] / safe_rms**2,

        "envelope_over_rms":
            values["envelope_rms_ac"] / safe_rms,
    }

    frame = pd.DataFrame(
        [row],
        columns=list(FEATURE_COLUMNS),
    )

    if not np.isfinite(
        frame.to_numpy(dtype=float)
    ).all():
        raise ValueError("상대 특징에 비유한값 포함")

    return frame


class RelativeRFInference:

    def __init__(self, folder=None):

        self.folder = Path(
            DEFAULT_BUNDLE if folder is None else folder
        ).expanduser().resolve()

        metadata_path = self.folder / "metadata.json"
        model_path = self.folder / "model.joblib"

        verification_path = (
            self.folder / "deployment_verification.json"
        )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        verification = json.loads(
            verification_path.read_text(encoding="utf-8")
        )

        if metadata.get("model_type") != (
            "experimental_relative_feature_rf"
        ):
            raise ValueError("상대 RF 모델 종류 불일치")

        if metadata.get("feature_columns") != list(
            FEATURE_COLUMNS
        ):
            raise ValueError("상대 RF 특징 순서 불일치")

        if metadata.get("classes") != list(CLASSES):
            raise ValueError("상대 RF 클래스 불일치")

        if metadata.get(
            "validated_for_operational_diagnosis"
        ) is not False:
            raise ValueError("운영 진단 제한 정책 불일치")

        if metadata.get("test_used") is not False:
            raise ValueError("상대 RF 시험 정책 불일치")

        expected = {
            "schema_version": 1,

            "purpose":
                "experimental_web_inference_only",

            "source_files_verified_at_export": True,

            "feature_source_sha256":
                metadata["feature_source_sha256"],

            "train_source_sha256":
                metadata["train_source_sha256"],

            "metadata_sha256": sha256(metadata_path),

            "model_sha256": sha256(model_path),

            "feature_columns": list(FEATURE_COLUMNS),

            "classes": list(CLASSES),

            "versions": runtime_versions(),

            "raw_sources_included": False,

            "approved_for_production": False,

            "validated_for_operational_diagnosis": False,
        }

        if verification != expected:
            raise ValueError(
                "상대 RF 배포 검증 기록 또는 환경 불일치"
            )

        # Verify original sources whenever available.
        # Missing local training files do not prevent
        # experimental inference from an exported bundle.

        source_checks = []

        for path_key, hash_key in (
            ("feature_source", "feature_source_sha256"),
            ("train_source", "train_source_sha256"),
        ):

            source = Path(metadata[path_key])

            if source.exists():

                if not source.is_file():
                    raise ValueError(
                        "상대 RF 원본 파일 형식 오류"
                    )

                if sha256(source) != metadata[hash_key]:
                    raise ValueError(
                        "상대 RF 학습 원본 SHA256 불일치"
                    )

                source_checks.append(True)

            else:

                source_checks.append(False)

        self.provenance_mode = (
            "source_files_verified"
            if all(source_checks)
            else "export_record_only"
        )

        # Load only the trusted project model artifact.

        self.model = joblib.load(model_path)

        if self.model.n_features_in_ != len(
            FEATURE_COLUMNS
        ):
            raise ValueError("상대 RF 입력 차원 불일치")

        if list(self.model.feature_names_in_) != list(
            FEATURE_COLUMNS
        ):
            raise ValueError("저장 모델 특징 순서 불일치")

        if set(self.model.classes_) != set(CLASSES):
            raise ValueError("저장 모델 클래스 불일치")

        self.metadata = metadata

    def predict(self, features):

        frame = relative_feature_frame(features)

        prediction = str(
            self.model.predict(frame)[0]
        )

        probabilities = np.asarray(
            self.model.predict_proba(frame)[0],
            dtype=float,
        )

        if (
            probabilities.shape != (len(CLASSES),)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError("상대 RF 분류 점수 오류")

        scores = {
            str(name): float(score)
            for name, score in zip(
                self.model.classes_,
                probabilities,
            )
        }

        if prediction not in CLASSES:
            raise ValueError("상대 RF 예측 클래스 오류")

        if not np.isclose(
            scores[prediction],
            probabilities.max(),
        ):
            raise ValueError("상대 RF 예측·점수 불일치")

        return {
            "model": "Relative RF",
            "raw_prediction": prediction,
            "scores_uncalibrated": scores,
            "provenance_mode": self.provenance_mode,
            "evaluation_scope": "experimental",
            "validated_for_operational_diagnosis": False,
            "approved_for_production": False,
        }


def predict_relative_rf(features, folder=None):

    return RelativeRFInference(folder).predict(features)
