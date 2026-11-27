"""P3 물리적 진동 특징 기반 실험용 개별 추론."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from vibration.physical_features import (
    FEATURE_NAMES,
    extract_physical_features,
)


SUPPORTED_DATES = (
    "20260912",
    "20260916",
    "20260919",
)

SUPPORTED_RPMS = (900, 1200, 1800)

METHOD = "P3_mixed_mean"

FEATURE_FLOOR = 1e-12
SCALE_FLOOR = 0.05

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "experimental_physical_features"
)


class PhysicalInferenceError(ValueError):
    """P3 실험용 추론 실행 오류."""


@dataclass(frozen=True)
class PhysicalReference:

    date: str
    rpm: int
    center: np.ndarray
    scale: np.ndarray
    threshold: float


def validate_condition(date, rpm):

    if date not in SUPPORTED_DATES:
        raise PhysicalInferenceError(
            f"Unsupported measurement date: {date}"
        )

    if rpm not in SUPPORTED_RPMS:
        raise PhysicalInferenceError(
            f"Unsupported RPM: {rpm}"
        )


def load_physical_reference(
    date,
    rpm,
    reference_root=None,
):

    validate_condition(date, rpm)

    root = (
        DEFAULT_ROOT
        if reference_root is None
        else Path(reference_root)
    )

    folder = root / date / f"{rpm}rpm"

    reference_path = (
        folder
        / "C_mixed_per_frequency_reference.npz"
    )

    metadata_path = (
        folder / "experiment_metadata.json"
    )

    if not (
        reference_path.is_file()
        and metadata_path.is_file()
    ):
        raise PhysicalInferenceError(
            f"Missing P3 reference: {date}/{rpm}"
        )

    try:

        meta = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        expected = {
            "date": date,
            "rpm": rpm,
            "feature_names": list(FEATURE_NAMES),
            "feature_transform": "log",
            "file_aggregation":
                "median_of_8_segments",
            "normal_reference": "median_mad",
            "feature_floor": FEATURE_FLOOR,
            "scale_floor": SCALE_FLOOR,
            "threshold_source":
                "normal_validation_q95",
            "evaluation_status":
                "exploratory_only",
        }

        for key, value in expected.items():

            if meta.get(key) != value:
                raise PhysicalInferenceError(
                    f"Metadata mismatch: {key}"
                )

        if (
            meta.get("approved_for_production")
                is not False
            or meta.get("run_independence_verified")
                is not False
            or meta.get("test_previously_examined")
                is not True
        ):
            raise PhysicalInferenceError(
                "Unexpected validation status"
            )

        with np.load(
            reference_path,
            allow_pickle=False,
        ) as data:

            center = np.asarray(
                data["center"],
                dtype=np.float64,
            ).copy()

            scale = np.asarray(
                data["scale"],
                dtype=np.float64,
            ).copy()

            names = tuple(
                data["feature_names"].tolist()
            )

        threshold = float(
            meta["thresholds"][METHOD]
        )

    except PhysicalInferenceError:
        raise

    except Exception as exc:
        raise PhysicalInferenceError(
            f"Failed to load P3 reference: {exc}"
        ) from exc

    if names != FEATURE_NAMES:
        raise PhysicalInferenceError(
            "Feature order mismatch"
        )

    if (
        center.shape != (10,)
        or scale.shape != (10,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale < SCALE_FLOOR).any()
    ):
        raise PhysicalInferenceError(
            "Invalid normal reference statistics"
        )

    if (
        not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise PhysicalInferenceError(
            "Invalid P3 threshold"
        )

    center.setflags(write=False)
    scale.setflags(write=False)

    return PhysicalReference(
        date=date,
        rpm=rpm,
        center=center,
        scale=scale,
        threshold=threshold,
    )


def analyze_physical_signals(
    signals,
    fs,
    date,
    rpm,
    reference_root=None,
):

    reference = load_physical_reference(
        date,
        rpm,
        reference_root,
    )

    try:

        segments = extract_physical_features(
            signals,
            fs,
            rpm,
        )

    except ValueError as exc:
        raise PhysicalInferenceError(
            f"Physical feature extraction failed: {exc}"
        ) from exc

    # 기존 P3 실험과 동일한 파일 집계
    features = np.median(
        segments,
        axis=0,
    )

    transformed = np.log(
        np.maximum(
            features,
            FEATURE_FLOOR,
        )
    )

    deviations = np.abs(
        (transformed - reference.center)
        / reference.scale
    )

    if (
        deviations.shape != (10,)
        or not np.isfinite(deviations).all()
    ):
        raise PhysicalInferenceError(
            "Invalid feature deviation scores"
        )

    score = float(deviations.mean())
    threshold = reference.threshold

    ratio = score / threshold

    if not np.isfinite(ratio):
        raise PhysicalInferenceError(
            "Invalid score ratio"
        )

    predicted_anomaly = bool(
        score > threshold
    )

    dominant_index = int(
        deviations.argmax()
    )

    return {
        "method": METHOD,
        "date": date,
        "rpm": rpm,

        "score": score,
        "threshold": threshold,
        "score_ratio": ratio,

        "predicted_anomaly":
            predicted_anomaly,

        "experimental_decision": (
            "Abnormal"
            if predicted_anomaly
            else "Normal"
        ),

        "features": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                features,
            )
        },

        "feature_deviations": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                deviations,
            )
        },

        "dominant_feature":
            FEATURE_NAMES[dominant_index],

        "segments": 8,
        "feature_count": 10,

        "evaluation_status":
            "exploratory_only",

        "approved_for_production": False,

        "run_independence_verified": False,

        "measurement_metadata_verified": False,
    }


def analyze_physical_mat(
    mat_source,
    date,
    rpm,
    reference_root=None,
):

    validate_condition(date, rpm)

    try:

        raw = loadmat(
            mat_source,
            variable_names=["signals", "fs"],
        )

        signals = raw["signals"]
        fs = raw["fs"]

    except Exception as exc:
        raise PhysicalInferenceError(
            f"Invalid MAT input: {exc}"
        ) from exc

    return analyze_physical_signals(
        signals=signals,
        fs=fs,
        date=date,
        rpm=rpm,
        reference_root=reference_root,
    )
