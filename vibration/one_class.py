"""CH3 정규화 스펙트럼을 사용하는 OCSVM·Isolation Forest 공통 흐름."""

from pathlib import Path
import json
import platform

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from vibration.channel import validate_spectra
from vibration.comparison_metrics import (
    file_scores,
    metrics,
    predictions,
    q95,
    score_rows,
    write_csv,
)
from vibration.experiment import (
    PARTITIONS,
    check_fingerprint,
    load_experiment,
    experiment_metadata,
)

PARAMETERS = {
    "ocsvm": {
        "kernel": "rbf",
        "nu": 0.05,
        "gamma": "scale",
        "cache_size": 200,
        "tol": 0.001,
    },
    "iforest": {
        "n_estimators": 300,
        "max_samples": "auto",
        "contamination": "auto",
        "max_features": 1.0,
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": 1,
    },
}


def features(values):
    """(구간 수, 1, 3201) → (구간 수, 3201). 추가 학습·축약·스케일링은 하지 않습니다."""
    validate_spectra(values)
    return np.ascontiguousarray(values[:, 0, :], dtype=np.float64)


def make_model(name):
    if name == "ocsvm":
        return OneClassSVM(**PARAMETERS[name])
    if name == "iforest":
        return IsolationForest(**PARAMETERS[name])
    raise ValueError(f"알 수 없는 비교 모델: {name}")


def anomaly_scores(model, values):
    # sklearn의 정상일수록 큰 점수에 부호를 반전합니다. predict()의 내장 경계는 사용하지 않습니다.
    result = -np.asarray(model.score_samples(features(values)), dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("비교 모델 이상점수에 NaN 또는 무한대가 있습니다.")
    return result


def run_model(output_dir, rpm, name):
    # 모델 생성·학습·결과 저장보다 먼저 Run과 모든 NPZ의 동일성을 검사합니다.
    datasets, fingerprint = load_experiment(output_dir, rpm)
    model = make_model(name)
    model.fit(features(datasets["train"]["X"]))
    # 두 단위의 임계값 모두 정상 검증 데이터만으로 결정합니다.
    validation = score_rows(
        datasets["validation"],
        "validation",
        anomaly_scores(model, datasets["validation"]["X"]),
    )
    thresholds = {"segment": q95(validation), "file": q95(file_scores(validation))}
    rows = []
    for split in PARTITIONS:
        data = datasets[split]
        rows.extend(
            validation
            if split == "validation"
            else score_rows(data, split, anomaly_scores(model, data["X"]))
        )
    levels = {
        "segment": predictions(rows, thresholds["segment"]),
        "file": predictions(file_scores(rows), thresholds["file"]),
    }
    configuration = {
        "model": name,
        "parameters": model.get_params(),
        "channels": [3],
        "feature_method": "normalized_CH3_segment_spectrum_3201_bins",
        "feature_shape": [3201],
        "file_aggregation": "mean_of_8_segment_scores",
        "score_direction": "negative_score_samples; higher_is_more_anomalous",
        "threshold_method": "normal_validation_Q95_strict_greater_than",
        "data_fingerprint": fingerprint,
        **experiment_metadata(output_dir, rpm),
        "fit_partition": "train",
        "sample_counts": {
            split: {
                "segments": len(data["X"]),
                "files": len(np.unique(data["source_file"])),
            }
            for split, data in datasets.items()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    base = Path(output_dir) / "comparisons" / name
    base.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "config": configuration, "thresholds": thresholds},
        base / "model.joblib",
    )
    for level, selected in levels.items():
        write_csv(base / f"scores_{level}.csv", selected)
    for filename, content in {
        "config.json": configuration,
        "thresholds.json": thresholds,
        "metrics.json": {
            level: metrics(selected) for level, selected in levels.items()
        },
    }.items():
        (base / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"{name}: 모델·구간/파일 점수·Q95·평가 결과 저장: {base}")


def load_model(path, *, fingerprint):
    """이 프로젝트에서 저장한 신뢰할 수 있는 joblib 파일만 불러오세요."""
    record = joblib.load(path)
    check_fingerprint(record["config"], fingerprint)
    return record
