"""저장된 기준 실험을 재학습 없이 이동 가능한 추론 패키지로 내보냅니다."""

from pathlib import Path
import csv
import hashlib
import json
import shutil

import numpy as np

from vibration.experiment import check_fingerprint, load_experiment
from vibration.preprocessing import PREPROCESSING_CONFIG, apply_normalization

MODEL_NAMES = ("autoencoder", "ocsvm", "iforest")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_id(manifest):
    fields = {key: value for key, value in manifest.items() if key != "bundle_id"}
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def export_bundle(source, destination, rpm):
    """배치 검증은 내보낼 때만 수행하고, 추론 때는 Train/Test NPZ를 요구하지 않습니다."""
    import joblib
    import sklearn
    import torch

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("기존 추론 패키지는 덮어쓰지 않습니다.")
    if destination.is_relative_to(source):
        raise ValueError("원본 실험 폴더 내부에는 내보내지 않습니다.")
    datasets, fingerprint = load_experiment(source, rpm)
    with np.load(source / f"normalization_{rpm}rpm.npz", allow_pickle=False) as data:
        mean, std = data["mean"].copy(), data["std"].copy()
    apply_normalization(np.zeros((8, 1, 3201), dtype=np.float32), mean, std)
    config_path = source / f"normalization_config_{rpm}rpm.json"
    if (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8"))["mode"]
        != "per_frequency"
    ):
        raise ValueError("채택한 주파수별 정규화 기준 모델만 내보냅니다.")
    for split in datasets:
        with np.load(source / f"{split}_spectra.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(
                apply_normalization(raw["X"], mean, std), datasets[split]["X"]
            )
    del datasets
    checkpoint = torch.load(
        source / f"autoencoder_{rpm}rpm.pt", map_location="cpu", weights_only=True
    )
    check_fingerprint(checkpoint, fingerprint)
    if checkpoint.get("channels") != [3] or checkpoint.get("input_shape") != [1, 3201]:
        raise ValueError("CH3 AE가 아닙니다.")
    thresholds_segment = json.loads(
        (source / f"threshold_{rpm}rpm.json").read_text(encoding="utf-8")
    )
    with (source / f"evaluation_file_{rpm}rpm.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        values = {float(r["threshold"]) for r in csv.DictReader(stream)}
    if len(values) != 1:
        raise ValueError("AE 저장 파일 임계값 불일치")
    models = {
        "autoencoder": {
            "artifact": "autoencoder.pt",
            "score_method": "double_precision_reconstruction_MSE",
            "segment_threshold": float(thresholds_segment["threshold"]),
            "file_threshold": values.pop(),
            "epoch": checkpoint["epoch"],
        }
    }
    sources = {
        "normalization.npz": source / f"normalization_{rpm}rpm.npz",
        "autoencoder.pt": source / f"autoencoder_{rpm}rpm.pt",
    }
    for level in ("segment", "file"):
        record = json.loads(
            (source / f"evaluation_{level}_context_{rpm}rpm.json").read_text(
                encoding="utf-8"
            )
        )
        check_fingerprint(record, fingerprint)
    for name in ("ocsvm", "iforest"):
        folder = source / "comparisons" / name
        saved = joblib.load(folder / "model.joblib")
        check_fingerprint(saved["config"], fingerprint)
        if saved["config"].get("channels") != [3] or saved["config"].get(
            "feature_shape"
        ) != [3201]:
            raise ValueError("비교 모델 입력 설정 불일치")
        thresholds = json.loads(
            (folder / "thresholds.json").read_text(encoding="utf-8")
        )
        np.testing.assert_allclose(
            [thresholds["segment"], thresholds["file"]],
            [saved["thresholds"]["segment"], saved["thresholds"]["file"]],
        )
        models[name] = {
            "artifact": f"{name}.joblib",
            "score_method": "negative_score_samples",
            "segment_threshold": float(thresholds["segment"]),
            "file_threshold": float(thresholds["file"]),
        }
        sources[f"{name}.joblib"] = folder / "model.joblib"
    summary = json.loads(
        (source / "comparison_summary.json").read_text(encoding="utf-8")
    )
    check_fingerprint(summary, fingerprint)
    for name in MODEL_NAMES:
        if not np.isfinite(
            [models[name]["segment_threshold"], models[name]["file_threshold"]]
        ).all():
            raise ValueError("임계값이 유한하지 않습니다.")
    manifest = {
        "schema_version": 1,
        "rpm": rpm,
        "source_experiment": source.name,
        "data_fingerprint": fingerprint,
        "experiment_mode": summary["experiment_mode"],
        "run_independence_verified": False,
        "preprocessing": PREPROCESSING_CONFIG,
        "preprocessing_provenance": "legacy batch formula, explicitly recorded and verified against archived pre-Z/normalized data",
        "normalization": {
            "artifact": "normalization.npz",
            "mode": "per_frequency",
            "reference_split": "train",
            "shape": [1, 1, 3201],
        },
        "file_aggregation": "arithmetic_mean_of_8_segment_scores",
        "decision_rule": "score > threshold",
        "models": models,
        "artifacts": {},
        "versions": {
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "scikit_learn": sklearn.__version__,
        },
        "reference_file_metrics": {
            name: summary["models"][name]["file"] for name in MODEL_NAMES
        },
        "limitations": [
            "Date-group preliminary models; actual independent Run boundaries unverified.",
            "High normal false-positive rates on Sep 19; not production-validated.",
            "RPM is caller-supplied nominal metadata, not measured speed.",
            "No DAE, fault-cause estimation, warning tier, SQLite or web UI in this inference function.",
        ],
    }
    destination.mkdir(parents=True)
    for name, path in sources.items():
        shutil.copyfile(path, destination / name)
        checksum = sha256_file(path)
        if sha256_file(destination / name) != checksum:
            raise ValueError("내보낸 모델/정규화 파일 해시 불일치")
        manifest["artifacts"][name] = checksum
    manifest["bundle_id"] = manifest_id(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
