"""DAE-only inference bundle schema and explicit legacy artifact conversion."""

from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import scipy
import torch


PREPROCESSING = {
    "id": "legacy_dae_welch_800_v1",
    "raw_shape": [6, 65536], "fs": 25600, "channel_index": 2,
    "quality_scope": "CH3", "ch3_dtype": "float64",
    "welch": {"window": "hann", "nperseg": 8192, "noverlap": 4096,
              "nfft": 8192, "detrend": "constant", "return_onesided": True,
              "scaling": "density", "average": "mean"},
    "psd_transform": "sqrt", "frequency_hz": [0, 10000],
    "upper_inclusive": False, "selected_bins": 3200,
    "pooling": "reshape(800,4).mean(axis=1)",
    "log": "log10(x+1e-12)", "output_dtype": "float32",
    "normalization": "(log_spectrum-mean)/std; preserve stored dtype; no clipping",
    "input_shape": [1, 1, 800],
}
SCORING = {"method": "float32_reconstruction_mse", "unit": "whole_file",
           "decision_rule": "score > threshold", "threshold_source": "legacy_saved_value"}
FILES = ("autoencoder_model.pth", "normalization.npz", "threshold.npz")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_id(manifest):
    content = {k: v for k, v in manifest.items() if k != "bundle_id"}
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_rpm_metadata(supported_rpms, evidence):
    if supported_rpms is None:
        if evidence is not None:
            raise ValueError("미확인 RPM에는 근거를 지정할 수 없습니다.")
        return
    if (
        not isinstance(supported_rpms, list) or not supported_rpms
        or any(type(rpm) is not int or rpm <= 0 for rpm in supported_rpms)
        or len(set(supported_rpms)) != len(supported_rpms)
        or not isinstance(evidence, str) or not evidence.strip()
    ):
        raise ValueError("지원 RPM 목록과 확인 근거가 필요합니다.")


def check_rpm(manifest, rpm):
    if type(rpm) is not int or rpm <= 0:
        raise ValueError("입력 RPM은 양의 정수로 명시해야 합니다.")
    supported = manifest["supported_rpms"]
    if supported is None:
        raise ValueError("DAE 모델의 지원 RPM이 미확인 상태입니다. 추론을 실행하지 않습니다.")
    if rpm not in supported:
        raise ValueError(f"DAE 번들이 지원하지 않는 RPM: {rpm}")


def load_integrity_manifest(folder):
    folder = Path(folder).resolve()
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("DAE manifest JSON 형식 오류") from error
    if not isinstance(manifest, dict):
        raise ValueError("DAE manifest는 객체여야 합니다.")
    required = {"schema_version", "model_type", "architecture", "model_version",
                "preprocessing", "scoring", "supported_rpms", "rpm_evidence",
                "runtime_versions", "training_provenance", "artifacts", "bundle_id"}
    if not required.issubset(manifest):
        raise ValueError("DAE manifest 필수 항목 누락")
    if (
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
        or manifest["model_type"] != "dae"
        or manifest["architecture"] != "legacy_conv_autoencoder_800_v1"
        or not isinstance(manifest["model_version"], str) or not manifest["model_version"].strip()
        or manifest["preprocessing"] != PREPROCESSING or manifest["scoring"] != SCORING
        or not isinstance(manifest["training_provenance"], str)
    ):
        raise ValueError("지원하지 않는 DAE 모델·전처리·점수 규격")
    validate_rpm_metadata(manifest["supported_rpms"], manifest["rpm_evidence"])
    versions = manifest["runtime_versions"]
    if not isinstance(versions,dict) or set(versions)!={"numpy","scipy","torch"} or not all(isinstance(v,str) and v for v in versions.values()):
        raise ValueError("DAE runtime_versions 형식 오류")
    try:
        valid_id = manifest_id(manifest)
    except (TypeError, ValueError) as error:
        raise ValueError("DAE manifest 값 형식 오류") from error
    if manifest["bundle_id"] != valid_id:
        raise ValueError("DAE manifest 해시 불일치")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(FILES):
        raise ValueError("DAE 번들 파일 목록 오류")
    for name in FILES:
        checksum = artifacts[name]
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdef" for c in checksum)):
            raise ValueError("DAE SHA256 형식 오류")
        path = folder / name
        if path.is_symlink():
            raise ValueError("DAE 번들 파일은 심볼릭 링크일 수 없습니다.")
        if not path.is_file():
            raise FileNotFoundError(f"DAE 번들 파일 누락: {path}")
        if sha256_file(path) != checksum:
            raise ValueError(f"DAE 파일 SHA256 불일치: {name}")
    return manifest


def load_manifest(folder):
    manifest=load_integrity_manifest(folder)
    expected,actual=manifest['runtime_versions'],runtime_versions()
    if expected!=actual:
        logging.getLogger(__name__).warning('DAE runtime mismatch: expected=%s actual=%s bundle_id=%s',expected,actual,manifest['bundle_id'])
        report=os.environ.get('ISU_CHEMICAL_DAE_RUNTIME_REPORT','')
        if not report:
            raise ValueError('DAE 번들의 NumPy/SciPy/PyTorch 버전과 실행 환경이 다릅니다. 검증된 호환성 보고서가 필요합니다.')
        from vibration.dae_runtime_compat import validate_report
        try:
            validate_report(report,manifest)
        except Exception as error:
            raise ValueError(f'DAE 실행 환경 호환성 근거 검증 실패: {type(error).__name__}: {error}') from error
    return manifest


def runtime_versions():
    return {"numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}


def convert_legacy_bundle(source, destination, *, model_version, supported_rpms=None,
                          rpm_evidence=None, training_provenance="unverified"):
    """Extract saved statistics only; never fit, infer RPMs, or overwrite outputs.

    supported_rpms defaults to None. Explicit RPM declarations require a caller's
    evidence description; conversion cannot independently establish model coverage.
    Recorded runtime versions describe conversion, not the original training run.
    """
    from vibration.dae_inference import SavedDAEInference, SavedDAEBundleInference

    validate_rpm_metadata(supported_rpms, rpm_evidence)
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("명시적인 DAE 모델 버전이 필요합니다.")
    if not isinstance(training_provenance, str) or not training_provenance.strip():
        raise ValueError("학습 출처 또는 unverified를 지정하세요.")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"기존 DAE 번들을 덮어쓰지 않습니다: {destination}")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("원본과 번들 경로는 서로 포함될 수 없습니다.")
    engine = SavedDAEInference(source)  # Validate statistics and actual checkpoint first.
    destination.mkdir(parents=True)
    try:
        shutil.copyfile(source / "autoencoder_model.pth", destination / "autoencoder_model.pth")
        np.savez(destination / "normalization.npz", mean=engine.mean, std=engine.std)
        np.savez(destination / "threshold.npz", threshold=engine.threshold)
        manifest = {
            "schema_version": 1, "model_type": "dae",
            "architecture": "legacy_conv_autoencoder_800_v1", "model_version": model_version,
            "preprocessing": deepcopy(PREPROCESSING), "scoring": deepcopy(SCORING),
            "supported_rpms": supported_rpms, "rpm_evidence": rpm_evidence,
            "training_provenance": training_provenance, "runtime_versions": runtime_versions(),
            "artifacts": {name: sha256_file(destination / name) for name in FILES},
        }
        manifest["bundle_id"] = manifest_id(manifest)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        SavedDAEBundleInference(destination)  # Validate emitted artifacts before reporting success.
    except Exception:
        shutil.rmtree(destination)
        raise
    return manifest
