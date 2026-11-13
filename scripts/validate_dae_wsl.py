"""Local-only DAE conversion/parity checks; never confirms RPM coverage or accuracy."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
from scipy.io import savemat

from vibration.dae_bundle import convert_legacy_bundle, runtime_versions, sha256_file
from vibration.dae_inference import SavedDAEBundleInference, SavedDAEInference
from vibration.dae_preprocessing import preprocess_mat_bytes

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("autoencoder_model.pth", "normal_dataset_normalized.npz",
             "normal_threshold.npz", "band_statistics.npz")

# Run trusted legacy code in a separate interpreter so its top-level imports do
# not shadow the current package. Do not import training/evaluation scripts.
REFERENCE_CODE = """
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import dae_inference
from web_preprocessing import preprocess_uploaded_mat
dae_inference.BASE_DIR = Path(sys.argv[2])
raw = BytesIO(Path(sys.argv[3]).read_bytes())
features = preprocess_uploaded_mat(raw)
result = dae_inference.run_dae_inference(raw)
np.savez(sys.argv[4], features=features, **result)
"""


def array_headers(path):
    """Inspect NPZ array headers without loading training matrices or pickle data."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"지원하지 않는 NPY header 버전: {version}")
            result[name[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    return result


def investigate_rpm(source, legacy_code):
    headers = array_headers(source / "normal_dataset_normalized.npz")
    observations = {}
    with np.load(source / "normal_dataset_normalized.npz", allow_pickle=False) as data:
        for key, header in headers.items():
            if not re.search(r"rpm|group|file|date|run", key, re.I):
                continue
            # Metadata only; never materialize large arrays or object/pickle data.
            if np.prod(header["shape"], dtype=np.int64) > 10000 or "object" in header["dtype"]:
                continue
            values = data[key]
            if values.dtype.kind in "US":
                found = set()
                for value in values.flat:
                    found.update(int(x) for x in re.findall(r"(\d+)RPM", str(value), re.I))
                observations[key] = sorted(found)
            elif "rpm" in key.lower() and values.dtype.kind in "iu":
                observations[key] = sorted(int(x) for x in np.unique(values))
    clues = []
    for path in sorted(legacy_code.glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\bRPMS?\b|train_test_split|X_train|X_val|mean\s*=|std\s*=", line):
                clues.append({"file": path.name, "line": number, "text": line.strip()[:240]})
    return {"supported_rpms": None, "status": "unconfirmed",
            "normal_dataset_headers": headers, "metadata_rpm_observations": observations,
            "code_clues": clues,
            "limitation": "Metadata/code references are clues, not validated model coverage. "
                          "Fault-evaluation RPM lists do not establish training RPMs or independent runs."}


def compare_result(reference, actual):
    for key in ("input_spectrum", "reconstructed_spectrum"):
        np.testing.assert_array_equal(reference[key], actual[key], err_msg=key)
    for key in ("mse", "threshold", "prediction"):
        expected = np.asarray(reference[key]).item()
        if expected != actual[key]:
            raise ValueError(f"legacy/new 결과 불일치: {key}: {expected!r} != {actual[key]!r}")


def validate(source, legacy_code, output, mat=None):
    source, legacy_code, output = [Path(p).expanduser().resolve() for p in (source, legacy_code, output)]
    if output.is_relative_to(ROOT):
        raise ValueError("검증 산출물은 Git 체크아웃 외부에 저장해야 합니다.")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("원본과 검증 출력 경로는 서로 포함될 수 없습니다.")
    for name in ARTIFACTS[:3]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"실제 파일에 접근할 수 없습니다: {source / name}")
    for name in ("model.py", "web_preprocessing.py", "dae_inference.py"):
        if not (legacy_code / name).is_file():
            raise FileNotFoundError(f"legacy 비교 코드 누락: {legacy_code / name}")
    mat = Path(mat).expanduser().resolve() if mat else None
    if mat is not None and not mat.is_file():
        raise FileNotFoundError(f"MAT 파일 누락: {mat}")
    before = {name: sha256_file(source / name) for name in ARTIFACTS if (source / name).is_file()}
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "runtime_versions": runtime_versions(),
              "input_kind": "provided_mat" if mat else "synthetic_mat",
              "performance_validated": False, "supported_rpms": None,
              "source_sha256": before, "checks": {}}
    report["legacy_code_sha256"] = {
        name: sha256_file(legacy_code / name)
        for name in ("model.py", "web_preprocessing.py", "dae_inference.py")
    }
    try:
        report["rpm_investigation"] = investigate_rpm(source, legacy_code)
        engine = SavedDAEInference(source)
        report["checks"]["weights_structure_and_finite"] = "passed"
        report["statistics"] = {
            "mean_shape": list(engine.mean.shape), "mean_dtype": str(engine.mean.dtype),
            "std_shape": list(engine.std.shape), "std_dtype": str(engine.std.dtype),
            "threshold": engine.threshold,
        }
        report["checks"]["statistics_and_threshold"] = "passed"
        if (source / "band_statistics.npz").is_file():
            with np.load(source / "band_statistics.npz", allow_pickle=False) as data:
                names, means, stds = data["band_names"], data["band_means"], data["band_stds"]
                if (names.tolist() != ["low", "mid", "high"] or means.shape != (3,)
                        or stds.shape != (3,) or not np.isfinite(means).all()
                        or not np.isfinite(stds).all() or np.any(stds <= 0)):
                    raise ValueError("band_statistics.npz 형식 또는 값 오류")
            report["checks"]["optional_band_statistics"] = "passed (not part of inference bundle)"
        else:
            report["checks"]["optional_band_statistics"] = "not provided"
        manifest = convert_legacy_bundle(
            source, output / "bundle", model_version="legacy-dae-local-validation-v1",
            training_provenance="unverified; inspect rpm_investigation in validation report",
        )
        bundled = SavedDAEBundleInference(output / "bundle")
        report["bundle_id"] = manifest["bundle_id"]
        report["checks"]["bundle_conversion_and_load"] = "passed"
        if mat is None:
            mat = output / "synthetic.mat"
            savemat(mat, {"signals": np.random.default_rng(42).normal(size=(6, 65536)), "fs": 25600})
        raw = mat.read_bytes()
        report["mat_sha256"] = sha256_file(mat)
        features = preprocess_mat_bytes(raw)
        result_path = output / "legacy_reference.npz"
        completed = subprocess.run(
            [sys.executable, "-B", "-c", REFERENCE_CODE, str(legacy_code), str(source),
             str(mat), str(result_path)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True,
            text=True, timeout=120,
        )
        if completed.returncode:
            raise RuntimeError("legacy 비교 실행 실패:\n" + completed.stderr[-6000:])
        with np.load(result_path, allow_pickle=False) as reference:
            np.testing.assert_array_equal(reference["features"], features)
            compare_result(reference, engine.predict_bytes(raw))
            # Diagnostic only: invoke the base numerical kernel to compare bundle
            # bytes while keeping its public RPM gate and null RPM metadata intact.
            compare_result(reference, SavedDAEInference.predict_bytes(bundled, raw))
        report["checks"]["legacy_preprocessing_and_numerical_parity"] = "passed (exact equality)"
        try:
            bundled.predict_bytes(raw, rpm=900)
        except ValueError as error:
            if "미확인" not in str(error):
                raise
            report["checks"]["public_inference_unknown_rpm_blocked"] = "passed"
        else:
            raise AssertionError("미확인 RPM 번들이 공개 추론을 허용했습니다.")
        report["status"] = "compatibility_checks_passed_rpm_unconfirmed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        after = {name: sha256_file(source / name) for name in before}
        report["source_unchanged"] = before == after
        if before != after:
            report["status"] = "failed"
            report["error"] = "원본 파일 해시 변경 감지"
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    if not report["source_unchanged"]:
        raise RuntimeError("원본 파일 해시 변경 감지")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "capstone")
    parser.add_argument("--legacy-code", type=Path, default=ROOT / "dae_legacy")
    parser.add_argument("--output", type=Path, default=Path.home() / "dae-validation" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--mat", type=Path, help="실제 MAT 경로. 생략하면 합성 입력으로만 검사")
    args = parser.parse_args()
    try:
        report = validate(args.source, args.legacy_code, args.output, args.mat)
    except Exception as error:
        print(f"FAIL: {error}\n출력 위치(생성된 경우): {args.output}", file=sys.stderr)
        return 1
    print(f"PASS: {report['status']}; input={report['input_kind']}")
    print(f"보고서·번들: {args.output}\n실제 탐지 성능 검증 아님. 지원 RPM 미확인; 공개 추론 차단 유지.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
