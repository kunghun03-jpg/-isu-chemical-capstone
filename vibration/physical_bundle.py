"""P3 배포 패키지 검증 및 실험용 추론."""

import hashlib
import hmac
import json

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from vibration.physical_inference import (
    SUPPORTED_DATES,
    SUPPORTED_RPMS,
    METHOD,
    PhysicalInferenceError,
    validate_condition,
    load_physical_reference,
    analyze_physical_mat,
)


DEFAULT_BUNDLE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "physical_p3"
)

REFERENCE_FILE = (
    "C_mixed_per_frequency_reference.npz"
)

METADATA_FILE = "experiment_metadata.json"


def digest(data):

    return hashlib.sha256(data).hexdigest()


def require(condition, message):

    if not condition:

        raise PhysicalInferenceError(message)


def verified_manifest(
    root,
    expected_manifest_sha256,
):

    try:

        raw = (
            root / "manifest.json"
        ).read_bytes()

        require(
            isinstance(
                expected_manifest_sha256, str
            )
            and hmac.compare_digest(
                digest(raw),
                expected_manifest_sha256,
            ),
            "P3 manifest integrity mismatch",
        )

        manifest = json.loads(raw)

        expected = {
            "schema_version": 1,
            "method": METHOD,
            "evaluation_status":
                "exploratory_only",
            "approved_for_production": False,
            "run_independence_verified": False,
        }

        for key, value in expected.items():

            require(
                manifest.get(key) == value,
                f"Manifest mismatch: {key}",
            )

        conditions = {
            (date, rpm)
            for date in SUPPORTED_DATES
            for rpm in SUPPORTED_RPMS
        }

        entries = manifest["datasets"]

        require(
            len(entries) == 9
            and {
                (item["date"], item["rpm"])
                for item in entries
            } == conditions,
            "Invalid manifest dataset inventory",
        )

        expected_files = {
            f"{date}/{rpm}rpm/{name}"
            for date, rpm in conditions
            for name in (
                REFERENCE_FILE,
                METADATA_FILE,
            )
        }

        require(
            set(manifest["files"])
            == expected_files,
            "Invalid manifest file inventory",
        )

        return manifest

    except PhysicalInferenceError:
        raise

    except Exception as exc:

        raise PhysicalInferenceError(
            f"Invalid P3 manifest: {exc}"
        ) from exc


def verified_file(root, manifest, key):

    try:

        raw = (root / key).read_bytes()

        expected = manifest["files"][key]

        require(
            isinstance(expected, str)
            and hmac.compare_digest(
                digest(raw), expected
            ),
            f"P3 file integrity mismatch: {key}",
        )

        return raw

    except PhysicalInferenceError:
        raise

    except Exception as exc:

        raise PhysicalInferenceError(
            f"P3 bundle file error: {key}"
        ) from exc


def verify_bundle(
    expected_manifest_sha256,
    bundle_root=None,
):

    root = (
        DEFAULT_BUNDLE_ROOT
        if bundle_root is None
        else Path(bundle_root)
    )

    manifest = verified_manifest(
        root,
        expected_manifest_sha256,
    )

    for key in manifest["files"]:

        verified_file(root, manifest, key)

    return {
        "datasets": 9,
        "verified_files": 18,
        "manifest_verified": True,
        "evaluation_status": "exploratory_only",
        "approved_for_production": False,
    }


def analyze_bundled_physical_mat(
    mat_source,
    date,
    rpm,
    expected_manifest_sha256,
    bundle_root=None,
):

    validate_condition(date, rpm)

    root = (
        DEFAULT_BUNDLE_ROOT
        if bundle_root is None
        else Path(bundle_root)
    )

    manifest = verified_manifest(
        root,
        expected_manifest_sha256,
    )

    prefix = f"{date}/{rpm}rpm"

    verified = {
        name: verified_file(
            root,
            manifest,
            f"{prefix}/{name}",
        )
        for name in (
            REFERENCE_FILE,
            METADATA_FILE,
        )
    }

    entry = next(
        item
        for item in manifest["datasets"]
        if item["date"] == date
        and item["rpm"] == rpm
    )

    # 검증한 바이트만 사용한다.
    # 검증 후 원본 파일을 다시 읽지 않는다.

    with TemporaryDirectory() as temp:

        temp_root = Path(temp)

        folder = (
            temp_root / date / f"{rpm}rpm"
        )

        folder.mkdir(parents=True)

        for name, raw in verified.items():

            (folder / name).write_bytes(raw)

        reference = load_physical_reference(
            date,
            rpm,
            reference_root=temp_root,
        )

        require(
            reference.threshold
            == entry["threshold"],
            "Manifest threshold mismatch",
        )

        if isinstance(
            mat_source, (bytes, bytearray)
        ):

            mat_source = BytesIO(mat_source)

        result = analyze_physical_mat(
            mat_source,
            date,
            rpm,
            reference_root=temp_root,
        )

    result.update({
        "bundle_integrity_verified": True,
        "manifest_sha256":
            expected_manifest_sha256,
        "reference_source":
            "physical_p3_bundle",
    })

    return result
