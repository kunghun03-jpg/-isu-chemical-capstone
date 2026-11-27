"""P3 실험용 정상 기준의 이식 가능한 배포 패키지 생성."""

import hashlib
import json
import tempfile

from pathlib import Path

import numpy as np

from vibration.physical_inference import (
    SUPPORTED_DATES,
    SUPPORTED_RPMS,
    METHOD,
    FEATURE_FLOOR,
    SCALE_FLOOR,
    load_physical_reference,
)

from vibration.physical_features import FEATURE_NAMES


SOURCE = Path("outputs/experimental_physical_features")
OUTPUT = Path("artifacts/physical_p3")


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def write_json(path, data):

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main():

    if OUTPUT.exists():

        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "schema_version": 1,
        "method": METHOD,
        "evaluation_status": "exploratory_only",
        "approved_for_production": False,
        "run_independence_verified": False,
        "datasets": [],
        "files": {},
    }

    with tempfile.TemporaryDirectory(
        dir=OUTPUT.parent
    ) as temp:

        staging = Path(temp) / "physical_p3"
        staging.mkdir()

        print("===== Export P3 references =====")

        for date in SUPPORTED_DATES:

            for rpm in SUPPORTED_RPMS:

                reference = load_physical_reference(
                    date,
                    rpm,
                    reference_root=SOURCE,
                )

                source_folder = (
                    SOURCE / date / f"{rpm}rpm"
                )

                source_paths = [
                    source_folder
                    / "C_mixed_per_frequency_reference.npz",
                    source_folder
                    / "experiment_metadata.json",
                ]

                source_hashes = {
                    p: sha256(p)
                    for p in source_paths
                }

                folder = (
                    staging / date / f"{rpm}rpm"
                )

                folder.mkdir(parents=True)

                reference_path = (
                    folder
                    / "C_mixed_per_frequency_reference.npz"
                )

                metadata_path = (
                    folder
                    / "experiment_metadata.json"
                )

                np.savez_compressed(
                    reference_path,
                    center=reference.center,
                    scale=reference.scale,
                    feature_names=np.array(
                        FEATURE_NAMES
                    ),
                )

                # 배포에 필요한 설정만 저장한다.
                # 원본 데이터 경로는 복사하지 않는다.

                metadata = {
                    "date": date,
                    "rpm": rpm,
                    "feature_names": list(
                        FEATURE_NAMES
                    ),
                    "feature_transform": "log",
                    "file_aggregation":
                        "median_of_8_segments",
                    "normal_reference": "median_mad",
                    "feature_floor": FEATURE_FLOOR,
                    "scale_floor": SCALE_FLOOR,
                    "thresholds": {
                        METHOD: reference.threshold
                    },
                    "threshold_source":
                        "normal_validation_q95",
                    "evaluation_status":
                        "exploratory_only",
                    "approved_for_production": False,
                    "run_independence_verified": False,
                    "test_previously_examined": True,
                }

                write_json(
                    metadata_path,
                    metadata,
                )

                # 원본 기준과 배포용 기준 일치 검증

                exported = load_physical_reference(
                    date,
                    rpm,
                    reference_root=staging,
                )

                np.testing.assert_array_equal(
                    exported.center,
                    reference.center,
                )

                np.testing.assert_array_equal(
                    exported.scale,
                    reference.scale,
                )

                if (
                    exported.threshold
                    != reference.threshold
                ):

                    raise ValueError(
                        "Threshold mismatch"
                    )

                for path in (
                    reference_path,
                    metadata_path,
                ):

                    key = path.relative_to(
                        staging
                    ).as_posix()

                    manifest["files"][key] = (
                        sha256(path)
                    )

                for path, original_hash in (
                    source_hashes.items()
                ):

                    if sha256(path) != original_hash:

                        raise ValueError(
                            "Source reference changed"
                        )

                manifest["datasets"].append({
                    "date": date,
                    "rpm": rpm,
                    "threshold": reference.threshold,
                })

                print(
                    date,
                    rpm,
                    "Reference parity: PASS",
                )

        if (
            len(manifest["datasets"]) != 9
            or len(manifest["files"]) != 18
        ):

            raise ValueError(
                "Unexpected package size"
            )

        write_json(
            staging / "manifest.json",
            manifest,
        )

        if OUTPUT.exists():

            raise FileExistsError(OUTPUT)

        staging.rename(OUTPUT)

    print("\n===== Final verification =====")
    print("Reference datasets: 9/9")
    print("Reference files: 9")
    print("Metadata files: 9")
    print("Manifest files: 1")
    print("Reference parity: PASS")
    print("Source reference changes: NONE")
    print("Production approval: FALSE")
    print("Output:", OUTPUT)
    print("P3 reference export: PASS")


if __name__ == "__main__":
    main()
