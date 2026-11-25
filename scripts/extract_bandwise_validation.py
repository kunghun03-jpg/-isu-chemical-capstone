"""Extract experimental band-wise AE scores from Normal Validation."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from vibration.bandwise_ae import BandwiseAE


ROOT = Path(__file__).resolve().parents[1]
VERSION = "date_prelim_20261106"
RPMS = (900, 1200, 1800)

OUTPUT_DIR = (
    ROOT / "outputs/experimental_bandwise_ae"
)

BANDS = (
    "1X",
    "2X",
    "0_100Hz",
    "100_1000Hz",
    "1000_10000Hz",
)


def path_key(value):
    return str(value).strip().replace("\\", "/")


def load_validation_files(rpm):

    root = ROOT / f"outputs/{rpm}rpm" / VERSION

    path = root / "validation_normalized.npz"

    with np.load(path, allow_pickle=False) as data:

        files = data["source_file"].astype(str)
        conditions = data["condition"].astype(str)
        segments = data["segment_id"]
        dates = data["date"].astype(str)
        rpms = data["rpm"].astype(str)

        if data["X"].shape != (len(files), 1, 3201):
            raise ValueError("Unexpected input shape")

        if not all(
            len(arr) == len(files)
            for arr in (
                conditions, segments, dates, rpms
            )
        ):
            raise ValueError("Metadata length mismatch")

    grouped = {}

    for i, file in enumerate(files):

        if conditions[i] not in ("Normal1", "Normal2"):
            raise ValueError(
                f"Non-normal condition: {conditions[i]}"
            )

        if dates[i] != "20260916":
            raise ValueError(
                f"Unexpected validation date: {dates[i]}"
            )

        if rpms[i] != str(rpm):
            raise ValueError("RPM mismatch")

        key = path_key(file)

        item = grouped.setdefault(
            key,
            {
                "condition": conditions[i],
                "segments": [],
                "date": dates[i],
            },
        )

        if item["condition"] != conditions[i]:
            raise ValueError("Condition mismatch")

        item["segments"].append(int(segments[i]))

    for file, item in grouped.items():

        ids = sorted(item["segments"])

        if ids not in (
            list(range(8)),
            list(range(1, 9)),
        ):
            raise ValueError(
                f"Invalid segment IDs: {file}: {ids}"
            )

    evaluation_path = (
        root / f"split_{rpm}rpm.csv"
    )

    reference = {}

    with evaluation_path.open(
        encoding="utf-8-sig", newline=""
    ) as f:

        for row in csv.DictReader(f):

            split = row["split"].strip().lower()

            if split not in (
                "validation",
                "val",
                "validation_normal",
                "normal_validation",
            ):
                continue

            key = path_key(row["file"])

            if key in reference:
                raise ValueError(
                    f"Duplicate reference file: {key}"
                )

            if row["condition"] not in (
                "Normal1", "Normal2"
            ):
                raise ValueError(
                    "Unexpected reference condition"
                )

            reference[key] = row

    if set(grouped) != set(reference):

        raise ValueError(
            f"{rpm} RPM Validation file mismatch: "
            f"NPZ={len(grouped)}, "
            f"CSV={len(reference)}, "
            f"NPZ-only={len(set(grouped)-set(reference))}, "
            f"CSV-only={len(set(reference)-set(grouped))}"
        )

    for file, item in grouped.items():

        row = reference[file]

        if row["condition"] != item["condition"]:
            raise ValueError("Reference condition mismatch")

        if row["date"].strip() != "20260916":
            raise ValueError("Reference date mismatch")

        if row["rpm"].strip() != str(rpm):
            raise ValueError("Reference RPM mismatch")

        source = Path(file)

        if not source.is_absolute():
            source = ROOT / source

        if not source.is_file():
            raise FileNotFoundError(source)

        item["path"] = source

    print(
        f"{rpm} RPM: "
        f"{len(grouped)} files, "
        f"{len(files)} segments"
    )

    print(
        "Conditions:",
        dict(
            Counter(
                item["condition"]
                for item in grouped.values()
            )
        ),
    )

    return grouped


def extract_scores(rpm, files):

    model = BandwiseAE(rpm)

    rows = []

    for index, (file, item) in enumerate(
        sorted(files.items()), start=1
    ):

        result = model.score_file(item["path"])

        overall = result["overall_mse"]
        if not np.isfinite(overall) or overall < 0:
            raise ValueError("Invalid overall MSE")

        row = {
            "rpm": rpm,
            "file": file,
            "condition": item["condition"],
            "date": item["date"],
            "split": "validation",
            "segment_count": 8,
            "bundle_id": result["bundle_id"],
            "overall_mse": overall,
            "reference_mse_verified": False,
            "run_independence_verified": False,
            "split_mode": "date_preliminary",
            "evaluation_status": "experimental_only",
        }

        for band in BANDS:

            mse = result["bands"][band]["mse"]

            if not np.isfinite(mse) or mse < 0:
                raise ValueError(
                    f"Invalid band MSE: {band}"
                )

            row[f"{band}_mse"] = mse

        rows.append(row)

        if index % 20 == 0 or index == len(files):

            print(
                f"[{rpm} RPM] "
                f"{index}/{len(files)} extracted",
                flush=True,
            )

    return rows


def main():

    print("===== Validation preflight =====")

    validation_sets = {
        rpm: load_validation_files(rpm)
        for rpm in RPMS
    }

    all_files = [
        item["path"].resolve()
        for files in validation_sets.values()
        for item in files.values()
    ]

    if len(all_files) != len(set(all_files)):
        raise ValueError("Duplicate source files")

    all_rows = []

    print("\n===== Bandwise extraction =====")

    for rpm in RPMS:
        all_rows.extend(
            extract_scores(rpm, validation_sets[rpm])
        )

    if len(all_rows) != len(all_files):
        raise ValueError("Missing extracted files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "validation_band_scores.csv"
    temporary = output.with_suffix(".csv.tmp")

    with temporary.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0])
        )

        writer.writeheader()
        writer.writerows(all_rows)

    temporary.replace(output)

    print("\n===== Extraction completed =====")
    print("Total files:", len(all_rows))

    for rpm in RPMS:

        rows = [
            row for row in all_rows
            if row["rpm"] == rpm
        ]

        print(f"{rpm} RPM: {len(rows)} files")

    print("Stored Validation score parity: unavailable")
    print("Run independence verified: False")

    print("Saved:", output)
    print("Operational model changed: False")


if __name__ == "__main__":
    main()
