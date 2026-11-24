"""Build experimental RF features from the exact EDA-selected MAT files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_fault_signatures import metadata
from scripts.experiment_rf_features import EXTRA_FEATURES, extract_extra_features
from vibration.web_inference import preview_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eda_dir = args.eda_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if output.exists():
        raise FileExistsError(f"새 출력 파일을 지정하세요: {output}")

    with (eda_dir / "file_features.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames
        rows = list(reader)

    if not original_fields:
        raise ValueError("EDA CSV 헤더가 없습니다.")

    if any(name in original_fields for name in EXTRA_FEATURES):
        raise ValueError("추가 특징 이름이 기존 CSV와 중복됩니다.")

    output.parent.mkdir(parents=True, exist_ok=True)

    extended_rows = []

    for index, row in enumerate(rows, 1):
        path = Path(row["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        info = metadata(path)

        for key in ("date", "condition", "family"):
            if str(info[key]) != str(row[key]):
                raise ValueError(f"{path}: {key} 불일치")

        if int(info["rpm"]) != int(row["rpm"]):
            raise ValueError(f"{path}: RPM 불일치")

        raw = path.read_bytes()
        loaded = preview_mat(raw)

        if loaded["fs"] != 25600 or loaded["samples"] != 65536:
            raise ValueError(f"{path}: 신호 규격 불일치")

        features = extract_extra_features(
            loaded["signal"],
            loaded["fs"],
            int(row["rpm"]),
        )

        if not all(np.isfinite(v) and v >= 0 for v in features.values()):
            raise ValueError(f"{path}: 추가 특징에 비정상 값이 있습니다.")

        extended_rows.append({**row, **features})

        if index % 100 == 0 or index == len(rows):
            print(f"진행: {index}/{len(rows)}", flush=True)

    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[*original_fields, *EXTRA_FEATURES],
        )
        writer.writeheader()
        writer.writerows(extended_rows)

    print(f"완료: {output}")
    print(f"파일 수: {len(extended_rows)}")
    print(f"추가 특징: {', '.join(EXTRA_FEATURES)}")


if __name__ == "__main__":
    main()
