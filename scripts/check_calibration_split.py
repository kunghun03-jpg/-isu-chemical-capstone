"""정상 기준 보정과 평가 데이터의 파일 중복 검사."""

import argparse
import csv
from pathlib import Path


def load_labels(path):
    with Path(path).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = {"file", "rpm", "condition", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("정답 CSV 필수 열 누락")

        rows = list(reader)

    keys = [(r["file"], r["rpm"]) for r in rows]

    if len(keys) != len(set(keys)):
        raise ValueError("CSV 내부 중복 파일 발견")

    return rows, set(keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()

    calibration, calibration_keys = load_labels(
        args.calibration
    )
    evaluation, evaluation_keys = load_labels(
        args.evaluation
    )

    if not calibration or not evaluation:
        raise ValueError("보정 또는 평가 데이터가 비어 있습니다.")

    for row in calibration:
        if (
            row["split"] != "calibration_normal"
            or row["condition"] not in ("Normal1", "Normal2")
        ):
            raise ValueError("보정 데이터에 비정상 조건 포함")

    for row in evaluation:
        if row["split"] not in ("test_normal", "test_anomaly"):
            raise ValueError("평가 데이터의 split 오류")

        normal = row["condition"] in ("Normal1", "Normal2")
        if normal != (row["split"] == "test_normal"):
            raise ValueError("평가 데이터의 정답 조건 불일치")

    overlap = calibration_keys & evaluation_keys

    print("===== 보정·평가 분리 검사 =====")
    print("보정 파일:", len(calibration_keys))
    print("평가 파일:", len(evaluation_keys))
    print("중복 파일:", len(overlap))

    if overlap:
        for item in sorted(overlap)[:5]:
            print("중복:", item)
        raise ValueError("보정·평가 파일 중복 발견")

    print("파일 경로 기준 분리 검사 통과")
    print("실제 내용 중복과 물리적 Run 독립성은 별도 확인 필요")


if __name__ == "__main__":
    main()
