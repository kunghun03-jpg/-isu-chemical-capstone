"""정상 기준 보정 데이터의 기존 분할 포함 여부 검사."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def load_csv(path, required):
    with Path(path).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path}: 필수 열 누락 "
                f"{required - set(reader.fieldnames or [])}"
            )
        return list(reader)


def normalize_path(value):
    return value.replace("\\", "/").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    calibration = load_csv(
        args.calibration,
        {"file", "rpm", "condition", "split"},
    )
    reference = load_csv(
        args.reference,
        {"file", "rpm", "condition", "split"},
    )

    if not calibration or not reference:
        raise ValueError("입력 데이터가 비어 있습니다.")

    lookup = {}
    for row in reference:
        key = (
            normalize_path(row["file"]),
            row["rpm"].strip(),
        )
        if key in lookup:
            raise ValueError(f"기존 분할에 중복 파일: {key}")
        lookup[key] = row

    counts = Counter()
    missing = []
    seen = set()

    for row in calibration:
        key = (
            normalize_path(row["file"]),
            row["rpm"].strip(),
        )

        if key in seen:
            raise ValueError(f"보정 목록 내부 중복: {key}")
        seen.add(key)

        if row["split"] != "calibration_normal":
            raise ValueError("보정 데이터 split 오류")

        if row["condition"] not in {"Normal1", "Normal2"}:
            raise ValueError("정상 아닌 보정 파일 발견")

        original = lookup.get(key)
        if original is None:
            missing.append(key)
            continue

        if original["condition"] != row["condition"]:
            raise ValueError(f"조건 불일치: {key}")

        counts[original["split"]] += 1

    print("===== 정상 기준 보정 데이터 출처 검사 =====")
    print("보정 파일 수:", len(calibration))

    for split, count in sorted(counts.items()):
        print(f"기존 {split}: {count}개")

    print("기존 분할에서 확인되지 않은 파일:", len(missing))

    if missing:
        for key in missing[:5]:
            print("미확인:", key)
        raise ValueError(
            "출처를 확인할 수 없는 보정 파일이 있습니다."
        )

    if counts.get("test_normal", 0) or counts.get(
        "test_anomaly", 0
    ):
        raise ValueError(
            "기존 Test 파일을 보정에 사용할 수 없습니다."
        )

    if counts.get("train", 0) or counts.get(
        "validation", 0
    ) or counts.get("val", 0):
        print(
            "판정: 개발용 탐색 실험만 허용"
        )
        print(
            "기존 모델의 독립 성능 검증으로 사용 금지"
        )
    else:
        print("판정: 분할 출처 확인 완료")

    print("물리적 Run 독립성은 확인되지 않았습니다.")
    print("기존 모델 및 운영 임계값 변경 없음")


if __name__ == "__main__":
    main()
