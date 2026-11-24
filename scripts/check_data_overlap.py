"""새 MAT 파일과 기존 실험 파일의 내용 중복 검사."""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()

    if not input_dir.is_dir():
        raise ValueError("입력 폴더가 없습니다.")

    incoming = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".mat"
    )

    if not incoming:
        raise ValueError("검사할 MAT 파일이 없습니다.")

    # 기존 실험의 전체 분할 파일 목록
    references = set()

    for rpm in (900, 1200, 1800):
        path = ROOT / (
            f"outputs/{rpm}rpm/date_prelim_20261106/"
            f"split_{rpm}rpm.csv"
        )

        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] in (
                    "train", "validation",
                    "test_normal", "test_anomaly"
                ):
                    references.add(row["file"])

    # 크기가 같은 파일만 해시 비교 대상으로 사용
    size_index = defaultdict(list)
    missing = 0

    for name in sorted(references):
        path = ROOT / name
        if path.is_file():
            size_index[path.stat().st_size].append(path)
        else:
            missing += 1

    print("===== 새 데이터 중복 검사 =====")
    print("입력 MAT:", len(incoming))
    print("기존 참조 파일:", len(references))
    print("기존 참조 파일 누락:", missing)

    if missing:
        raise ValueError("기존 참조 파일이 누락되어 검사 불완전")

    hash_cache = {}
    overlaps = 0
    incoming_hashes = {}
    internal_duplicates = 0

    for path in incoming:
        digest = sha256(path)

        if digest in incoming_hashes:
            internal_duplicates += 1
            print(
                "[입력 내부 중복]",
                path.name,
                "↔",
                incoming_hashes[digest].name
            )
        else:
            incoming_hashes[digest] = path

        matches = []

        for reference in size_index[path.stat().st_size]:
            if reference not in hash_cache:
                hash_cache[reference] = sha256(reference)

            if hash_cache[reference] == digest:
                matches.append(reference)

        if matches:
            overlaps += 1
            print("\n[기존 데이터와 중복]", path)
            for match in matches[:3]:
                print("  기존:", match.relative_to(ROOT))

    print("\n===== 검사 결과 =====")
    print("입력 파일 수:", len(incoming))
    print("기존 데이터와 동일한 파일:", overlaps)
    print("입력 내부 중복 파일:", internal_duplicates)

    if overlaps or internal_duplicates:
        print("결론: 독립 데이터로 간주하면 안 되는 파일 발견")
    else:
        print("결론: 파일 내용 중복은 발견되지 않음")

    print("물리적 Run 독립성은 이 검사로 확인할 수 없음")


if __name__ == "__main__":
    main()
