"""확인된 정상 파일의 점수로 실험용 Q95 기준을 계산합니다."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = {"autoencoder", "ocsvm", "iforest"}
RPMS = {"900", "1200", "1800"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), set(reader.fieldnames or [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-files", type=int, default=20)
    args = parser.parse_args()

    if args.min_files < 20:
        raise ValueError("최소 정상 파일 수는 20개입니다.")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("기존 보정 결과는 덮어쓰지 않습니다.")

    predictions, pcols = read_csv(Path(args.predictions))
    labels, lcols = read_csv(Path(args.labels))

    if not {
        "file", "rpm", "model", "score", "error", "bundle_id"
    } <= pcols:
        raise ValueError("추론 CSV 열이 부족합니다.")

    if not {"file", "rpm", "condition", "split"} <= lcols:
        raise ValueError("정답 CSV 열이 부족합니다.")

    # 보정 전용 정답 목록
    normal_files = set()

    for row in labels:
        key = (row["file"], row["rpm"])

        if key in normal_files:
            raise ValueError(f"중복 정상 파일: {key}")

        if row["rpm"] not in RPMS:
            raise ValueError(f"지원하지 않는 RPM: {key}")

        if row["condition"] not in {"Normal1", "Normal2"}:
            raise ValueError(f"정상 아닌 조건 포함: {key}")

        if row["split"] != "calibration_normal":
            raise ValueError(
                "보정 데이터는 calibration_normal로 "
                "명시해야 합니다."
            )

        normal_files.add(key)

    if not normal_files:
        raise ValueError("보정용 정상 파일이 없습니다.")

    scores = defaultdict(list)
    bundles = defaultdict(set)
    seen = set()

    for row in predictions:
        key = (row["file"], row["rpm"])
        model = row["model"]

        if row["error"].strip():
            raise ValueError(f"추론 실패 파일: {key}")

        if key not in normal_files:
            raise ValueError(f"정상 확인되지 않은 파일: {key}")

        if model not in MODELS:
            raise ValueError(f"지원하지 않는 모델: {model}")

        unique = (*key, model)

        if unique in seen:
            raise ValueError(f"중복 추론: {unique}")

        seen.add(unique)

        score = float(row["score"])
        if not np.isfinite(score):
            raise ValueError(f"비유한 점수: {unique}")

        if not row["bundle_id"].strip():
            raise ValueError(f"번들 식별값 누락: {unique}")

        scores[(row["rpm"], model)].append(score)
        bundles[(row["rpm"], model)].add(row["bundle_id"])

    expected = {
        (*key, model)
        for key in normal_files
        for model in MODELS
    }

    if seen != expected:
        raise ValueError(
            f"정상 파일의 모델 추론 누락: {len(expected - seen)}"
        )

    results = []

    for (rpm, model), values in sorted(scores.items()):
        if len(values) < args.min_files:
            raise ValueError(
                f"{rpm} RPM {model}: 정상 파일 부족 "
                f"({len(values)}/{args.min_files})"
            )

        if len(bundles[(rpm, model)]) != 1:
            raise ValueError("동일 RPM·모델에 여러 번들이 섞였습니다.")

        results.append({
            "rpm": int(rpm),
            "model": model,
            "normal_count": len(values),
            "calibrated_q95": float(np.percentile(values, 95)),
            "bundle_id": next(iter(bundles[(rpm, model)])),
            "status": "experimental_only",
        })

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as f:
        json.dump(
            {
                "method": "confirmed_normal_calibration_q95",
                "minimum_files": args.min_files,
                "results": results,
                "limitations": [
                    "실험용 정상 기준이며 운영 임계값을 변경하지 않음",
                    "보정 파일과 평가 파일은 서로 분리해야 함",
                    "물리적 Run 독립성은 별도 확인 필요",
                    "결함 탐지율은 이 결과만으로 검증되지 않음",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("===== 정상 기준 보정 완료 =====")

    for r in results:
        print(
            f"{r['rpm']} RPM | {r['model']:<12} "
            f"정상={r['normal_count']} "
            f"Q95={r['calibrated_q95']:.6f}"
        )

    print("\n실험 결과:", output)
    print("기존 AE 모델·운영 임계값 변경 없음")


if __name__ == "__main__":
    main()
