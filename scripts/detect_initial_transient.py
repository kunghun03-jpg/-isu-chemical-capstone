"""시간순 정상 데이터에서 초기 이상 판정 집중 현상을 검사한다."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--early-count", type=int, default=20)
    args = parser.parse_args()

    if args.early_count < 1:
        raise ValueError("early-count는 1 이상이어야 합니다.")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    with Path(args.predictions).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        required = {
            "file", "rpm", "model", "score",
            "threshold", "predicted_anomaly", "error"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("추론 CSV 필수 열 누락")
        rows = list(reader)

    groups = defaultdict(dict)

    for row in rows:
        if row["error"].strip():
            raise ValueError(f"추론 오류: {row['file']}")

        key = (row["rpm"], row["model"])
        filename = Path(row["file"]).name

        if filename in groups[key]:
            raise ValueError(f"중복 파일: {key}, {filename}")

        score = float(row["score"])
        threshold = float(row["threshold"])
        predicted = int(row["predicted_anomaly"])

        if not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError("유효하지 않은 점수 또는 임계값")

        if predicted not in (0, 1):
            raise ValueError("판정 값 오류")

        if predicted != int(score > threshold):
            raise ValueError(f"판정 불일치: {filename}")

        groups[key][filename] = {
            "score": score,
            "predicted": predicted,
        }

    if not groups:
        raise ValueError("추론 데이터가 없습니다.")

    results = []
    rpm_files = {}

    for (rpm, model), files in sorted(groups.items()):
        names = sorted(files)

        if rpm in rpm_files and rpm_files[rpm] != names:
            raise ValueError(
                f"{rpm} RPM 모델별 파일 목록 불일치"
            )
        rpm_files[rpm] = names

        if len(names) <= args.early_count:
            raise ValueError("후반 분석 파일이 부족합니다.")

        early_names = names[:args.early_count]
        late_names = names[args.early_count:]

        early = [files[n] for n in early_names]
        late = [files[n] for n in late_names]

        early_fp = sum(r["predicted"] for r in early)
        late_fp = sum(r["predicted"] for r in late)

        early_median = float(
            np.median([r["score"] for r in early])
        )
        late_median = float(
            np.median([r["score"] for r in late])
        )

        concentrated = (
            early_fp > 0
            and late_fp == 0
            and early_median > late_median
        )

        results.append({
            "rpm": int(rpm),
            "model": model,
            "early_count": len(early),
            "late_count": len(late),
            "early_alarms": early_fp,
            "late_alarms": late_fp,
            "early_score_median": early_median,
            "late_score_median": late_median,
            "initial_alarm_concentration": concentrated,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump({
            "analysis_type": "exploratory_initial_transient",
            "results": results,
            "limitations": [
                "초기 구간 길이는 사전에 지정한 분석 기준",
                "정상으로 확인된 파일에서만 오탐률로 해석 가능",
                "조립·분해 또는 안정화 원인 확정 불가",
                "결함 탐지율 및 독립 Run 성능 미검증",
                "기존 모델 판정과 임계값 변경 없음"
            ],
        }, f, ensure_ascii=False, indent=2)

    print("===== 초기 구간 이상 판정 집중 검사 =====")

    for r in results:
        print(
            f"{r['rpm']} RPM | {r['model']:<12} "
            f"초반={r['early_alarms']}/{r['early_count']} "
            f"후반={r['late_alarms']}/{r['late_count']} "
            f"초기 집중={r['initial_alarm_concentration']}"
        )

    print("결과 저장:", output)


if __name__ == "__main__":
    main()
