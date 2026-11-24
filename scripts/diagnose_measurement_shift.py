"""연속 측정 파일의 이상 점수 분포 변화를 탐색적으로 분석한다."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = {"autoencoder", "ocsvm", "iforest"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    if args.window < 5:
        raise ValueError("window는 최소 5개여야 합니다.")

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
            raise ValueError("필수 CSV 열 누락")

        rows = list(reader)

    if not rows:
        raise ValueError("분석할 데이터가 없습니다.")

    groups = defaultdict(dict)

    for row in rows:
        if row["error"].strip():
            raise ValueError(f"추론 실패: {row['file']}")

        rpm = row["rpm"].strip()
        model = row["model"].strip()
        filename = Path(row["file"]).name

        if rpm not in {"900", "1200", "1800"}:
            raise ValueError(f"지원하지 않는 RPM: {rpm}")

        if model not in MODELS:
            raise ValueError(f"지원하지 않는 모델: {model}")

        if not re.fullmatch(
            r"\d{8}_\d{6}_Vib\.mat", filename
        ):
            raise ValueError(
                f"측정 시각을 확인할 수 없는 파일: {filename}"
            )

        key = (rpm, model)

        if filename in groups[key]:
            raise ValueError(f"중복 예측: {key}, {filename}")

        score = float(row["score"])
        threshold = float(row["threshold"])
        predicted = int(row["predicted_anomaly"])

        if not np.isfinite(score) or not np.isfinite(threshold):
            raise ValueError(f"유효하지 않은 점수: {filename}")

        if predicted not in (0, 1):
            raise ValueError(f"판정 형식 오류: {filename}")

        if predicted != int(score > threshold):
            raise ValueError(f"판정 불일치: {filename}")

        groups[key][filename] = {
            "score": score,
            "alarm": predicted,
        }

    # RPM별 세 모델이 동일한 파일을 분석했는지 확인
    rpms = sorted({rpm for rpm, _ in groups})
    results = []

    for rpm in rpms:
        model_groups = {
            model: groups.get((rpm, model), {})
            for model in MODELS
        }

        if any(not files for files in model_groups.values()):
            raise ValueError(f"{rpm} RPM 모델 결과 누락")

        names = sorted(model_groups["autoencoder"])

        for model, files in model_groups.items():
            if sorted(files) != names:
                raise ValueError(
                    f"{rpm} RPM 모델별 파일 목록 불일치"
                )

        if len(names) < args.window * 2:
            raise ValueError(
                f"{rpm} RPM: 비교할 파일 부족"
            )

        for model in sorted(MODELS):
            files = model_groups[model]

            scores = np.array([
                files[name]["score"] for name in names
            ])

            alarms = np.array([
                files[name]["alarm"] for name in names
            ])

            w = args.window

            early_scores = scores[:w]
            late_scores = scores[-w:]

            early_median = float(np.median(early_scores))
            late_median = float(np.median(late_scores))

            delta = late_median - early_median

            if delta < 0:
                direction = "decreased"
            elif delta > 0:
                direction = "increased"
            else:
                direction = "unchanged"

            results.append({
                "rpm": int(rpm),
                "model": model,
                "total_files": len(names),
                "window_size": w,
                "early_first_file": names[0],
                "early_last_file": names[w - 1],
                "late_first_file": names[-w],
                "late_last_file": names[-1],
                "early_median": early_median,
                "late_median": late_median,
                "median_delta": float(delta),
                "score_direction": direction,
                "early_alarm_count": int(alarms[:w].sum()),
                "late_alarm_count": int(alarms[-w:].sum()),
                "status": "exploratory_diagnostic",
            })

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "analysis": "measurement_score_shift",
            "results": results,
            "limitations": [
                "앞쪽과 뒤쪽 구간의 점수 차이만 분석",
                "변화 원인이나 측정 상태 불안정 확정 불가",
                "정상 여부가 확인되지 않으면 오탐으로 해석 금지",
                "서로 다른 측정 Run을 연결하면 해석 불가",
                "기존 모델 판정 및 임계값 변경 없음"
            ]
        }, f, ensure_ascii=False, indent=2)

    print("===== 측정 상태 변화 탐색 진단 =====")

    for r in results:
        print(
            f"{r['rpm']} RPM | {r['model']:<12} "
            f"점수 변화={r['score_direction']:<9} "
            f"중앙값 차이={r['median_delta']:+.6f} "
            f"경보={r['early_alarm_count']}"
            f"→{r['late_alarm_count']}"
        )

    print("결과 저장:", output)
    print("기존 모델 및 임계값 변경 없음")


if __name__ == "__main__":
    main()
