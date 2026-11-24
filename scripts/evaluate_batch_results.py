"""MAT 일괄 추론 결과를 독립적인 정답 CSV와 비교합니다."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

MODELS = {"autoencoder", "ocsvm", "iforest"}
SPLITS = {"test_normal", "test_anomaly"}
FIELDS = [
    "rpm", "model", "normal_count", "anomaly_count",
    "fpr", "recall", "auroc", "auprc"
]


def read_csv(path):
    with Path(path).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV 열 이름 없음: {path}")
        return list(reader), set(reader.fieldnames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()

    if output.exists():
        raise FileExistsError("기존 평가 결과는 덮어쓰지 않습니다.")

    predictions, pcols = read_csv(args.predictions)
    labels, lcols = read_csv(args.labels)

    required_p = {
        "file", "rpm", "model", "score",
        "threshold", "predicted_anomaly", "error"
    }
    required_l = {"file", "rpm", "condition", "split"}

    if not required_p <= pcols or not required_l <= lcols:
        raise ValueError("필수 CSV 열이 누락되었습니다.")

    # 정답은 입력 폴더 기준 상대경로와 정확히 일치해야 합니다.
    label_map = {}

    for row in labels:
        key = (row["file"], row["rpm"])

        if key in label_map:
            raise ValueError(f"중복 정답: {key}")

        if row["split"] not in SPLITS:
            raise ValueError(f"시험 분할이 아닌 정답: {key}")

        if row["rpm"] not in {"900", "1200", "1800"}:
            raise ValueError(f"지원하지 않는 RPM: {key}")

        if not row["condition"].strip():
            raise ValueError(f"측정 조건 누락: {key}")

        is_normal = row["condition"] in {"Normal1", "Normal2"}

        if is_normal != (row["split"] == "test_normal"):
            raise ValueError(f"정상/결함 정답 불일치: {key}")

        label_map[key] = row

    if not label_map:
        raise ValueError("정답 데이터가 없습니다.")

    records = defaultdict(list)
    seen = set()

    for row in predictions:
        if row["error"].strip():
            raise ValueError(
                f"분석 실패 파일이 있어 평가를 중단합니다: {row['file']}"
            )

        key = (row["file"], row["rpm"])
        model = row["model"]

        if key not in label_map:
            raise ValueError(f"정답에 없는 추론 파일: {key}")

        if model not in MODELS:
            raise ValueError(f"알 수 없는 모델: {model}")

        unique = (*key, model)
        if unique in seen:
            raise ValueError(f"중복 추론 결과: {unique}")
        seen.add(unique)

        score = float(row["score"])
        threshold = float(row["threshold"])
        predicted = int(row["predicted_anomaly"])

        if not np.isfinite([score, threshold]).all():
            raise ValueError(f"유효하지 않은 점수: {unique}")

        if predicted not in (0, 1):
            raise ValueError(f"유효하지 않은 판정: {unique}")

        if predicted != int(score > threshold):
            raise ValueError(f"점수와 판정 불일치: {unique}")

        truth = label_map[key]
        actual = int(truth["split"] == "test_anomaly")

        records[(int(row["rpm"]), model)].append(
            (actual, score, predicted)
        )

    # 모든 정답 파일에 세 모델의 추론 결과가 있어야 합니다.
    expected = {
        (file, rpm, model)
        for file, rpm in label_map
        for model in MODELS
    }

    if seen != expected:
        missing = expected - seen
        raise ValueError(
            f"평가 파일 누락 또는 모델 결과 누락: {len(missing)}개"
        )

    output_rows = []

    for (rpm, model), values in sorted(records.items()):
        array = np.asarray(values, dtype=float)
        y = array[:, 0].astype(int)
        scores = array[:, 1]
        pred = array[:, 2].astype(int)

        if set(y.tolist()) != {0, 1}:
            raise ValueError(
                f"{rpm} RPM: 정상과 결함이 모두 필요합니다."
            )

        result = {
            "rpm": rpm,
            "model": model,
            "normal_count": int((y == 0).sum()),
            "anomaly_count": int((y == 1).sum()),
            "fpr": float(pred[y == 0].mean()),
            "recall": float(pred[y == 1].mean()),
            "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)),
        }

        output_rows.append(result)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print("===== 새 데이터 모델 성능 평가 =====")

    for row in output_rows:
        print(
            f"{row['rpm']} RPM | {row['model']:<12} "
            f"FPR={row['fpr']:.1%} "
            f"Recall={row['recall']:.1%} "
            f"AUROC={row['auroc']:.3f} "
            f"AUPRC={row['auprc']:.3f}"
        )

    print("\n결과 저장:", output)
    print("모델 재학습 및 임계값 변경 없음")
    print("물리적 Run 독립성은 별도 확인 필요")


if __name__ == "__main__":
    main()
