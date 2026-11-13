"""구간·파일 평가와 세 모델의 동일 시험 집합 비교."""

from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from vibration.experiment import check_fingerprint, load_experiment, experiment_metadata


def score_rows(data, split, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(data["X"]),) or not np.isfinite(scores).all():
        raise ValueError("이상점수의 길이 또는 유한값을 확인하세요.")
    return [
        {
            "file": str(data["source_file"][i]),
            "condition": str(data["condition"][i]),
            "split": split,
            "date": str(data["date"][i]),
            "rpm": str(data["rpm"][i]),
            "run_id": str(data["run_id"][i]),
            "segment_id": int(data["segment_id"][i]),
            "score": float(score),
        }
        for i, score in enumerate(scores)
    ]


def file_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["file"])].append(row)
    result = []
    for _, entries in sorted(grouped.items()):
        if len(entries) != 8 or {r["segment_id"] for r in entries} != set(range(8)):
            raise ValueError("파일 점수에는 중복 없는 8개 구간이 필요합니다.")
        row = {
            key: value
            for key, value in entries[0].items()
            if key not in {"segment_id", "score"}
        }
        row["score"] = float(np.mean([r["score"] for r in entries]))
        result.append(row)
    return result


def q95(rows):
    values = [r["score"] for r in rows if r["split"] == "validation"]
    if not values or not np.isfinite(values).all():
        raise ValueError("정상 검증 이상점수가 필요합니다.")
    return float(np.percentile(values, 95))


def predictions(rows, threshold):
    return [
        {
            **row,
            "threshold": threshold,
            "predicted_anomaly": int(row["score"] > threshold),
        }
        for row in rows
    ]


def metrics(rows):
    selected = [r for r in rows if r["split"] in {"test_normal", "test_anomaly"}]
    y = np.array([int(r["split"] == "test_anomaly") for r in selected])
    scores = np.array([r["score"] for r in selected], dtype=np.float64)
    alarms = np.array([r["predicted_anomaly"] for r in selected], dtype=int)
    if set(y.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("정상·결함 시험 데이터와 유한한 이상점수가 모두 필요합니다.")
    fpr, recall = float(alarms[y == 0].mean()), float(alarms[y == 1].mean())
    conditions = sorted({r["condition"] for r in selected})
    return {
        "normal_count": int((y == 0).sum()),
        "anomaly_count": int((y == 1).sum()),
        "false_positive_rate": fpr,
        "specificity": 1 - fpr,
        "detection_rate": recall,
        "auroc": float(roc_auc_score(y, scores)),
        "auprc_average_precision": float(average_precision_score(y, scores)),
        "condition_alarm_rates": {
            c: float(
                np.mean(
                    [r["predicted_anomaly"] for r in selected if r["condition"] == c]
                )
            )
            for c in conditions
        },
    }


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_test_rows(rows, datasets, level):
    expected = {}
    for split in ("test_normal", "test_anomaly"):
        for r in score_rows(
            datasets[split], split, np.zeros(len(datasets[split]["X"]))
        ):
            key = (
                (split, r["file"], r["segment_id"])
                if level == "segment"
                else (split, r["file"])
            )
            expected[key] = r["condition"]
    seen = set()
    for row in rows:
        key = (
            (row["split"], row["file"], int(row["segment_id"]))
            if level == "segment"
            else (row["split"], row["file"])
        )
        if key in seen or key not in expected or row["condition"] != expected[key]:
            raise ValueError("모델 간 시험 파일·구간·상태가 다르거나 중복되었습니다.")
        seen.add(key)
        score, threshold = float(row["score"]), float(row["threshold"])
        if not np.isfinite([score, threshold]).all() or int(
            row["predicted_anomaly"]
        ) != int(score > threshold):
            raise ValueError("점수·임계값·판정이 일치하지 않습니다.")
        row.update(
            score=score,
            threshold=threshold,
            predicted_anomaly=int(row["predicted_anomaly"]),
        )
    if seen != set(expected):
        raise ValueError("모델 평가에 누락된 시험 파일 또는 구간이 있습니다.")


def run(project_dir, output_dir, rpm):
    """기존 평가 CSV만 비교합니다. 여기서는 어떤 모델도 학습하지 않습니다."""
    output_dir = Path(output_dir)
    datasets, fingerprint = load_experiment(output_dir, rpm)
    summary = {
        "data_fingerprint": fingerprint,
        **experiment_metadata(output_dir, rpm),
        "auprc_definition": "average_precision",
        "models": {},
    }
    for model in ("autoencoder", "ocsvm", "iforest"):
        summary["models"][model] = {}
        for level in ("segment", "file"):
            if model == "autoencoder":
                context = json.loads(
                    (
                        output_dir / f"evaluation_{level}_context_{rpm}rpm.json"
                    ).read_text(encoding="utf-8")
                )
                path = output_dir / (
                    f"evaluation_{rpm}rpm.csv"
                    if level == "segment"
                    else f"evaluation_file_{rpm}rpm.csv"
                )
            else:
                base = output_dir / "comparisons" / model
                context = json.loads((base / "config.json").read_text(encoding="utf-8"))
                path = base / f"scores_{level}.csv"
            check_fingerprint(context, fingerprint)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [
                    r
                    for r in csv.DictReader(stream)
                    if r["split"] in {"test_normal", "test_anomaly"}
                ]
            validate_test_rows(rows, datasets, level)
            summary["models"][model][level] = metrics(rows)
    # 모든 모델의 데이터 일치 검사가 끝난 경우에만 비교 결과를 저장합니다.
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("세 모델의 동일 시험 파일·구간 비교 결과 저장 완료")
