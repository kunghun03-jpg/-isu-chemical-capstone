"""전체 기존 시험 MAT에서 단일 파일 추론과 배치 결과를 비교합니다. 학습 없음."""

from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np

from scripts.analyze_date_score_distributions import snapshot
from vibration.dataset import read_rows
from vibration.inference import SavedModelInference


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="date_prelim_20261106")
    args = parser.parse_args()
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_name
    ):
        parser.error("잘못된 실험 이름")
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / f"{args.run_name}_single_inference_validation"
    if output.exists():
        raise ValueError("검증 결과를 덮어쓰지 않습니다.")
    sources = [p for p in (root / "outputs").rglob("*") if p.is_file()]
    before = snapshot(sources)
    quality = {
        r["file"]: r
        for r in csv_rows(root / "outputs/data_preparation/quality_all_files.csv")
    }
    output.mkdir(parents=True)
    records = []
    audits = []
    raw_count = 0
    try:
        for rpm in (900, 1200, 1800):
            source = root / "outputs" / f"{rpm}rpm" / args.run_name
            engine = SavedModelInference(
                root / "outputs/inference_bundles" / args.run_name / f"{rpm}rpm"
            )
            test = [
                r
                for r in read_rows(source / f"split_{rpm}rpm.csv")
                if r["split"] in {"test_normal", "test_anomaly"}
            ]
            archived = {}
            for split in ("test_normal", "test_anomaly"):
                with np.load(source / f"{split}_normalized.npz") as data:
                    # 압축 NPZ 배열은 한 번만 읽어 파일마다 재압축 해제하지 않습니다.
                    source_files = data["source_file"]
                    values = data["X"]
                    segment_ids = data["segment_id"]
                    for file in np.unique(source_files):
                        mask = source_files == file
                        archived[str(file)] = values[mask][
                            np.argsort(segment_ids[mask])
                        ]
            files = {}
            segments = {}
            for name in ("autoencoder", "ocsvm", "iforest"):
                file_path = (
                    source / f"evaluation_file_{rpm}rpm.csv"
                    if name == "autoencoder"
                    else source / "comparisons" / name / "scores_file.csv"
                )
                seg_path = (
                    source / f"evaluation_{rpm}rpm.csv"
                    if name == "autoencoder"
                    else source / "comparisons" / name / "scores_segment.csv"
                )
                files[name] = {
                    r["file"]: r
                    for r in csv_rows(file_path)
                    if r["split"] in {"test_normal", "test_anomaly"}
                }
                segments[name] = {
                    (r["file"], int(r["segment_id"])): r
                    for r in csv_rows(seg_path)
                    if r["split"] in {"test_normal", "test_anomaly"}
                }
                assert set(files[name]) == {r["file"] for r in test}
            for index, row in enumerate(test, 1):
                path = root / row["file"]
                stat = path.stat()
                raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == quality[row["file"]]["sha256"]
                _, normalized = engine.preprocess(raw)
                np.testing.assert_array_equal(normalized, archived[row["file"]])
                result = engine.predict_bytes(raw, rpm=rpm, filename=path.name)
                assert (stat.st_size, stat.st_mtime_ns) == (
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                raw_count += 1
                for name, actual in result["models"].items():
                    expected = files[name][row["file"]]
                    error = abs(actual["file_score"] - float(expected["score"]))
                    # CPU Conv1d의 배치 크기(기존 32, 단일 파일 8)에 따른 반올림 차이를 허용합니다.
                    np.testing.assert_allclose(
                        actual["file_score"],
                        float(expected["score"]),
                        rtol=1e-6,
                        atol=1e-9,
                    )
                    assert actual["file_threshold"] == float(expected["threshold"])
                    assert actual["predicted_anomaly"] == int(
                        expected["predicted_anomaly"]
                    )
                    segment_error = 0.0
                    for segment in actual["segments"]:
                        old = segments[name][(row["file"], segment["segment_id"])]
                        np.testing.assert_allclose(
                            segment["score"], float(old["score"]), rtol=1e-6, atol=1e-9
                        )
                        assert actual["segment_threshold"] == float(old["threshold"])
                        assert segment["predicted_anomaly"] == int(
                            old["predicted_anomaly"]
                        )
                        segment_error = max(
                            segment_error, abs(segment["score"] - float(old["score"]))
                        )
                    records.append(
                        {
                            "rpm": rpm,
                            "file": row["file"],
                            "condition": row["condition"],
                            "model": name,
                            "batch_score": float(expected["score"]),
                            "single_score": actual["file_score"],
                            "absolute_difference": error,
                            "max_segment_difference": segment_error,
                            "threshold": actual["file_threshold"],
                            "batch_prediction": int(expected["predicted_anomaly"]),
                            "single_prediction": actual["predicted_anomaly"],
                            "decision_matches": True,
                        }
                    )
                if index % 50 == 0:
                    print(
                        f"{rpm}RPM single-file verification {index}/{len(test)}",
                        flush=True,
                    )
            for name in engine.models:
                selected = [
                    r for r in records if r["rpm"] == rpm and r["model"] == name
                ]
                audits.append(
                    {
                        "rpm": rpm,
                        "model": name,
                        "files": len(selected),
                        "matching_file_decisions": len(selected),
                        "matching_segment_decisions": len(selected) * 8,
                        "max_file_absolute_difference": max(
                            r["absolute_difference"] for r in selected
                        ),
                        "max_segment_absolute_difference": max(
                            r["max_segment_difference"] for r in selected
                        ),
                    }
                )
            print(
                f"{rpm}RPM: 시험 {len(test)}개 × 세 모델의 전처리·점수·판정 일치",
                flush=True,
            )
        with (output / "per_file_comparison.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = {
            "run_name": args.run_name,
            "raw_test_files": raw_count,
            "file_model_comparisons": len(records),
            "segment_model_comparisons": len(records) * 8,
            "score_tolerance": {"rtol": 1e-6, "atol": 1e-9},
            "decision_mismatches": 0,
            "normalized_arrays_exact_match": True,
            "audits": audits,
            "training_performed": False,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        after = snapshot(sources)
        (output / "preservation.json").write_text(
            json.dumps(
                {
                    "unchanged": before == after,
                    "existing_output_files": len(before),
                    "before_sha256": before,
                    "after_sha256": after,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if before != after:
            raise RuntimeError("기존 실험 또는 추론 패키지 변경 발견")


if __name__ == "__main__":
    main()
