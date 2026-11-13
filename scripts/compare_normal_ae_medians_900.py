"""기존 AE 시험 결과의 조건별 중앙값 대표 파일 비교. 재학습 없음."""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from vibration.inference import SavedModelInference
from vibration.preprocessing import frequency_bins


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/900rpm/date_prelim_20261106"
    output = root / "outputs/normal1_normal2_900_ae_medians_20261110"
    if output.exists():
        raise ValueError("기존 비교 결과를 덮어쓰지 않습니다.")
    evaluation = source / "evaluation_file_900rpm.csv"
    rows = read_csv(evaluation)
    selected, population = [], []
    for condition, predicted in (("Normal1", 0), ("Normal2", 1)):
        group = [row for row in rows if row["split"] == "test_normal" and row["condition"] == condition and int(row["predicted_anomaly"]) == predicted]
        if not group:
            raise ValueError(f"{condition} 선택 조건에 해당하는 파일이 없습니다.")
        median = float(np.median([float(row["score"]) for row in group]))
        closest = min(group, key=lambda row: (abs(float(row["score"]) - median), row["file"]))
        selected.append({**closest, "group_count": len(group), "group_median": median,
                         "distance_from_median": abs(float(closest["score"]) - median)})
        population.extend({**row, "group_median": median, "distance_from_median": abs(float(row["score"]) - median), "selected": row["file"] == closest["file"]} for row in group)
    inventory = {row["file"]: row for row in read_csv(root / "manifests/run_inventory_unconfirmed.csv")}
    quality = {row["file"]: row for row in read_csv(root / "outputs/data_preparation/quality_all_files.csv")}
    bundle = root / "outputs/inference_bundles/date_prelim_20261106/900rpm"
    paths = [root / row["file"] for row in selected]
    watched = paths + [evaluation, source / "evaluation_900rpm.csv"] + [p for p in bundle.iterdir() if p.is_file()]
    def snapshot():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = snapshot()
    engine = SavedModelInference(bundle, models=("autoencoder",))
    old_segments = {(row["file"], int(row["segment_id"])): row for row in read_csv(source / "evaluation_900rpm.csv")}
    signals, spectra, results, summaries, segments = [], [], [], [], []
    for row, path in zip(selected, paths):
        meta = inventory[row["file"]]
        assert meta["date"] == "20260919" and meta["rpm"] == "900" and meta["condition"] == row["condition"] and meta["is_normal"] == "1"
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == quality[row["file"]]["sha256"]
        signal = loadmat(path)["signals"][2].copy()
        spectrum, _ = engine.preprocess(raw)
        result = engine.predict_bytes(raw, rpm=900, filename=path.name)
        item = result["models"]["autoencoder"]
        assert (item["file_score"], item["file_threshold"], item["predicted_anomaly"]) == (float(row["score"]), float(row["threshold"]), int(row["predicted_anomaly"]))
        for segment in item["segments"]:
            old = old_segments[(row["file"], segment["segment_id"])]
            assert (segment["score"], item["segment_threshold"], segment["predicted_anomaly"]) == (float(old["score"]), float(old["threshold"]), int(old["predicted_anomaly"]))
            segments.append({"condition": row["condition"], "file": path.name, "segment": segment["segment_id"] + 1,
                             "score": segment["score"], "segment_threshold": item["segment_threshold"], "above": segment["predicted_anomaly"]})
        signals.append(signal); spectra.append(spectrum[:, 0]); results.append(result)
        summaries.append({"condition": row["condition"], "file": path.name, "group_count": row["group_count"],
                          "group_median": row["group_median"], "distance_from_median": row["distance_from_median"],
                          "file_score": item["file_score"], "file_threshold": item["file_threshold"],
                          "prediction": "이상(오탐)" if item["predicted_anomaly"] else "정상",
                          "segment_threshold": item["segment_threshold"], "segments_above_threshold": sum(s["predicted_anomaly"] for s in item["segments"]),
                          "rms": float(np.sqrt(np.mean(signal ** 2))), "peak_absolute": float(np.abs(signal).max())})
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    colors = ["#2369b3", "#d46616"]
    labels = ["Normal1 · 정상 판정 대표", "Normal2 · 오탐 대표"]
    t = np.arange(65536) / 25600
    f = frequency_bins()
    peak = max(float(np.abs(signal).max()) for signal in signals)
    spectral_peak = max(float(values.max()) for values in spectra)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex="row", sharey="row", constrained_layout=True)
    for i, (signal, values) in enumerate(zip(signals, spectra)):
        axes[0, i].plot(t, signal, color=colors[i], linewidth=.45)
        axes[0, i].set(title=f"{labels[i]}\n{paths[i].name}", xlabel="시간 (초)", ylabel="원본 CH3 진폭 (물리 단위 미확정)", xlim=(0, 2.56), ylim=(-peak * 1.05, peak * 1.05))
        for values_segment in values:
            axes[1, i].plot(f, values_segment, color=colors[i], alpha=.15, linewidth=.6)
        axes[1, i].plot(f, values.mean(axis=0, dtype=np.float64), color=colors[i], label="8개 log1p 스펙트럼 평균", linewidth=1.2)
        axes[1, i].set(title="정규화 전 log1p 스펙트럼", xlabel="주파수 (Hz)", ylabel="log1p(FFT 단측 진폭)", xlim=(0, 10000), ylim=(0, spectral_peak * 1.05))
        axes[1, i].legend()
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle("9/19 · 900RPM · AE 그룹 중앙값 대표 비교 — 파형/스펙트럼 각각 동일 축")
    fig.savefig(output / "waveform_spectrum.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for i, result in enumerate(results):
        item = result["models"]["autoencoder"]
        ax.plot(np.arange(1, 9), [s["score"] for s in item["segments"]], marker="o", color=colors[i], label=labels[i])
    threshold = results[0]["models"]["autoencoder"]["segment_threshold"]
    assert threshold == results[1]["models"]["autoencoder"]["segment_threshold"]
    ax.axhline(threshold, color="red", linestyle="--", label=f"저장 구간 임계값 {threshold:.9f}")
    ax.set(title="AE 8구간 점수 — 파일 판정은 별도의 파일 평균·파일 임계값 기준", xlabel="구간 번호 (각 0.32초)", ylabel="복원 MSE (클수록 이상)", xticks=np.arange(1, 9), xlim=(.7, 8.3))
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(output / "segment_scores.png", dpi=160); plt.close(fig)
    for filename, data in (("summary.csv", summaries), ("segment_scores.csv", segments), ("selection_population.csv", population)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    assert before == snapshot()
    report = {"selection": selected, "summary": summaries, "predictions": results,
              "tie_break": "absolute score distance, then lexicographic full relative path", "archived_batch_match": "exact file and segment scores, thresholds, predictions",
              "before_sha256": before, "after_sha256": snapshot(), "originals_unchanged": True, "retraining": False, "threshold_changed": False}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
