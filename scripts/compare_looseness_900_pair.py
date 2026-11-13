"""저장 모델로 두 Looseness0.1 파일을 비교합니다. 학습·임계값 변경 없음."""

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


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/looseness_900_pair_20261110"
    if output.exists():
        raise ValueError("기존 비교 결과를 덮어쓰지 않습니다.")
    paths = [root / "data/20260912/20260912_900RPM/Looseness0.1/Vib/20260912_170916_Vib.mat",
             root / "data/20260919/20260919_900RPM/Looeness0.1/Vib/20260919_152236_Vib.mat"]
    bundle = root / "outputs/inference_bundles/date_prelim_20261106/900rpm"
    watched = paths + [p for p in bundle.iterdir() if p.is_file()]
    def snapshot():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = snapshot()
    with (root / "manifests/run_inventory_unconfirmed.csv").open(encoding="utf-8-sig", newline="") as stream:
        inventory = {r["file"]: r for r in csv.DictReader(stream)}
    engine = SavedModelInference(bundle)
    raws, spectra, predictions, stats = [], [], [], []
    for path in paths:
        metadata = inventory[path.relative_to(root).as_posix()]
        assert metadata["rpm"] == "900" and metadata["condition"] == "Looseness0.1"
        raw = path.read_bytes()
        signal = loadmat(path)["signals"][2].copy()
        spectrum, _ = engine.preprocess(raw)
        result = engine.predict_bytes(raw, rpm=900, filename=path.name)
        raws.append(signal); spectra.append(spectrum[:, 0]); predictions.append(result)
        stats.append({"file": path.relative_to(root).as_posix(), "sha256": result["input"]["sha256"],
                      "date": metadata["date"], "rpm": metadata["rpm"], "condition": metadata["condition"],
                      "mean": float(signal.mean()), "rms": float(np.sqrt(np.mean(signal ** 2))),
                      "ac_rms": float(np.std(signal)), "peak_absolute": float(np.max(np.abs(signal)))})
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
    colors = ["#2369b3", "#d46616"]
    labels = ["9/12 · 20260912_170916", "9/19 · 20260919_152236"]
    frequency = frequency_bins()
    time = np.arange(65536) / 25600
    peak = max(float(np.abs(signal).max()) for signal in raws)
    ymax = max(float(values.max()) for values in spectra)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex="row", sharey="row", constrained_layout=True)
    for index, (signal, values) in enumerate(zip(raws, spectra)):
        axes[0, index].plot(time, signal, color=colors[index], linewidth=.45)
        axes[0, index].set(title=labels[index] + " · CH3 파형", xlim=(0, 2.56), ylim=(-peak * 1.05, peak * 1.05), xlabel="시간 (초)", ylabel="원본 진폭 (물리 단위 미확정)")
        for segment in values:
            axes[1, index].plot(frequency, segment, color=colors[index], alpha=.16, linewidth=.6)
        axes[1, index].plot(frequency, values.mean(axis=0, dtype=np.float64), color=colors[index], linewidth=1.2, label="8구간 log1p 평균")
        axes[1, index].set(title=labels[index] + " · 정규화 전 스펙트럼", xlim=(0, 10000), ylim=(0, ymax * 1.05), xlabel="주파수 (Hz)", ylabel="log1p(FFT 단측 진폭)")
        axes[1, index].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("900RPM · Looseness0.1 비교 — 행별 동일 축 / 연한 선: 각 구간 스펙트럼")
    fig.savefig(output / "waveform_spectrum.png", dpi=160)
    plt.close(fig)
    rows, segment_rows = [], []
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for ax, name, label in zip(axes, engine.models, ("AE", "OCSVM", "IF")):
        for index, result in enumerate(predictions):
            item = result["models"][name]
            scores = np.array([s["score"] for s in item["segments"]])
            assert float(scores.mean()) == item["file_score"]
            ax.plot(np.arange(1, 9), scores, marker="o", color=colors[index], label=labels[index])
            rows.append({"date": stats[index]["date"], "file": paths[index].name, "model": label,
                         "file_score": item["file_score"], "file_threshold": item["file_threshold"],
                         "decision": "이상" if item["predicted_anomaly"] else "정상",
                         "segment_threshold": item["segment_threshold"], "segments_above_threshold": int((scores > item["segment_threshold"]).sum())})
            segment_rows.extend({"date": stats[index]["date"], "model": label, "segment": s["segment_id"] + 1,
                                 "score": s["score"], "threshold": item["segment_threshold"], "predicted_anomaly": s["predicted_anomaly"]} for s in item["segments"])
        threshold = predictions[0]["models"][name]["segment_threshold"]
        assert threshold == predictions[1]["models"][name]["segment_threshold"]
        ax.axhline(threshold, color="red", linestyle="--", label=f"저장 구간 임계값 {threshold:.9g}")
        if name == "iforest":
            value = predictions[0]["models"][name]["segments"][2]["score"]
            ax.annotate(f"9/12 구간 3: {value:.9f}", xy=(3, value), xytext=(4.1, value), arrowprops={"arrowstyle": "->"}, color=colors[0])
        ax.set(title=label, xlabel="구간 번호 (각 0.32초)", ylabel="이상점수 (클수록 이상)", xticks=np.arange(1, 9), xlim=(.7, 8.3))
        ax.grid(alpha=.25); ax.legend(loc="best")
    fig.suptitle("같은 저장 모델·같은 구간 임계값 — 파일 판정은 별도의 8구간 평균 기준")
    fig.savefig(output / "segment_scores.png", dpi=160)
    plt.close(fig)
    for filename, values in (("file_scores.csv", rows), ("segment_scores.csv", segment_rows), ("signal_statistics.csv", stats)):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    assert before == snapshot(), "원본 또는 모델 파일 변경 발견"
    report = {"source_experiment": "date_prelim_20261106", "bundle_id": engine.manifest["bundle_id"],
              "files": stats, "summary": rows, "predictions": predictions, "originals_and_models_unchanged": True,
              "before_sha256": before, "after_sha256": snapshot(), "retraining": False, "threshold_changed": False}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": rows, "signal_statistics": stats, "if_sep12_segment3": predictions[0]["models"]["iforest"]["segments"][2]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
