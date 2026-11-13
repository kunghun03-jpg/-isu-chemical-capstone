"""정상 두 대표 파일의 저장 AE 주파수별 복원 제곱오차 분해. 재학습 없음."""

from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vibration.inference import SavedModelInference
from vibration.preprocessing import frequency_bins


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/normal1_normal2_900_ae_medians_20261110"
    output = root / "outputs/normal1_normal2_900_ae_frequency_error_20261110"
    if output.exists():
        raise ValueError("기존 분석 결과를 덮어쓰지 않습니다.")
    bundle = root / "outputs/inference_bundles/date_prelim_20261106/900rpm"
    paths = [root / f"data/20260919/20260919_900RPM/{condition}/Vib/{file}" for condition, file in
             (("Normal1", "20260919_131708_Vib.mat"), ("Normal2", "20260919_145354_Vib.mat"))]
    watched = paths + [source / "summary.csv"] + [p for p in bundle.iterdir() if p.is_file()]
    def snapshot():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = snapshot()
    with (source / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        old = list(csv.DictReader(stream))
    engine = SavedModelInference(bundle, models=("autoencoder",))
    errors, verification = [], []
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        _, normalized = engine.preprocess(raw)
        result = engine.predict_bytes(raw, rpm=900, filename=path.name)["models"]["autoencoder"]
        with torch.inference_mode():
            values = torch.from_numpy(normalized).float()
            squared = (engine._estimators["autoencoder"](values).double() - values.double()).square().numpy()
        per_segment = squared.mean(axis=(1, 2))
        # NumPy와 PyTorch의 합산 순서에 따른 double 반올림 차이만 허용합니다.
        np.testing.assert_allclose(per_segment, [s["score"] for s in result["segments"]], rtol=0, atol=1e-14)
        error_frequency = squared.mean(axis=(0, 1))
        file_score = result["file_score"]
        assert file_score == float(old[index]["file_score"])
        np.testing.assert_allclose(per_segment.mean(), file_score, rtol=0, atol=1e-14)
        np.testing.assert_allclose(error_frequency.mean(), file_score, rtol=0, atol=1e-14)
        errors.append(error_frequency)
        verification.append({"file": path.name, "existing_file_score": file_score,
                             "frequency_mean": float(error_frequency.mean()), "absolute_difference": abs(float(error_frequency.mean()) - file_score)})
    frequency = frequency_bins()
    n = len(frequency)
    delta = errors[1] - errors[0]
    gap = verification[1]["existing_file_score"] - verification[0]["existing_file_score"]
    def band(low, high, last=False):
        mask = (frequency >= low) & ((frequency <= high) if last else (frequency < high))
        c1, c2 = float(errors[0][mask].sum() / n), float(errors[1][mask].sum() / n)
        difference = c2 - c1
        return {"low_hz": low, "high_hz": high, "upper_inclusive": last, "bins": int(mask.sum()),
                "normal1_score_contribution": c1, "normal2_score_contribution": c2,
                "score_gap_contribution": difference, "percent_of_total_gap": difference / gap * 100}
    bands = [band(low, low + 500, low == 9500) for low in range(0, 10000, 500)]
    focus = band(2200, 2600)
    np.testing.assert_allclose(sum(b["score_gap_contribution"] for b in bands), gap, rtol=0, atol=1e-14)
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for index, (error, label, color) in enumerate(zip(errors, ("Normal1 · 정상 대표", "Normal2 · 오탐 대표"), ("#2369b3", "#d46616"))):
        axes[0].plot(frequency, error, label=label, color=color, linewidth=.8)
    axes[0].set(title="주파수별 복원 제곱오차 — 8구간 평균, 정규화된 입력 기준", ylabel="평균 (정규화 입력 − AE 복원값)²", xlim=(0, 10000))
    axes[0].legend()
    axes[1].plot(frequency, delta, color="#7445a0", linewidth=.8, label="Normal2 − Normal1")
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set(title="주파수별 제곱오차 차이 — 양수: Normal2 점수 증가, 음수: 감소", xlabel="주파수 (Hz)", ylabel="8구간 평균 제곱오차 차이", xlim=(0, 10000))
    for ax in axes:
        ax.axvspan(2200, 2600, color="gold", alpha=.22, label="확인 대역 2.2–2.6kHz")
        ax.grid(alpha=.2)
    axes[1].legend()
    fig.savefig(output / "frequency_squared_error.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    y = np.arange(len(bands))
    ax.barh(y, [b["score_gap_contribution"] for b in bands], color=["#d46616" if b["score_gap_contribution"] >= 0 else "#2369b3" for b in bands])
    ax.set(yticks=y, yticklabels=[f"{b['low_hz']}–{b['high_hz']} Hz" for b in bands], xlabel="파일 점수 차이 기여량 = 대역 내 (Normal2 − Normal1) 오차 합 / 3201",
           title=f"겹치지 않는 500Hz 대역별 기여량 · 전체 점수 차이 {gap:.9f}")
    ax.invert_yaxis(); ax.axvline(0, color="black", linewidth=.7); ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "band_contributions.png", dpi=160); plt.close(fig)
    def write_csv(filename, rows):
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv("frequency_errors.csv", [{"frequency_hz": float(f), "normal1_error": float(e1), "normal2_error": float(e2), "difference": float(e2-e1)} for f, e1, e2 in zip(frequency, *errors)])
    write_csv("bands_500hz.csv", bands)
    write_csv("focus_2200_2600hz.csv", [focus])
    assert before == snapshot()
    report = {"verification": verification, "score_gap": gap, "focus_band": focus, "bands": bands,
              "source_experiment": "date_prelim_20261106", "normalization": "saved per-frequency Train mean/std",
              "before_sha256": before, "after_sha256": snapshot(), "originals_and_models_unchanged": True,
              "retraining": False, "threshold_changed": False}
    (output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verification": verification, "score_gap": gap, "focus": focus,
                      "largest_bands_by_absolute_contribution": sorted(bands, key=lambda b: abs(b["score_gap_contribution"]), reverse=True)[:8]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
