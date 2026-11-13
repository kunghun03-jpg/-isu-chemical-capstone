"""Read-only AE normal-date diagnostics; fixed weights/statistics/thresholds."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.analyze_date_score_distributions import read_csv, write_csv, verify_saved, snapshot
from scripts.analyze_normal_ch3_dates import BANDS
from vibration.comparison_metrics import score_rows, file_scores
from vibration.experiment import load_experiment, check_fingerprint
from vibration.inference import SavedModelInference
from vibration.preprocessing import apply_normalization, frequency_bins

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test_normal")
DATES = {"train": "20260912", "validation": "20260916", "test_normal": "20260919"}
COLORS = {"train": "#2878B5", "validation": "#2D9C54", "test_normal": "#D85F26"}


@contextmanager
def checked_numbers():
    with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise"):
        warnings.simplefilter("error", RuntimeWarning)
        yield


def score_summary(rows, threshold):
    result = []
    for split in SPLITS:
        for condition in ("ALL", "Normal1", "Normal2"):
            selected = [r for r in rows if r["split"] == split and (condition == "ALL" or r["condition"] == condition)]
            if not selected:
                continue
            values = np.array([r["score"] for r in selected])
            result.append({"split": split, "date": DATES[split], "condition": condition,
                           "files": len(selected), "median": float(np.median(values)),
                           "q10": float(np.percentile(values, 10)), "q90": float(np.percentile(values, 90)),
                           "q95_descriptive_only": float(np.percentile(values, 95)),
                           "above_fixed_threshold": int((values > threshold).sum()),
                           "above_threshold_rate": float((values > threshold).mean()),
                           "fixed_file_threshold": threshold})
    return result


def masks(frequency):
    return {name: (frequency >= low) & (frequency <= high if name == "high" else frequency < high)
            for name, low, high in BANDS}


def plots(output, rpm, rows, profiles, threshold, std):
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    rng = np.random.default_rng(42)
    for i, split in enumerate(SPLITS):
        for j, condition in enumerate(("Normal1", "Normal2")):
            values = [r["score"] for r in rows if r["split"] == split and r["condition"] == condition]
            if not values:
                continue
            x = i + (-.16 if j == 0 else .16)
            ax.boxplot([values], positions=[x], widths=.25, showfliers=False)
            ax.scatter(x + rng.uniform(-.07, .07, len(values)), values, s=13,
                       color=("#2878B5", "#E27C24")[j], alpha=.55,
                       label=condition if i == 0 else None)
    ax.axhline(threshold, color="#C52222", ls="--", label=f"Stored validation file Q95 = {threshold:.8g}")
    ax.set(xticks=[0, 1, 2], xticklabels=["Train\nSep 12", "Validation\nSep 16", "Test normal\nSep 19"],
           ylabel="File MSE (mean of eight segment scores)", title=f"{rpm} RPM | Fixed AE / fixed threshold")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(output / f"{rpm}_scores.png", dpi=160); plt.close(fig)
    frequency = frequency_bins()
    fig, axes = plt.subplots(2, 3, figsize=(21, 9), constrained_layout=True)
    for i, condition in enumerate(("Normal1", "Normal2")):
        for j, key in enumerate(("pre_z", "normalized", "error")):
            ax = axes[i, j]
            for split in SPLITS:
                selected = [p[key] for p in profiles if p["split"] == split and p["condition"] == condition]
                if not selected:
                    continue
                values = np.stack(selected)
                median = np.median(values, axis=0)
                lower, upper = np.percentile(values, [10, 90], axis=0)
                ax.plot(frequency, median, color=COLORS[split], lw=.8, label=f"{DATES[split]} n={len(values)}")
                ax.fill_between(frequency, lower, upper, color=COLORS[split], alpha=.12)
            title = ("pre-Z log1p spectrum", "fixed Train Z-score spectrum", "AE squared reconstruction error")[j]
            ax.set(title=f"{condition}: {title}",
                   xlabel="Frequency (Hz)", ylabel="Median and 10-90% across file profiles")
            ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"{rpm} RPM | Each profile = mean of eight segment spectra")
    fig.savefig(output / f"{rpm}_spectra.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].semilogy(frequency, std.reshape(-1))
    axes[0].axhline(1e-6, ls="--", color="gray", label="Existing std floor")
    axes[0].set(title="Stored Train std (unchanged)", xlabel="Frequency (Hz)"); axes[0].legend()
    for condition, style in (("Normal1", "-"), ("Normal2", "--")):
        base = [p["pre_z"] for p in profiles if p["split"] == "train" and p["condition"] == condition]
        if not base:
            continue
        baseline = np.median(np.stack(base), axis=0)
        for split in SPLITS[1:]:
            current = [p["pre_z"] for p in profiles if p["split"] == split and p["condition"] == condition]
            if current:
                # Mathematical scaling diagnostic, not an AE error or cause attribution.
                delta = (np.median(np.stack(current), axis=0) - baseline) / std.reshape(-1)
                axes[1].plot(frequency, delta, color=COLORS[split], ls=style, lw=.8,
                             label=f"{condition} {DATES[split]}")
    axes[1].set(title="Pre-Z median shift / fixed Train std (NOT reconstruction error)", xlabel="Frequency (Hz)")
    axes[1].legend(fontsize=8)
    fig.savefig(output / f"{rpm}_normalization.png", dpi=160); plt.close(fig)


def compute_rpm(source, bundle, output, rpm):
    datasets, fingerprint = load_experiment(source, rpm)
    engine = SavedModelInference(bundle, models=("autoencoder",))
    check_fingerprint(engine.manifest, fingerprint)
    if engine.rpm != rpm:
        raise ValueError("bundle RPM mismatch")
    checkpoint = torch.load(source / f"autoencoder_{rpm}rpm.pt", map_location="cpu", weights_only=True)
    check_fingerprint(checkpoint, fingerprint)
    model = engine._estimators["autoencoder"]
    for key, value in model.state_dict().items():
        if not torch.equal(value, checkpoint["model_state_dict"][key]):
            raise ValueError("original checkpoint and bundle AE weights differ")
    threshold = float(engine.manifest["models"]["autoencoder"]["file_threshold"])
    segment_rows, profiles, bands = [], [], []
    frequency = frequency_bins()
    band_masks = masks(frequency)
    for split in SPLITS:
        data = datasets[split]
        if set(data["condition"].astype(str)) - {"Normal1", "Normal2"} or set(data["date"].astype(str)) != {DATES[split]}:
            raise ValueError(f"{split}: unexpected normal condition/date")
        with np.load(source / f"{split}_spectra.npz", allow_pickle=False) as raw:
            pre_z = raw["X"].copy()
            np.testing.assert_array_equal(raw["frequency"], frequency)
            for key in ("source_file", "segment_id", "condition", "date", "rpm", "run_id"):
                np.testing.assert_array_equal(raw[key], data[key], err_msg=f"pre-Z/normalized {key}")
        if not np.isfinite(pre_z).all():
            raise ValueError("nonfinite pre-Z spectrum")
        np.testing.assert_array_equal(apply_normalization(pre_z, engine.mean, engine.std), data["X"])
        errors = []
        with torch.inference_mode():
            for start in range(0, len(data["X"]), 32):
                x = torch.from_numpy(data["X"][start:start+32]).float()
                squared = (model(x).double() - x.double()).square().numpy()
                if not np.isfinite(squared).all():
                    raise ValueError("nonfinite reconstruction error")
                errors.append(squared)
        errors = np.concatenate(errors)
        scores = errors.mean(axis=(1, 2))
        segment_rows.extend(score_rows(data, split, scores))
        for file in np.unique(data["source_file"]):
            selected = data["source_file"] == file
            condition = str(data["condition"][selected][0])
            pre_profile = pre_z[selected, 0].mean(axis=0, dtype=np.float64)
            z_profile = data["X"][selected, 0].mean(axis=0, dtype=np.float64)
            error_profile = errors[selected, 0].mean(axis=0)
            profiles.append({"file": str(file), "split": split, "condition": condition,
                             "pre_z": pre_profile, "normalized": z_profile, "error": error_profile})
            for name, mask in band_masks.items():
                bands.append({"file": str(file), "rpm": rpm, "split": split, "date": DATES[split],
                              "condition": condition, "band": name, "bins": int(mask.sum()),
                              "pre_z_mean": float(pre_z[selected, 0][:, mask].mean()),
                              "z_rms": float(np.sqrt(np.square(data["X"][selected, 0][:, mask].astype(np.float64)).mean())),
                              "band_mse": float(error_profile[mask].mean()),
                              "score_contribution": float(error_profile[mask].sum() / 3201)})
    files = file_scores(segment_rows)
    for row in files:
        row.update(rpm=rpm, threshold=threshold, predicted_anomaly=int(row["score"] > threshold))
    calculated_q95 = float(np.percentile([r["score"] for r in files if r["split"] == "validation"], 95))
    np.testing.assert_allclose(calculated_q95, threshold, rtol=1e-7, atol=1e-10)
    # Only original normal-test rows are needed: no fault scores enter analysis or threshold checks.
    saved = [r for r in read_csv(source / f"evaluation_file_{rpm}rpm.csv") if r["split"] == "test_normal"]
    if not saved:
        raise ValueError("original normal-test evaluation records absent")
    max_difference = verify_saved(saved, files, "file")
    original = {(r["split"], r["file"]): r for r in saved}
    comparisons = []
    for condition in ("ALL", "Normal1", "Normal2"):
        selected = [r for r in files if r["split"] == "test_normal" and (condition == "ALL" or r["condition"] == condition)]
        if not selected:
            continue
        old = [original[(r["split"], r["file"])] for r in selected]
        if any(float(r["threshold"]) != threshold for r in old):
            raise ValueError("original evaluation threshold differs from bundle stored value")
        new_fp = sum(r["predicted_anomaly"] for r in selected)
        old_fp = sum(int(r["predicted_anomaly"]) for r in old)
        changed = [r["file"] for r, before in zip(selected, old)
                   if r["predicted_anomaly"] != int(before["predicted_anomaly"])]
        comparisons.append({"rpm": rpm, "condition": condition, "normal_files": len(selected),
                            "recomputed_fp": new_fp, "original_fp": old_fp,
                            "recomputed_fpr": new_fp / len(selected), "original_fpr": old_fp / len(selected),
                            "changed_decisions": changed})
    distributions = score_summary(files, threshold)
    shifts = []
    shift_masks = dict(band_masks)
    shift_masks.update({f"{low:05d}-{low+500:05d}Hz":
                        (frequency >= low) & (frequency <= low+500 if low == 9500 else frequency < low+500)
                        for low in range(0, 10000, 500)})
    for condition in ("Normal1", "Normal2"):
        baseline = [p for p in profiles if p["split"] == "train" and p["condition"] == condition]
        if not baseline:
            continue
        reference = np.median(np.stack([p["pre_z"] for p in baseline]), axis=0)
        for split in SPLITS[1:]:
            current = [p for p in profiles if p["split"] == split and p["condition"] == condition]
            if not current:
                continue
            delta = np.median(np.stack([p["pre_z"] for p in current]), axis=0) - reference
            for name, mask in shift_masks.items():
                shifts.append({"rpm": rpm, "split": split, "condition": condition, "band": name,
                               "mean_absolute_pre_z_median_shift": float(np.abs(delta[mask]).mean()),
                               "mean_absolute_shift_divided_by_train_std": float(np.abs(delta[mask] / engine.std.reshape(-1)[mask]).mean())})
    for file in files:
        contributions = [b["score_contribution"] for b in bands if b["file"] == file["file"]]
        np.testing.assert_allclose(sum(contributions), file["score"], rtol=1e-12, atol=1e-12)
    write_csv(output / f"{rpm}_file_scores.csv", files)
    write_csv(output / f"{rpm}_distributions.csv", distributions)
    write_csv(output / f"{rpm}_band_errors.csv", bands)
    write_csv(output / f"{rpm}_band_shifts.csv", shifts)
    np.savez_compressed(output / f"{rpm}_profiles.npz", frequency=frequency,
                        source_file=np.array([p["file"] for p in profiles]),
                        split=np.array([p["split"] for p in profiles]),
                        condition=np.array([p["condition"] for p in profiles]),
                        pre_z=np.stack([p["pre_z"] for p in profiles]),
                        normalized=np.stack([p["normalized"] for p in profiles]),
                        squared_error=np.stack([p["error"] for p in profiles]), train_std=engine.std)
    plots(output, rpm, files, profiles, threshold, engine.std)
    return {"rpm": rpm, "stored_file_threshold": threshold, "validation_q95_check": calculated_q95,
            "q95_difference": calculated_q95 - threshold, "max_saved_score_difference": max_difference,
            "fpr_comparison": comparisons, "distributions": distributions,
            "std_floor_bins": int((engine.std == 1e-6).sum()),
            "run_independence_verified": False,
            "reproduced_decisions": all(not r["changed_decisions"] for r in comparisons)}


def analyze(outputs_root, bundle_root, output, run_name="date_prelim_20261106"):
    if not re_run_name(run_name):
        raise ValueError("invalid run name")
    outputs_root, bundle_root, output = [Path(p).expanduser().resolve() for p in (outputs_root, bundle_root, output)]
    if (output.is_relative_to(ROOT) or any((p / ".git/HEAD").is_file() or (p / ".git").is_file()
                                        for p in (output, *output.parents))
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (outputs_root, bundle_root))):
        raise ValueError("output must be a new directory outside Git and input directories")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "observations": [], "hypotheses_not_confirmed": [
        "Input spectrum differences across dates/normal conditions may reflect domain shift; sensor/mount/operation causes are unconfirmed.",
        "Small fixed Train std can amplify pre-Z shifts mathematically; this alone does not establish the cause of AE false alarms.",
        "Normal1/Normal2 composition may influence pooled distributions; compare each condition separately.",
    ], "limitations": ["Date-group preliminary experiment; actual Run independence unconfirmed.",
                         "Test data are diagnostic only. No threshold selection/optimization is performed.",
                         "All four normalized partitions are read for existing fingerprint validation; only normal partitions are analyzed."],
              "retraining": False, "threshold_changed": False, "source_before_sha256": {}}
    try:
        for rpm in (900, 1200, 1800):
            source, bundle = outputs_root / f"{rpm}rpm" / run_name, bundle_root / f"{rpm}rpm"
            paths = [p for p in source.iterdir() if p.is_file() and p.suffix in {".npz", ".pt", ".json", ".csv"}]
            paths += [p for p in bundle.iterdir() if p.is_file()]
            report["source_before_sha256"].update(snapshot(paths))
            with checked_numbers():
                result = compute_rpm(source, bundle, output, rpm)
            report["observations"].append(result)
        report["status"] = "completed_diagnostics" if all(r["reproduced_decisions"] for r in report["observations"]) else "decision_mismatch_requires_review"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        before = report["source_before_sha256"]
        after = snapshot([Path(p) for p in before if Path(p).is_file()])
        report["source_unchanged"] = before == after
        if before != after:
            report.update(status="failed", error="input artifacts changed")
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def re_run_name(value):
    return bool(value) and all(c.isascii() and (c.isalnum() or c in "_-") for c in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--run-name", default="date_prelim_20261106")
    parser.add_argument("--output", type=Path, default=Path.home() / "ae-normal-shift" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    bundle_root = args.bundle_root or args.outputs_root / "inference_bundles" / args.run_name
    try:
        report = analyze(args.outputs_root, bundle_root, args.output, args.run_name)
    except Exception as error:
        print(f"FAIL: {error}; report (if created): {args.output}", file=sys.stderr)
        return 1
    for rpm in report["observations"]:
        total = next(r for r in rpm["fpr_comparison"] if r["condition"] == "ALL")
        print(f"{rpm['rpm']}RPM: FPR {total['recomputed_fpr']:.2%}, original {total['original_fpr']:.2%}, fixed Q95 {rpm['stored_file_threshold']:.12g}")
    print(f"{report['status']}: {args.output}; causal hypotheses remain unconfirmed; no tuning")
    return 0 if report["status"] == "completed_diagnostics" and report["source_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
