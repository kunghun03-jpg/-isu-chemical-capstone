"""Read-only single-MAT comparison of offline AE equations and the real V2 adapter."""

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import loadmat
import torch

from vibration import preprocessing
from vibration.dashboard_v2 import DashboardConfig, analyze, configuration
from vibration.inference import SavedModelInference
from vibration.inference_bundle import sha256_file
from vibration.model import Autoencoder

ROOT = Path(__file__).resolve().parents[1]
TARGET = "20260912_144039_Vib.mat"


def locate_mat(explicit=None, roots=(), name=TARGET):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MAT not accessible: {path}")
        return path
    found = sorted({p.resolve() for root in roots if Path(root).expanduser().is_dir()
                    for p in Path(root).expanduser().rglob(name) if p.is_file()})
    if not found:
        raise FileNotFoundError(f"{name} not found; specify --file or --search-root")
    if len(found) != 1:
        raise ValueError(f"Multiple MAT matches; specify --file: {[str(p) for p in found]}")
    return found[0]


class CaptureWeb:
    """Observe actual helper locals and AE output without replacing calculations.

    A short-lived Python trace and a read-only instance forward hook are used
    only in this diagnostic process, and always removed/restored on exit.
    """
    def __init__(self):
        self.arrays, self.hooks = {}, []

    def trace(self, frame, event, arg):
        code, local = frame.f_code, frame.f_locals
        watched = (preprocessing.ch3_log_spectra.__code__, preprocessing.apply_normalization.__code__,
                   SavedModelInference.__init__.__code__)
        if code not in watched:
            return None
        if code is preprocessing.ch3_log_spectra.__code__:
            if "segments" in local and "segments" not in self.arrays:
                self.arrays["segments"] = local["segments"].copy()
            if event == "return" and arg is not None:
                for name, key in (("ch3", "selected"), ("mean_removed", "segments"),
                                  ("hann_window", "window"), ("fft_amplitude", "amplitude"),
                                  ("fft_frequency", "frequency")):
                    self.arrays[name] = local[key].copy()
                self.arrays["log1p_spectrum"] = arg.copy()
                frequency = local["frequency"]
                self.arrays["selected_frequency"] = frequency[frequency <= 10000].copy()
        elif code is preprocessing.apply_normalization.__code__ and event == "return" and arg is not None:
            self.arrays["normalized"] = arg.copy()
            self.arrays["train_mean"] = local["mean"].copy()
            self.arrays["train_std"] = local["std"].copy()
        elif code is SavedModelInference.__init__.__code__ and event == "return":
            model = getattr(local.get("self"), "_estimators", {}).get("autoencoder")
            if model is not None:
                self.web_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                self.hooks.append(model.register_forward_hook(self.forward))
        return self.trace

    def forward(self, model, inputs, output):
        self.arrays["model_input"] = inputs[0].detach().cpu().numpy().copy()
        self.arrays["reconstruction"] = output.detach().cpu().numpy().copy()

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *_):
        sys.settrace(self.previous)
        for hook in self.hooks:
            hook.remove()


def offline(raw, source, rpm):
    """Reproduce spectra/normalize/evaluate/file_evaluate math without their writers."""
    data = loadmat(BytesIO(raw))
    signals, fs = data["signals"], np.asarray(data["fs"])
    if (signals.shape != (6, 65536) or signals.dtype.kind not in "fiu"
            or not np.isfinite(signals).all() or (np.abs(signals) > 1e100).any()
            or fs.size != 1 or fs.dtype.kind not in "fiu" or float(fs.item()) != 25600):
        raise ValueError("MAT does not meet the existing six-channel quality/shape/fs requirements")
    arrays = {"ch3": signals[2:3].copy()}
    arrays["segments"] = arrays["ch3"].reshape(1, 8, 8192).transpose(1, 0, 2)
    arrays["mean_removed"] = arrays["segments"] - arrays["segments"].mean(axis=2, keepdims=True)
    arrays["hann_window"] = np.hanning(8192)
    amplitude = np.abs(np.fft.rfft(arrays["mean_removed"] * arrays["hann_window"], axis=2)) / arrays["hann_window"].sum()
    amplitude[:, :, 1:-1] *= 2
    arrays["fft_amplitude"] = amplitude
    frequency = np.fft.rfftfreq(8192, d=1 / 25600.)
    arrays["fft_frequency"] = frequency
    arrays["selected_frequency"] = frequency[frequency <= 10000]
    arrays["log1p_spectrum"] = np.log1p(amplitude[:, :, frequency <= 10000]).astype(np.float32)
    with np.load(source / f"normalization_{rpm}rpm.npz", allow_pickle=False) as stats:
        mean, std = stats["mean"].copy(), stats["std"].copy()
    if (mean.shape != (1, 1, 3201) or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std < 1e-6).any()):
        raise ValueError("Invalid original Train normalization statistics")
    arrays["train_mean"], arrays["train_std"] = mean, std
    arrays["normalized"] = ((arrays["log1p_spectrum"].astype(np.float64) - mean) / std).astype(np.float32)
    checkpoint = torch.load(source / f"autoencoder_{rpm}rpm.pt", map_location="cpu", weights_only=True)
    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(arrays["normalized"]).float()
        reconstructed = model(batch)
        scores = (reconstructed.double() - batch.double()).square().mean(dim=(1, 2)).numpy()
    arrays["model_input"] = batch.numpy()
    arrays["reconstruction"] = reconstructed.numpy()
    arrays["segment_scores"] = scores
    arrays["file_score"] = np.asarray(float(np.mean(scores)))
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite offline intermediate/score; no prediction produced")
    with (source / f"evaluation_file_{rpm}rpm.csv").open(encoding="utf-8-sig", newline="") as stream:
        archived = list(csv.DictReader(stream))
    thresholds = {float(row["threshold"]) for row in archived}
    if len(thresholds) != 1 or not np.isfinite(list(thresholds)).all():
        raise ValueError("Original file-evaluation threshold missing/inconsistent")
    threshold = thresholds.pop()
    return arrays, checkpoint, model, threshold, archived


def compare_array(stage, before, after, exact_required, rtol, atol):
    same_shape = before.shape == after.shape
    finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
    exact = same_shape and finite and np.array_equal(before, after)
    close = same_shape and finite and np.allclose(before, after, rtol=rtol, atol=atol)
    return {"stage": stage, "offline_shape": list(before.shape), "web_shape": list(after.shape),
            "exact_equal": bool(exact), "within_tolerance": bool(close), "exact_required": exact_required,
            "passed": bool(exact if exact_required else close),
            "max_absolute_difference": float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64))))
            if same_shape and finite and before.size else None}


def verify(path, rpm, experiment_dir, config, output=None, rtol=1e-6, atol=1e-9):
    path, source = Path(path).expanduser().resolve(), Path(experiment_dir).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MAT not accessible: {path}")
    if config.model != "autoencoder":
        raise ValueError("V2 model is not AE; this diagnostic does not silently change configuration")
    encoded = {int(m.group(1)) for p in path.parts if (m := re.fullmatch(r"\d{8}_(\d+)RPM", p, re.I))}
    if encoded and encoded != {rpm}:
        raise ValueError("MAT path RPM and requested RPM differ")
    bundle = config.bundle_root / f"{rpm}rpm"
    manifest = json.loads((bundle / "manifest.json").read_text())
    inputs = [path, source / f"autoencoder_{rpm}rpm.pt", source / f"normalization_{rpm}rpm.npz",
              source / f"evaluation_file_{rpm}rpm.csv", bundle / "manifest.json"]
    for name in manifest["artifacts"]:
        if Path(name).name != name:
            raise ValueError("Invalid bundle artifact path")
        inputs.append(bundle / name)
    before = {str(p): sha256_file(p) for p in inputs}
    if output is not None:
        output = Path(output).expanduser().resolve()
        if (output.is_relative_to(ROOT) or any((p / ".git/HEAD").is_file() or (p / ".git").is_file()
                                             for p in (output, *output.parents))
                or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, config.bundle_root, path.parent))):
            raise ValueError("report directory must be new, outside Git and separate from input folders")
        output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mat": str(path), "rpm": rpm, "mat_sha256": before[str(path)],
              "actual_mat_available": True, "performance_validated": False, "stage_comparisons": [],
              "comparison": "offline single-file calculation reproduction vs actual V2 analyze adapter",
              "source_before_sha256": before, "rtol": rtol, "atol": atol}
    try:
        torch.set_num_threads(min(4, torch.get_num_threads()))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            old, checkpoint, model, threshold, archived = offline(path.read_bytes(), source, rpm)
        if checkpoint.get("data_fingerprint") != manifest["data_fingerprint"]:
            raise ValueError("Original checkpoint and bundle data fingerprints differ")
        with CaptureWeb() as captured:
            web, versions = analyze(path.read_bytes(), rpm, path.name, config)
        item = web["models"]["autoencoder"]
        captured.arrays["segment_scores"] = np.asarray([s["score"] for s in item["segments"]])
        captured.arrays["file_score"] = np.asarray(item["file_score"])
        for stage, value in old.items():
            report["stage_comparisons"].append(compare_array(
                stage, value, captured.arrays[stage], stage not in {"reconstruction", "segment_scores", "file_score"}, rtol, atol))
        weights_equal = set(model.state_dict()) == set(captured.web_state) and all(
            torch.equal(value, captured.web_state[key]) for key, value in model.state_dict().items())
        threshold_equal = threshold == item["file_threshold"]
        offline_alarm = int(float(old["file_score"]) > threshold)
        report.update(weights_equal=weights_equal, thresholds_equal=threshold_equal,
                      offline={"score": float(old["file_score"]), "threshold": threshold,
                               "decision": "이상" if offline_alarm else "정상"},
                      web={"score": item["file_score"], "threshold": item["file_threshold"],
                           "decision": "이상" if item["predicted_anomaly"] else "정상",
                           "bundle_id": web["bundle_id"], "model_versions": versions},
                      decision_equal=offline_alarm == item["predicted_anomaly"])
        if web["input"]["sha256"] != before[str(path)]:
            raise ValueError("Web analyzed bytes differ from the original file hash")
        # The reported Sep 12 IR file was not part of the Sep 19 test split.
        archive_matches = [r for r in archived if Path(r["file"].replace("\\", "/")).name == path.name]
        report["archived_evaluation_rows_with_same_basename"] = archive_matches
        report["archived_score_note"] = "basename candidates only, not proof of file identity" if archive_matches else "no archived score row; offline score was recomputed"
        failures = [r["stage"] for r in report["stage_comparisons"] if not r["passed"]]
        if not weights_equal: failures.insert(0, "model_weights")
        if not threshold_equal: failures.append("stored_file_threshold")
        if not report["decision_equal"]: failures.append("file_decision")
        report["first_mismatch_stage"] = failures[0] if failures else None
        report["mismatch_stages"] = failures
        report["status"] = "passed_calculation_parity" if not failures else "mismatch_requires_review"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["inputs_unchanged"] = all(Path(p).is_file() and sha256_file(p) == checksum for p, checksum in before.items())
        if not report["inputs_unchanged"]:
            report.update(status="failed", error="input MAT/model/statistics/threshold artifacts changed")
        if output:
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--search-root", nargs="+", type=Path, default=[Path.home() / "capstone", ROOT / "data"])
    parser.add_argument("--name", default=TARGET)
    parser.add_argument("--rpm", type=int, choices=(900, 1200, 1800), default=900)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path, help="optional fresh report directory outside Git; default stdout only")
    args = parser.parse_args()
    try:
        path = locate_mat(args.file, args.search_root, args.name)
        config = DashboardConfig("autoencoder", args.bundle_root.expanduser().resolve()) if args.bundle_root else configuration()
        source = args.experiment_dir or ROOT / "outputs" / f"{args.rpm}rpm" / "date_prelim_20261106"
        report = verify(path, args.rpm, source, config, args.output)
    except Exception as error:
        print(f"NOT VERIFIED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Offline | score={report['offline']['score']:.12g} | threshold={report['offline']['threshold']:.12g} | {report['offline']['decision']}")
    print(f"V2 web  | score={report['web']['score']:.12g} | threshold={report['web']['threshold']:.12g} | {report['web']['decision']}")
    return 0 if report["status"] == "passed_calculation_parity" else 2


if __name__ == "__main__":
    raise SystemExit(main())
