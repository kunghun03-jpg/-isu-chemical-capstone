"""Offline fixed-DAE evaluation. Never declares RPM support or edits a bundle."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import loadmat

from scripts.validate_dae_wsl import array_headers
from vibration.dae_bundle import runtime_versions, sha256_file
from vibration.dae_inference import SavedDAEInference
from vibration.dae_preprocessing import normalize_spectrum, preprocess_mat_bytes

ROOT = Path(__file__).resolve().parents[1]
RPMS = (900, 1200, 1800)
NORMAL = ("Normal1", "Normal2", "Normal3")
FAULT = ("IR", "OR", "Roller", "Looseness0.05", "Looseness0.1", "LoosenessFull",
         "Unbalance_-1", "Unbalance_-2")
ARTIFACTS = ("autoencoder_model.pth", "normal_dataset_normalized.npz", "normal_threshold.npz")


def digest_array(value):
    # The legacy model consumes float32, independent of saved NPZ precision.
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def file_id(value):
    text = str(value).replace("\\", "/")
    match = re.search(r"(?:^|/)(\d{8}_\d+RPM/.*)$", text, re.I)
    return match.group(1) if match else text


def metadata_values(data, headers, keys, count, notes):
    found = {}
    for key in keys:
        if key not in headers:
            continue
        header = headers[key]
        if header["shape"] not in ([count], [count, 1]):
            notes.append(f"{key}: row count/shape mismatch; not used as split evidence")
            continue
        if "object" in header["dtype"]:
            notes.append(f"{key}: object array not loaded (allow_pickle=False)")
            continue
        values = data[key].reshape(-1)
        if values.dtype.kind not in "USiu":
            notes.append(f"{key}: unsupported metadata dtype")
            continue
        found[key] = [str(v) for v in values]
    return found


def inspect_datasets(source):
    """Return report plus conservative train/validation exclusion fingerprints."""
    report = {"normal": {}, "fault": {}, "split_overlaps": [], "notes": [],
              "normalization_train_only": "unconfirmed",
              "run_independence": "unconfirmed", "training_rpm_status": "unconfirmed"}
    split_files, split_features, split_dates, split_runs = {}, {}, {}, {}
    for kind, filename in (("normal", "normal_dataset_normalized.npz"), ("fault", "fault_dataset.npz")):
        path = source / filename
        if not path.is_file():
            report[kind] = {"status": "not provided"}
            continue
        try:
            headers = array_headers(path)
            entry = report[kind] = {"status": "inspected", "headers": headers, "metadata": {}}
            with np.load(path, allow_pickle=False) as data:
                if kind == "normal":
                    for split, aliases in (("train", ("train",)), ("validation", ("val", "validation")),
                                           ("test", ("test",))):
                        xkey = next((f"X_{a}" for a in aliases if f"X_{a}" in headers), None)
                        if xkey is None:
                            report["notes"].append(f"{split}: X array absent")
                            continue
                        shape = headers[xkey]["shape"]
                        if len(shape) != 2 or shape[1] != 800 or "object" in headers[xkey]["dtype"]:
                            report["notes"].append(f"{xkey}: expected numeric (N,800); fingerprints unavailable")
                            continue
                        count = shape[0]
                        keys = [f"{prefix}_{a}" for a in aliases
                                for prefix in ("files", "file_names", "groups", "rpms", "rpm", "dates", "run_ids")]
                        keys += [f"{a}_{suffix}" for a in aliases
                                 for suffix in ("files", "file_names", "groups", "rpms", "rpm", "dates", "run_ids")]
                        meta = metadata_values(data, headers, keys, count, report["notes"])
                        entry["metadata"][split] = meta
                        files = {file_id(v) for k, values in meta.items() if "file" in k for v in values}
                        split_files[split] = files
                        values = data[xkey]
                        if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                            raise ValueError(f"{xkey}: invalid numeric data")
                        split_features[split] = {digest_array(row) for row in values}
                        del values
                        rpms, dates, runs = set(), set(), set()
                        for key, values in meta.items():
                            for value in values:
                                rpms.update(int(v) for v in re.findall(r"(\d+)RPM", value, re.I))
                                dates.update(re.findall(r"(?<!\d)(\d{8})(?!\d)", value))
                                if "rpm" in key and value.isdigit():
                                    rpms.add(int(value))
                                if "run_ids" in key and value.strip():
                                    runs.add(value)
                        split_dates[split], split_runs[split] = dates, runs
                        entry.setdefault("split_evidence", {})[split] = {
                            "rows": count, "metadata_rpm_clues": sorted(rpms),
                            "dates": sorted(dates), "explicit_run_ids": sorted(runs),
                            "unique_file_ids": len(files), "unique_input_fingerprints": len(split_features[split]),
                            "qualification": "saved metadata clues, not verified original file membership",
                        }
                else:
                    count = headers.get("X", {}).get("shape", [0])[0]
                    entry["metadata"] = metadata_values(
                        data, headers, ["labels", "groups", "file_names", "rpms", "rpm", "dates", "run_ids"],
                        count, report["notes"],
                    )
                    entry["rpm_clues"] = sorted({int(v) for k, values in entry["metadata"].items()
                                                 for value in values for v in re.findall(r"(\d+)RPM", value, re.I)})
        except Exception as error:
            report[kind]["status"] = "inspection incomplete"
            report[kind]["error"] = f"{type(error).__name__}: {error}"
            report["notes"].append(f"{filename}: provenance audit incomplete")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        report["split_overlaps"].append({
            "splits": [left, right],
            "file_ids": sorted(split_files.get(left, set()) & split_files.get(right, set())),
            "equal_float32_input_count": len(split_features.get(left, set()) & split_features.get(right, set())),
            "dates": sorted(split_dates.get(left, set()) & split_dates.get(right, set())),
            "explicit_run_ids": sorted(split_runs.get(left, set()) & split_runs.get(right, set())),
        })
    report["limitations"] = [
        "동일 float32 특성은 중복 단서이며 원본 동일성의 증명은 아닙니다.",
        "저장된 정규화 배열만으로 mean/std가 Train에서만 계산됐는지 증명할 수 없습니다.",
        "메타데이터 부재 또는 겹침 미검출은 데이터 누출 부재를 의미하지 않습니다.",
        "날짜_RPM group은 실제 Run ID가 아닙니다. 같은 날짜 혼입은 위험 단서입니다.",
        "정상 NPZ 생성 코드와 원본 행 대응/분할 기록을 별도로 확인해야 합니다.",
    ]
    fault_meta = report["fault"].get("metadata", {})
    fault_files = {file_id(v) for v in fault_meta.get("file_names", [])}
    report["fault_file_overlap_with_normal_splits"] = {
        split: sorted(files & fault_files) for split, files in split_files.items()
    }
    excluded_files = split_files.get("train", set()) | split_files.get("validation", set())
    excluded_features = split_features.get("train", set()) | split_features.get("validation", set())
    return report, excluded_files, excluded_features


def discover(data_root, normal_conditions, fault_conditions, dates=None):
    rows, errors = [], []
    def onerror(error):
        errors.append(str(error))
    for directory, dirs, files in os.walk(data_root, followlinks=False, onerror=onerror):
        for name in list(dirs):
            if (Path(directory) / name).is_symlink():
                errors.append(f"symlink directory not traversed: {Path(directory) / name}")
                dirs.remove(name)
        for name in files:
            if not name.lower().endswith(".mat"):
                continue
            path = Path(directory) / name
            row = {"file": str(path), "date": "", "rpm": "", "condition": "",
                   "source_condition": "", "class": "", "excluded_reason": ""}
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(data_root):
                    raise ValueError("symlink_or_outside_data_root")
                parts = path.relative_to(data_root).parts
                matches = [(i, re.fullmatch(r"(\d{8})_(\d+)RPM", part, re.I))
                           for i, part in enumerate(parts)]
                matches = [(i, m) for i, m in matches if m]
                if len(matches) != 1:
                    raise ValueError("date_rpm_path_unrecognized")
                i, match = matches[0]
                if i + 1 >= len(parts) - 1:
                    raise ValueError("condition_folder_missing")
                date, rpm = match.group(1), int(match.group(2))
                datetime.strptime(date, "%Y%m%d")
                if i and re.fullmatch(r"\d{8}", parts[i - 1]) and parts[i - 1] != date:
                    raise ValueError("parent_date_mismatch")
                condition = parts[i + 1].replace("Looeness", "Looseness")
                row.update(date=date, rpm=rpm, condition=condition, source_condition=parts[i + 1])
                if rpm not in RPMS:
                    raise ValueError("rpm_outside_evaluation_targets")
                if dates and date not in dates:
                    raise ValueError("date_not_selected")
                if condition in normal_conditions:
                    row["class"] = "normal"
                elif condition in fault_conditions:
                    row["class"] = "fault"
                else:
                    raise ValueError("condition_not_selected_or_unknown")
            except Exception as error:
                row["excluded_reason"] = str(error)
            rows.append(row)
    return sorted(rows, key=lambda r: r["file"]), errors


def aggregate(rows):
    result = []
    for rpm in RPMS:
        for kind in ("normal", "fault"):
            group = [r for r in rows if r["rpm"] == rpm and r["class"] == kind]
            for condition in ["ALL"] + sorted({r["condition"] for r in group}):
                selected = [r for r in group if condition == "ALL" or r["condition"] == condition]
                scored = [r for r in selected if not r["excluded_reason"]]
                alarms = sum(r["alarm"] for r in scored)
                result.append({"rpm": rpm, "class": kind, "condition": condition,
                               "discovered": len(selected), "evaluated": len(scored),
                               "excluded": len(selected) - len(scored), "alarms": alarms,
                               "rate": alarms / len(scored) if scored else None,
                               "metric": "false_positive_rate" if kind == "normal" else "detection_rate",
                               "dates": sorted({r["date"] for r in scored})})
    return result


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(source, data_root, output, *, dates=None, normal_conditions=NORMAL, fault_conditions=FAULT):
    source, data_root, output = [Path(p).expanduser().resolve() for p in (source, data_root, output)]
    if (output.is_relative_to(ROOT) or any(
            (p / ".git/HEAD").is_file() or (p / ".git").is_file() for p in (output, *output.parents))
            or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(data_root) or data_root.is_relative_to(output)):
        raise ValueError("출력은 Git 저장소 밖이며 원본/데이터 경로와 중첩되지 않는 새 폴더여야 합니다.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"MAT 데이터 루트 없음: {data_root}")
    if set(normal_conditions) & set(fault_conditions):
        raise ValueError("정상/결함 상태 목록이 겹칩니다.")
    if dates:
        for date in dates:
            if not re.fullmatch(r"\d{8}", date):
                raise ValueError("날짜는 YYYYMMDD 형식이어야 합니다.")
            datetime.strptime(date, "%Y%m%d")
    paths = [source / name for name in ARTIFACTS]
    paths += [source / name for name in ("fault_dataset.npz", "band_statistics.npz") if (source / name).is_file()]
    before = {p.name: sha256_file(p) for p in paths}
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "evaluation_label": "독립성 미확인 예비 평가",
              "supported_rpms": None, "web_activation_authorized": False,
              "runtime_versions": runtime_versions(), "source_sha256": before,
              "normal_conditions": list(normal_conditions), "fault_conditions": list(fault_conditions),
              "selected_dates": dates, "rows": [], "leakage_and_provenance": {}}
    rows = []
    try:
        engine = SavedDAEInference(source)  # Diagnostic adapter, no bundle/RPM setting writes.
        report["threshold"] = engine.threshold
        audit, known_files, known_features = inspect_datasets(source)
        report["leakage_and_provenance"] = audit
        rows, traversal_errors = discover(data_root, normal_conditions, fault_conditions, dates)
        report["traversal_errors"] = traversal_errors
        for row in rows:
            if row["excluded_reason"]:
                continue
            try:
                raw = Path(row["file"]).read_bytes()
                row["file_sha256"] = hashlib.sha256(raw).hexdigest()
                spectrum = preprocess_mat_bytes(raw)
                normalized = normalize_spectrum(spectrum, engine.mean, engine.std)
                fingerprint = digest_array(normalized)
                row["model_input_sha256"] = fingerprint
                signal = loadmat(BytesIO(raw), variable_names=("signals",))["signals"][2]
                row["ch3_sha256"] = hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float64).tobytes()).hexdigest()
                if file_id(row["file"]) in known_files:
                    raise ValueError("known_train_or_validation_file")
                if fingerprint in known_features:
                    raise ValueError("matching_train_or_validation_input")
            except Exception as error:
                row["excluded_reason"] = f"{type(error).__name__}: {error}"
        # Avoid counting exact copied signals as independent samples, and exclude
        # all copies with conflicting condition/RPM metadata rather than picking a label.
        content_groups = {}
        for row in rows:
            if row.get("ch3_sha256"):
                content_groups.setdefault(row["ch3_sha256"], []).append(row)
        for group in content_groups.values():
            if len({(r["rpm"], r["condition"]) for r in group}) > 1:
                for row in group:
                    if not row["excluded_reason"]:
                        row["excluded_reason"] = "duplicate_signal_conflicting_metadata"
            else:
                candidates = [r for r in group if not r["excluded_reason"]]
                for row in candidates[1:]:
                    row["excluded_reason"] = "duplicate_ch3_signal"
        for row in rows:
            if row["excluded_reason"]:
                continue
            try:
                raw = Path(row["file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["file_sha256"]:
                    raise ValueError("MAT_changed_during_evaluation")
                result = engine.predict_bytes(raw)
                row.update(mse=result["mse"], threshold=result["threshold"], alarm=int(result["mse"] > engine.threshold))
            except Exception as error:
                row["excluded_reason"] = f"{type(error).__name__}: {error}"
        report["metrics"] = aggregate(rows)
        report["counts"] = {"discovered": len(rows), "evaluated": sum(not r["excluded_reason"] for r in rows),
                            "excluded": sum(bool(r["excluded_reason"]) for r in rows)}
        report["exclusion_reasons"] = {reason: sum(r["excluded_reason"] == reason for r in rows)
                                       for reason in sorted({r["excluded_reason"] for r in rows if r["excluded_reason"]})}
        report["leakage_and_provenance"]["evaluation_overlap_exclusions"] = {
            "known_train_validation_files": sum("known_train_or_validation_file" in r["excluded_reason"] for r in rows),
            "matching_train_validation_inputs": sum("matching_train_or_validation_input" in r["excluded_reason"] for r in rows),
        }
        report["status"] = "preliminary_evaluation_completed"
        if traversal_errors or not report["counts"]["evaluated"]:
            raise ValueError("탐색 오류 또는 평가 성공 파일 0개; report.json을 확인하세요.")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.pop("rows", None)
        report["source_unchanged"] = all(p.is_file() and sha256_file(p) == before[p.name] for p in paths)
        if not report["source_unchanged"]:
            report["status"] = "failed"
            report["error"] = "원본 모델/데이터 파일 변경 감지"
        write_csv(output / "files.csv", rows)
        write_csv(output / "metrics.csv", report.get("metrics", []))
        (output / "provenance.json").write_text(
            json.dumps(report["leakage_and_provenance"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["source_unchanged"]:
        raise RuntimeError("원본 파일 변경 감지")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / "capstone")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path.home() / "dae-performance" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dates", nargs="+", help="YYYYMMDD; 생략하면 모든 날짜를 예비 평가")
    parser.add_argument("--normal-conditions", nargs="+", default=list(NORMAL))
    parser.add_argument("--fault-conditions", nargs="+", default=list(FAULT))
    args = parser.parse_args()
    try:
        report = evaluate(args.source, args.data_root, args.output, dates=args.dates,
                          normal_conditions=args.normal_conditions, fault_conditions=args.fault_conditions)
    except Exception as error:
        print(f"FAIL: {error}\n출력(생성된 경우): {args.output}", file=sys.stderr)
        return 1
    print("독립성 미확인 예비 평가 — 학습 출처/누출 미확인; 지원 RPM 확정 아님")
    for item in report["metrics"]:
        rate = "평가 불가" if item["rate"] is None else f"{item['rate']:.2%}"
        print(f"{item['rpm']}RPM {item['class']} {item['condition']}: {item['alarms']}/{item['evaluated']} = {rate}")
    print(f"보고서: {args.output}; 원본 불변={report['source_unchanged']}; 웹 차단 유지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
