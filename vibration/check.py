"""원본을 읽기 전용으로 검사하고 전체 파일별 결과와 제외 후보를 기록합니다."""

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.io import loadmat

from vibration.dataset import (
    NORMAL_CONDITIONS,
    DEFAULT_EXCLUDED_CONDITIONS,
    discover_files,
    path_metadata,
    write_rows,
)


def inspect_file(project_dir, row):
    path = Path(project_dir) / row["file"]
    result = {
        **row,
        "source_condition": path_metadata(row["file"])["source_condition"],
        "is_normal": int(row["condition"] in NORMAL_CONDITIONS),
        "default_included": int(row["condition"] not in DEFAULT_EXCLUDED_CONDITIONS),
        "status": "pass",
        "reason": "",
        "detail": "",
        "shape": "",
        "dtype": "",
        "fs": "",
        "nan_count": 0,
        "inf_count": 0,
        "extreme_count": 0,
        "ch3_nonfinite_count": 0,
        "ch3_extreme_count": 0,
        "sha256": "",
        "source_stat_unchanged": 0,
    }
    reasons, details = [], []
    before = path.stat()
    try:
        # 원본은 한 번 읽고, scipy에는 메모리 버퍼를 전달합니다. 원본 쓰기는 없습니다.
        raw = path.read_bytes()
        result["sha256"] = hashlib.sha256(raw).hexdigest()
        data = loadmat(BytesIO(raw))
    except Exception as error:
        reasons.append("read_error")
        details.append(f"{type(error).__name__}: {error}")
        data = None
    if data is not None:
        signals = data.get("signals")
        sampling = data.get("fs")
        if signals is None or sampling is None:
            reasons.append("missing_fields")
            details.append("signals 또는 fs 없음")
        else:
            result["shape"] = str(signals.shape)
            result["dtype"] = str(signals.dtype)
            valid_type = signals.dtype.kind in "fiu"
            valid_fs = sampling.size == 1 and sampling.dtype.kind in "fiu"
            fs = float(sampling.item()) if valid_fs else None
            result["fs"] = fs if fs is not None else "invalid"
            if signals.shape != (6, 65536) or not valid_type or fs != 25600.0:
                reasons.append("unexpected_format")
                details.append(f"shape={signals.shape}, dtype={signals.dtype}, fs={fs}")
            if valid_type:
                result["nan_count"] = int(np.isnan(signals).sum())
                result["inf_count"] = int(np.isinf(signals).sum())
                result["extreme_count"] = int(
                    (np.isfinite(signals) & (np.abs(signals) > 1e100)).sum()
                )
                if signals.ndim == 2 and signals.shape[0] >= 3:
                    ch3 = signals[2]
                    result["ch3_nonfinite_count"] = int((~np.isfinite(ch3)).sum())
                    result["ch3_extreme_count"] = int(
                        (np.isfinite(ch3) & (np.abs(ch3) > 1e100)).sum()
                    )
                if (
                    result["nan_count"]
                    or result["inf_count"]
                    or result["extreme_count"]
                ):
                    reasons.append("suspect_values")
                    details.append(
                        f"NaN={result['nan_count']}, Inf={result['inf_count']}, extreme={result['extreme_count']}"
                    )
    after = path.stat()
    result["source_stat_unchanged"] = int(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    )
    if not result["source_stat_unchanged"]:
        reasons.append("source_changed_during_check")
    result.update(
        status="fail" if reasons else "pass",
        reason="; ".join(reasons),
        detail="; ".join(details),
    )
    return result


def summarize(details):
    groups = defaultdict(list)
    for row in details:
        groups[(row["date"], row["rpm"], row["condition"])].append(row)
    by_condition = []
    for (date, rpm, condition), rows in sorted(groups.items()):
        passed = sum(row["status"] == "pass" for row in rows)
        by_condition.append(
            {
                "date": date,
                "rpm": rpm,
                "condition": condition,
                "total": len(rows),
                "passed": passed,
                "failed": len(rows) - passed,
                "is_normal": condition in NORMAL_CONDITIONS,
                "default_included": condition not in DEFAULT_EXCLUDED_CONDITIONS,
            }
        )
    return {
        "total": len(details),
        "passed": sum(r["status"] == "pass" for r in details),
        "failed": sum(r["status"] == "fail" for r in details),
        "normal_total": sum(r["is_normal"] for r in details),
        "normal3_total": sum(r["condition"] == "Normal3" for r in details),
        "default_normal_total": sum(
            r["is_normal"] and r["default_included"] for r in details
        ),
        "default_normal_passed": sum(
            r["is_normal"] and r["default_included"] and r["status"] == "pass"
            for r in details
        ),
        "reason_counts": dict(
            Counter(
                reason for r in details for reason in r["reason"].split("; ") if reason
            )
        ),
        "by_condition": by_condition,
        "policy": "Inspect all six raw channels, report CH3 counts separately; Normal3 normal but excluded from default experiments.",
    }


def run(project_dir, output_dir, rpm, *, dates=None):
    records = discover_files(project_dir, rpm, dates)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details = []
    for index, row in enumerate(records, 1):
        details.append(inspect_file(project_dir, row))
        if index % 100 == 0 or index == len(records):
            print(f"{rpm}RPM 품질검사: {index}/{len(records)}", flush=True)
    issues = [
        {key: row[key] for key in ("file", "reason", "detail")}
        for row in details
        if row["status"] == "fail"
    ]
    write_rows(
        output_dir / f"quality_check_{rpm}rpm.csv", issues, ["file", "reason", "detail"]
    )
    write_rows(output_dir / f"quality_details_{rpm}rpm.csv", details, list(details[0]))
    write_rows(output_dir / f"quality_checked_{rpm}rpm.csv", records)
    summary = summarize(details)
    (output_dir / f"quality_summary_{rpm}rpm.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{rpm}RPM: 전체 {summary['total']}, 통과 {summary['passed']}, 문제 후보 {summary['failed']}",
        flush=True,
    )
    return details
