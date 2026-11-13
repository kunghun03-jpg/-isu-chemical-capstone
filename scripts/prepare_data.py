"""전체 데이터 품질검사와 미확정 Run 목록만 준비합니다. 모델은 실행하지 않습니다."""

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import json

from vibration.check import run, summarize
from vibration.dataset import (
    NORMAL_CONDITIONS,
    discover_files,
    write_inventory,
    write_rows,
)


def write_measurement_groups(report, records):
    """폴더별 시간 범위는 Run 확인의 참고 정보일 뿐 독립 Run ID가 아닙니다."""
    groups = defaultdict(list)
    for row in records:
        if row["condition"] in NORMAL_CONDITIONS:
            groups[(row["date"], row["rpm"], row["condition"])].append(row)
    result = []
    for (date, rpm, condition), rows in sorted(groups.items()):
        times = []
        for row in rows:
            try:
                times.append(
                    datetime.strptime(Path(row["file"]).stem[:15], "%Y%m%d_%H%M%S")
                )
            except ValueError:
                pass
        times.sort()
        gaps = Counter((b - a).total_seconds() for a, b in zip(times, times[1:]))
        result.append(
            {
                "date": date,
                "rpm": rpm,
                "condition": condition,
                "file_count": len(rows),
                "parsed_timestamps": len(times),
                "first_record": times[0].isoformat() if times else "",
                "last_record": times[-1].isoformat() if times else "",
                "gap_seconds_counts": json.dumps(dict(gaps), sort_keys=True),
                "run_id": "",
                "independence_confirmed": 0,
            }
        )
    write_rows(Path(report) / "normal_measurement_groups.csv", result, list(result[0]))


def main():
    root = Path(__file__).resolve().parents[1]
    report = root / "outputs" / "data_preparation"
    report.mkdir(parents=True, exist_ok=True)
    records = [r for rpm in (900, 1200, 1800) for r in discover_files(root, rpm)]
    actual = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.mat")}
    if {r["file"] for r in records} != actual or len(records) != len(actual):
        raise ValueError(
            "탐색 대상과 실제 전체 MAT 목록이 다릅니다. 중복·미인식 경로를 확인하세요."
        )
    # 원본 목록만 갱신합니다. confirmed 목록은 만들거나 덮어쓰지 않습니다.
    write_inventory(root / "manifests" / "run_inventory_unconfirmed.csv", records)
    write_measurement_groups(report, records)
    details = []
    for rpm in (900, 1200, 1800):
        details.extend(run(root, report, rpm))
    write_rows(report / "quality_all_files.csv", details, list(details[0]))
    issues = [r for r in details if r["status"] == "fail"]
    write_rows(report / "quality_issues.csv", issues, list(details[0]))
    summary = summarize(details)
    (report / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_rows(
        report / "counts_by_condition.csv",
        summary["by_condition"],
        list(summary["by_condition"][0]),
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "by_condition"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
