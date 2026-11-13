"""확정 Run 목록에 따른 분할과 비교용 파일 비율 분할."""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json

from vibration.dataset import (
    NORMAL_CONDITIONS,
    DEFAULT_EXCLUDED_CONDITIONS,
    discover_files,
    read_rows,
    validate_assignments,
    validate_file_metadata,
    write_rows,
    write_inventory,
    date_preliminary_rows,
    validate_date_groups,
    FIELDS,
)


def inventory(project_dir, output_dir, rpm, dates=None):
    # 날짜·RPM·상태만 경로에서 읽습니다. Run ID와 split은 미확정 상태로 둡니다.
    rows = discover_files(project_dir, rpm, dates)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"run_inventory_{rpm}rpm.csv"
    write_inventory(path, rows)
    print(f"Run 미확정 목록 저장: {path} ({len(rows)}개)")
    return path


def run(project_dir, output_dir, rpm, *, mode="run", manifest=None, dates=None):
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    if mode not in {"run", "file", "date_preliminary"}:
        raise ValueError(f"지원하지 않는 분할 방식: {mode}")
    if mode == "run" and manifest is None:
        raise ValueError(
            "Run 분할에는 확인된 --run-manifest가 필요합니다. inventory로 목록부터 만드세요."
        )
    discovered = discover_files(project_dir, rpm, dates)
    discovered = [
        row for row in discovered if row["condition"] not in DEFAULT_EXCLUDED_CONDITIONS
    ]
    available = {row["file"]: row for row in discovered}
    quality_path = output_dir / f"quality_check_{rpm}rpm.csv"
    with quality_path.open(encoding="utf-8-sig", newline="") as stream:
        excluded = {row["file"] for row in csv.DictReader(stream)}

    if mode == "run":
        rows = read_rows(manifest)
        # 통합 목록의 Normal3는 빈 배정으로 유지하고 기본 실험에서는 빼 둡니다.
        for row in rows:
            if row["condition"] in DEFAULT_EXCLUDED_CONDITIONS and row["split"]:
                raise ValueError("Normal3에는 기본 실험 split을 배정하지 마세요.")
        rows = [
            row for row in rows if row["condition"] not in DEFAULT_EXCLUDED_CONDITIONS
        ]
        # 선택 날짜 밖이나 다른 RPM에 걸친 동일 Run 충돌도 먼저 검사합니다.
        validate_assignments(rows)
        for row in rows:
            validate_file_metadata(project_dir, row, int(row["rpm"]))
        selected = [row for row in rows if row["file"] in available]
        if {row["file"] for row in selected} != set(available):
            missing = sorted(set(available) - {row["file"] for row in selected})
            raise ValueError(f"Run 목록에서 원본 파일이 누락되었습니다: {missing[:5]}")
        # 새 날짜를 추가한 뒤 과거 품질검사 결과를 그대로 쓰는 실수를 막습니다.
        coverage_path = output_dir / f"quality_checked_{rpm}rpm.csv"
        if not coverage_path.exists():
            raise ValueError(
                "Run 분할 전 check를 다시 실행하여 품질검사 대상 목록을 생성하세요."
            )
        with coverage_path.open(encoding="utf-8-sig", newline="") as stream:
            checked = {row["file"] for row in csv.DictReader(stream)}
        if not set(available).issubset(checked):
            raise ValueError(
                "품질검사를 하지 않은 파일이 있습니다. 같은 --dates로 check를 실행하세요."
            )
        rows = [row for row in selected if row["file"] not in excluded]
    elif mode == "date_preliminary":
        if manifest is not None:
            raise ValueError(
                "날짜 그룹 실험에는 실제 Run manifest를 사용하지 않습니다."
            )
        coverage = read_rows(output_dir / f"quality_checked_{rpm}rpm.csv")
        if not set(available).issubset({r["file"] for r in coverage}):
            raise ValueError("날짜 실험에 미검사 파일이 있습니다.")
        rows = date_preliminary_rows(
            [row for row in discovered if row["file"] not in excluded]
        )
        validate_date_groups(rows)
    else:
        if manifest is not None:
            raise ValueError("file 모드에서는 Run 목록을 사용하지 않습니다.")
        # 종전 시간순 60/20/20을 날짜·RPM·상태별로 적용합니다. Run 독립성을 보장하지 않습니다.
        groups = defaultdict(list)
        for row in discovered:
            if row["file"] not in excluded:
                groups[(row["date"], row["rpm"], row["condition"])].append(dict(row))
        rows = []
        for group in groups.values():
            n_train, n_val = int(len(group) * 0.6), int(len(group) * 0.2)
            for index, row in enumerate(group):
                if row["condition"] not in NORMAL_CONDITIONS:
                    row["split"] = "test_anomaly"
                elif index < n_train:
                    row["split"] = "train"
                elif index < n_train + n_val:
                    row["split"] = "validation"
                else:
                    row["split"] = "test_normal"
                rows.append(row)
        print("주의: file 모드는 동일 Run의 분할 간 독립성을 보장하지 않습니다.")

    validate_assignments(rows, require_run=mode == "run")
    for split in ("train", "validation", "test_normal"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"{split} 데이터가 없습니다. Run 수와 배정을 확인하세요.")
    path = output_dir / f"split_{rpm}rpm.csv"
    write_rows(
        path, rows, FIELDS + (["date_group"] if mode == "date_preliminary" else [])
    )
    info = {
        "mode": mode,
        "dates": sorted({row["date"] for row in rows}),
        "manifest": str(Path(manifest).resolve()) if manifest else None,
        "run_independence_verified": False,
        "group_type": (
            "date_not_actual_run"
            if mode == "date_preliminary"
            else "run" if mode == "run" else "file"
        ),
        "note": (
            "User-authorized date groups, not actual Run IDs; independent Runs remain unverified."
            if mode == "date_preliminary"
            else "Run assignments are user supplied; physical independence requires acquisition records."
        ),
    }
    # Run ID 충돌 검사와 실제 운전의 독립성 증명을 구분합니다.
    with (output_dir / f"split_config_{rpm}rpm.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(info, stream, ensure_ascii=False, indent=2)
    for split, count in sorted(Counter(row["split"] for row in rows).items()):
        print(f"{split}: {count}개")
    print(f"분할 방식: {mode}, 목록 저장: {path}")
