"""날짜·RPM별 원본 경로 탐색과 명시적인 Run 목록 검사."""

from datetime import datetime
from pathlib import Path
import csv
import re

NORMAL_CONDITIONS = {"Normal1", "Normal2", "Normal3"}
DEFAULT_EXCLUDED_CONDITIONS = {"Normal3"}
CONDITION_ALIASES = {
    "Looeness0.05": "Looseness0.05",
    "Looeness0.1": "Looseness0.1",
    "LooenessFull": "LoosenessFull",
}
SPLITS = {"train", "validation", "test_normal", "test_anomaly"}
FIELDS = ["file", "date", "rpm", "condition", "run_id", "split"]
DATA_FOLDER = re.compile(r"(?P<date>\d{8})_(?P<rpm>\d+)RPM")


def canonical_condition(name):
    return CONDITION_ALIASES.get(name, name)


def path_metadata(file):
    """기존 평면 경로와 날짜 상위 폴더 경로를 모두 해석합니다."""
    path = Path(file)
    parts = path.parts
    if path.is_absolute() or ".." in parts or len(parts) < 4 or parts[0] != "data":
        raise ValueError(f"잘못된 프로젝트 상대 경로: {path}")
    index = 2 if re.fullmatch(r"\d{8}", parts[1]) else 1
    if len(parts) <= index + 2:
        raise ValueError(f"측정 상태가 없는 경로: {path}")
    match = DATA_FOLDER.fullmatch(parts[index])
    if not match or (index == 2 and parts[1] != match["date"]):
        raise ValueError(f"날짜·RPM 폴더 확인 필요: {path}")
    datetime.strptime(match["date"], "%Y%m%d")
    return {
        "date": match["date"],
        "rpm": match["rpm"],
        "condition": canonical_condition(parts[index + 1]),
        "source_condition": parts[index + 1],
    }


def write_inventory(path, rows):
    """정상 여부·기본 제외 정책과 원본 표기를 기록하되 Run은 확정하지 않습니다."""
    enriched = []
    for row in rows:
        info = path_metadata(row["file"])
        enriched.append(
            {
                **row,
                "source_condition": info["source_condition"],
                "is_normal": int(row["condition"] in NORMAL_CONDITIONS),
                "default_included": int(
                    row["condition"] not in DEFAULT_EXCLUDED_CONDITIONS
                ),
            }
        )
    write_rows(
        path, enriched, FIELDS + ["source_condition", "is_normal", "default_included"]
    )


def discover_files(project_dir, rpm, dates=None):
    """두 경로 구조를 지원하며 Normal3도 집계합니다. Run은 추정하지 않습니다."""
    project_dir = Path(project_dir).resolve()
    dates = set(dates) if dates else None
    records = []
    found_dates = set()
    data_root = project_dir / "data"
    candidates = list(data_root.glob(f"*_{rpm}RPM")) + list(
        data_root.glob(f"*/*_{rpm}RPM")
    )
    for folder in sorted(set(candidates)):
        match = DATA_FOLDER.fullmatch(folder.name)
        if not folder.is_dir() or not match:
            continue
        date = match["date"]
        datetime.strptime(date, "%Y%m%d")
        if folder.parent != data_root and folder.parent.name != date:
            raise ValueError(f"상위 날짜와 RPM 폴더 날짜가 다릅니다: {folder}")
        if dates is not None and date not in dates:
            continue
        found_dates.add(date)
        for condition in sorted(folder.iterdir()):
            if not condition.is_dir():
                continue
            for path in sorted(condition.rglob("*.mat")):
                if not path.resolve().is_relative_to(project_dir / "data"):
                    raise ValueError(f"data 외부를 가리키는 경로: {path}")
                records.append(
                    {
                        "file": path.relative_to(project_dir).as_posix(),
                        "date": date,
                        "rpm": str(rpm),
                        "condition": canonical_condition(condition.name),
                        "run_id": "",
                        "split": "",
                    }
                )
    if dates is not None and dates - found_dates:
        raise ValueError(f"데이터 폴더가 없는 날짜: {sorted(dates - found_dates)}")
    if not records:
        raise ValueError(f"{rpm}RPM MAT 파일이 없습니다.")
    return records


def write_rows(path, rows, fields=FIELDS):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"목록에 필요한 열: {', '.join(FIELDS)}")
        keys = FIELDS + (["date_group"] if "date_group" in reader.fieldnames else [])
        return [{key: (row[key] or "").strip() for key in keys} for row in reader]


def date_preliminary_rows(records):
    """사용자가 지정한 날짜 예비 실험. 실제 Run ID는 비워 둡니다."""
    result = []
    for record in records:
        date, condition = record["date"], record["condition"]
        if condition in DEFAULT_EXCLUDED_CONDITIONS:
            continue
        normal = condition in {"Normal1", "Normal2"}
        if normal and date in {"20260912", "20260916", "20260919"}:
            assigned = {
                "20260912": "train",
                "20260916": "validation",
                "20260919": "test_normal",
            }[date]
        elif not normal and date == "20260919":
            assigned = "test_anomaly"
        else:
            continue
        result.append(
            {**record, "run_id": "", "date_group": f"date:{date}", "split": assigned}
        )
    return result


def validate_date_groups(rows):
    validate_assignments(rows, require_run=False)
    if not rows:
        raise ValueError("날짜 예비 실험 데이터가 없습니다.")
    for row in rows:
        expected = date_preliminary_rows([row])
        if (
            row["run_id"]
            or not expected
            or row.get("date_group") != expected[0]["date_group"]
            or row["split"] != expected[0]["split"]
        ):
            raise ValueError(
                f"날짜 예비 실험 배정 불일치 또는 실제 Run ID 혼용: {row['file']}"
            )


def validate_assignments(rows, require_run=True):
    """Run ID는 전역 식별자입니다. 날짜·RPM으로 다시 묶어 중복을 숨기지 않습니다."""
    seen_files = set()
    run_splits = {}
    for row in rows:
        file = row["file"]
        if row["condition"] in DEFAULT_EXCLUDED_CONDITIONS:
            raise ValueError(
                f"Normal3는 정상 상태이지만 기본 실험에서 제외합니다: {file}"
            )
        if file in seen_files:
            raise ValueError(f"중복 파일: {file}")
        seen_files.add(file)
        split = row["split"]
        if split not in SPLITS:
            raise ValueError(f"미확정 또는 잘못된 split: {file}: {split!r}")
        normal = row["condition"] in NORMAL_CONDITIONS
        if (normal and split == "test_anomaly") or (
            not normal and split != "test_anomaly"
        ):
            raise ValueError(f"정상·결함 상태와 split이 일치하지 않습니다: {file}")
        run_id = row.get("run_id", "").strip()
        if require_run and (
            not run_id
            or run_id.lower() in {"unknown", "pending", "tbd", "unconfirmed", "미확정"}
        ):
            raise ValueError(f"Run ID 미확정: {file}")
        if run_id:
            if run_id in run_splits and run_splits[run_id] != split:
                raise ValueError(
                    f"동일 Run 중복 배정: {run_id}: {run_splits[run_id]} / {split}"
                )
            run_splits[run_id] = split


def validate_file_metadata(project_dir, row, rpm):
    """목록의 날짜·RPM·상태가 실제 프로젝트 내부 경로와 일치하는지 확인합니다."""
    root = Path(project_dir).resolve()
    path = Path(row["file"])
    info = path_metadata(row["file"])
    if int(info["rpm"]) != int(rpm):
        raise ValueError(f"RPM 또는 날짜 폴더 확인 필요: {path}")
    if row.get("date", info["date"]) != info["date"] or str(row.get("rpm", rpm)) != str(
        rpm
    ):
        raise ValueError(f"날짜·RPM 메타데이터 불일치: {path}")
    if row["condition"] != info["condition"]:
        raise ValueError(f"상태 메타데이터 불일치: {path}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root / "data") or not resolved.is_file():
        raise ValueError(f"원본 파일 누락 또는 data 외부 경로: {path}")
    return resolved
