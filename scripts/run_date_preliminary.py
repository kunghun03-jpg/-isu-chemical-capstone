"""사용자가 승인한 9/12 학습·9/16 검증·9/19 시험 예비 실험. 실제 Run 실험이 아닙니다."""

from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from vibration.dataset import discover_files, read_rows, validate_file_metadata
from vibration.split import run as split_run


def prepare(root, name):
    quality = root / "outputs" / "data_preparation"
    with (quality / "quality_all_files.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        quality_rows = list(csv.DictReader(stream))
    checked = {r["file"]: r for r in quality_rows}
    discovered = [r for rpm in (900, 1200, 1800) for r in discover_files(root, rpm)]
    if set(checked) != {r["file"] for r in discovered} or len(checked) != len(
        discovered
    ):
        raise ValueError("실제 원본 목록과 기존 품질검사 범위가 다릅니다.")
    folders = [root / "outputs" / f"{rpm}rpm" / name for rpm in (900, 1200, 1800)]
    if any(p.exists() for p in folders):
        raise ValueError("기존 실험 폴더가 있습니다. 다른 --run-name을 사용하세요.")
    selected = []
    for rpm, folder in zip((900, 1200, 1800), folders):
        folder.mkdir(parents=True)
        for pattern in ("quality_check", "quality_checked"):
            shutil.copyfile(
                quality / f"{pattern}_{rpm}rpm.csv", folder / f"{pattern}_{rpm}rpm.csv"
            )
        split_run(
            root,
            folder,
            rpm,
            mode="date_preliminary",
            dates=["20260912", "20260916", "20260919"],
        )
        selected.extend(read_rows(folder / f"split_{rpm}rpm.csv"))
    hashes, paths = {}, set()
    for index, row in enumerate(selected, 1):
        if (
            row["file"] in paths
            or row["condition"] == "Normal3"
            or checked[row["file"]]["status"] != "pass"
        ):
            raise ValueError("파일 중복 또는 제외 대상 혼입")
        paths.add(row["file"])
        path = validate_file_metadata(root, row, int(row["rpm"]))
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != checked[row["file"]]["sha256"]:
            raise ValueError(f"품질검사 이후 원본이 달라졌습니다: {row['file']}")
        if checksum in hashes:
            raise ValueError(f"동일 내용 파일 중복: {hashes[checksum]} / {row['file']}")
        hashes[checksum] = row["file"]
        if index % 100 == 0:
            print(f"분할·SHA256 중복 검사 {index}/{len(selected)}", flush=True)
    audit = {
        "experiment_mode": "date_preliminary",
        "group_type": "date_not_actual_run",
        "run_independence_verified": False,
        "file_count": len(selected),
        "duplicate_paths": 0,
        "duplicate_content": 0,
        "quality_hash_matches": len(selected),
        "counts": {
            str(rpm): dict(
                Counter(r["split"] for r in selected if r["rpm"] == str(rpm))
            )
            for rpm in (900, 1200, 1800)
        },
    }
    for rpm, folder in zip((900, 1200, 1800), folders):
        (folder / "preflight.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # split 단계 완료 기록: 이후 기존 CLI를 동일 폴더에 안전하게 연결합니다.
        (folder / "run_log.json").write_text(
            json.dumps(
                [
                    {
                        "stage": "split",
                        "rpm": rpm,
                        "split_mode": "date_preliminary",
                        "note": "user-authorized date groups; not actual Run IDs",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
    print("세 RPM 전체 분할·중복·품질검사 검증 통과", audit["counts"], flush=True)
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="date_prelim_20261106")
    args = parser.parse_args()
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_name
    ):
        parser.error("잘못된 실험 이름")
    root = Path(__file__).resolve().parents[1]
    folders = prepare(root, args.run_name)
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    timings = []
    for rpm, folder in zip((900, 1200, 1800), folders):
        for stage in (
            "spectra",
            "normalize",
            "train",
            "evaluate",
            "file_evaluate",
            "ocsvm",
            "iforest",
            "compare",
        ):
            started = time.perf_counter()
            command = [
                sys.executable,
                "-B",
                str(root / "run_project.py"),
                "--rpm",
                str(rpm),
                "--stage",
                stage,
                "--run-name",
                args.run_name,
                "--split-mode",
                "date_preliminary",
            ]
            print(f"시작: {rpm}RPM / {stage}", flush=True)
            with (folder / f"{stage}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            timings.append(
                {
                    "rpm": rpm,
                    "stage": stage,
                    "wall_seconds": time.perf_counter() - started,
                    "exit_code": result.returncode,
                }
            )
            (folder / "stage_timings.json").write_text(
                json.dumps([r for r in timings if r["rpm"] == rpm], indent=2),
                encoding="utf-8",
            )
            if result.returncode:
                raise RuntimeError(
                    f"{rpm}RPM {stage} 실패. {folder / (stage + '.log')} 확인"
                )
            print(
                f"완료: {rpm}RPM / {stage} ({timings[-1]['wall_seconds']:.1f}s)",
                flush=True,
            )
    summaries = {
        str(rpm): json.loads(
            (folder / "comparison_summary.json").read_text(encoding="utf-8")
        )
        for rpm, folder in zip((900, 1200, 1800), folders)
    }
    (root / "outputs" / f"{args.run_name}_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("날짜 그룹 예비 실험 3개 RPM × 3개 모델 완료", flush=True)


if __name__ == "__main__":
    main()
