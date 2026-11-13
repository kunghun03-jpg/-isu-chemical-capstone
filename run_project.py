# 명령행에서 RPM·실험 이름·처리 단계를 받아 해당 모듈을 실행합니다.

"""RPM별 검사·전처리·학습·평가 실행 진입점."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib
import json
import sys

# 처리 순서: 검사 → 분할 → 스펙트럼 → 정규화 → 학습 → 평가 및 비교
STAGES = (
    "inventory",
    "check",
    "split",
    "spectra",
    "normalize",
    "train",
    "evaluate",
    "file_evaluate",
    "baseline",
    "features",
    "auxiliary",
    "combined",
    "ocsvm",
    "iforest",
    "compare",
)

# all은 개발가이드의 기본 오토엔코더 흐름만 실행합니다.
# 평균 스펙트럼 비교와 회전 주파수 보조 분석은 개별 --stage로 선택합니다.
CORE_STAGES = (
    "check",
    "split",
    "spectra",
    "normalize",
    "train",
    "evaluate",
    "file_evaluate",
)


def main():

    # 1. 실행 옵션을 읽습니다. 새 실험은 --run-name으로 구분합니다.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpm", type=int, choices=(900, 1200, 1800), required=True)
    parser.add_argument("--stage", choices=STAGES + ("all",), required=True)
    parser.add_argument("--normalization", choices=("per_frequency", "global_train"),
                        default="per_frequency", help="기본 주파수별 또는 RPM별 Train 공통 정규화")
    parser.add_argument(
        "--dates", nargs="+", help="대상 날짜 YYYYMMDD 목록. 생략하면 모든 날짜"
    )
    parser.add_argument(
        "--split-mode",
        choices=("run", "file", "date_preliminary"),
        default="run",
        help="run: 확인된 Run 목록. file: 시간순 비율. date_preliminary: 별도 날짜 그룹 예비 실험",
    )
    parser.add_argument(
        "--run-manifest", type=Path, help="확정 Run ID와 split을 기록한 CSV"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="data 폴더가 있는 isu_chemical 경로",
    )
    parser.add_argument(
        "--run-name",
        default="ch3_runs",
        help="CH3 실험 이름. 새 실험에는 다른 이름 사용",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="동일 단계 재실행을 명시적으로 허용"
    )
    args = parser.parse_args()
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_name
    ):
        parser.error("--run-name에는 영문·숫자·밑줄·하이픈만 사용하세요.")

    # 2. RPM별 실험 폴더와 실행 기록 경로를 준비합니다.
    root = args.project_root.resolve()
    output = root / "outputs" / f"{args.rpm}rpm" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run_log.json"
    if not args.overwrite and not log_path.exists() and any(output.iterdir()):
        parser.error(
            f"기존 결과가 있지만 실행 기록이 없습니다: {output}\n"
            "새 --run-name을 사용하거나, 기존 결과를 바꾸려면 --overwrite를 지정하세요."
        )
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    stages = CORE_STAGES if args.stage == "all" else (args.stage,)
    manifest = args.run_manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = root / manifest
    if "split" in stages and args.split_mode == "run" and manifest is None:
        parser.error(
            "Run 분할에는 --run-manifest가 필요합니다. 먼저 --stage inventory로 목록을 만드세요."
        )

    # 3. 모듈을 실행하고 성공한 단계만 기록합니다.
    for stage in stages:
        if not args.overwrite and any(r["stage"] == stage for r in log):
            parser.error(
                f"{stage} 실행 기록이 있습니다. 새 --run-name을 사용하거나 --overwrite를 지정하세요."
            )
        print(f"\n=== {args.rpm}RPM / {stage} ===", flush=True)
        module = importlib.import_module(
            "vibration.split"
            if stage == "inventory"
            else (
                "vibration.comparison_metrics"
                if stage == "compare"
                else f"vibration.{stage}"
            )
        )
        try:
            if stage == "inventory":
                module.inventory(root, output, args.rpm, dates=args.dates)
            elif stage == "split":
                module.run(
                    root,
                    output,
                    args.rpm,
                    mode=args.split_mode,
                    manifest=manifest,
                    dates=args.dates,
                )
            elif stage == "check":
                module.run(root, output, args.rpm, dates=args.dates)
            elif stage == "normalize":
                module.run(root, output, args.rpm, mode=args.normalization)
            else:
                module.run(root, output, args.rpm)
        except Exception:
            print(f"중단된 단계: {stage}\n결과 폴더: {output}", file=sys.stderr)
            raise
        log.append(
            {
                "stage": stage,
                "rpm": args.rpm,
                "dates": args.dates,
                "split_mode": args.split_mode,
                "normalization_requested": args.normalization,
                "run_manifest": str(manifest) if manifest else None,
                "python": sys.version,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        log_path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\n완료: {output}")


if __name__ == "__main__":
    main()
