"""주파수별 정규화 기준의 기존 세 모델을 내보냅니다. 학습은 하지 않습니다."""

from pathlib import Path
import argparse
import json

from scripts.analyze_date_score_distributions import snapshot
from vibration.inference_bundle import export_bundle


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
    destination = root / "outputs/inference_bundles" / args.run_name
    if destination.exists():
        raise ValueError("기존 패키지는 덮어쓰지 않습니다.")
    sources = [p for p in (root / "outputs").rglob("*") if p.is_file()]
    before = snapshot(sources)
    destination.mkdir(parents=True)
    try:
        for rpm in (900, 1200, 1800):
            manifest = export_bundle(
                root / "outputs" / f"{rpm}rpm" / args.run_name,
                destination / f"{rpm}rpm",
                rpm,
            )
            print(
                f"{rpm}RPM: 기존 AE·OCSVM·IF + 주파수별 정규화 + 저장 임계값 패키지 생성 ({manifest['bundle_id'][:12]})",
                flush=True,
            )
    finally:
        after = snapshot(sources)
        (destination / "preservation.json").write_text(
            json.dumps(
                {
                    "unchanged": before == after,
                    "existing_output_files": len(before),
                    "before_sha256": before,
                    "after_sha256": after,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if before != after:
            raise RuntimeError("기존 결과 변경 발견")


if __name__ == "__main__":
    main()
