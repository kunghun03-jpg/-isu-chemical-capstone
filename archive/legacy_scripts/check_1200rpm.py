# 보관된 초기 실험 코드: check_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import Counter
from scipy.io import loadmat
import numpy as np
import csv

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data" / "20260916_1200RPM"
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"
output_dir.mkdir(exist_ok=True)

shapes = Counter()
sampling_rates = Counter()
issues = []
total_files = 0

for folder in sorted(data_dir.iterdir()):
    if not folder.is_dir():
        continue

    files = sorted(folder.rglob("*.mat"))
    suspect_count = 0
    read_error_count = 0

    for path in files:
        relative_path = path.relative_to(project_dir).as_posix()

        try:
            data = loadmat(path)
            signals = data["signals"]
            fs = float(data["fs"].item())

        except Exception as error:
            read_error_count += 1

            issues.append(
                {
                    "file": relative_path,
                    "reason": "read_error",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )

            print(
                f"  읽기 실패: {path.relative_to(data_dir)}\n"
                f"    {type(error).__name__}: {error}"
            )
            continue

        shapes[signals.shape] += 1
        sampling_rates[fs] += 1

        nonfinite = ~np.isfinite(signals)
        extreme = np.isfinite(signals) & (np.abs(signals) > 1e100)

        reasons = []
        details = []

        if signals.shape != (6, 65536) or fs != 25600.0:
            reasons.append("unexpected_format")
            details.append(f"shape={signals.shape}, fs={fs}")

        if nonfinite.any() or extreme.any():
            reasons.append("suspect_values")
            details.append(
                f"NaN·무한대={int(nonfinite.sum())}, " f"극단값={int(extreme.sum())}"
            )

        if reasons:
            suspect_count += 1

            issues.append(
                {
                    "file": relative_path,
                    "reason": "; ".join(reasons),
                    "detail": "; ".join(details),
                }
            )

            print(
                f"  확인 필요: {path.relative_to(data_dir)} " f"| {'; '.join(details)}"
            )

    total_files += len(files)

    print(
        f"\n{folder.name}: 전체 {len(files)}개, "
        f"읽기 실패 {read_error_count}개, "
        f"값·형식 확인 필요 {suspect_count}개\n"
    )

report_path = output_dir / "quality_check_1200rpm.csv"

with report_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "reason", "detail"])
    writer.writeheader()
    writer.writerows(issues)

print("전체 파일:", total_files, "개")
print("읽기 성공:", sum(shapes.values()), "개")
print("문제 후보 파일:", len(issues), "개")
print("데이터 모양별 파일 수:", dict(shapes))
print("샘플링 주파수별 파일 수:", dict(sampling_rates))
print("검사 기록 저장:", report_path)
