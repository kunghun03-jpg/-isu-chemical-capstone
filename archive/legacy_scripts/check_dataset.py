# 보관된 초기 실험 코드: check_dataset. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "20260916_900RPM"

total = 0

for folder in sorted(data_dir.iterdir()):
    if folder.is_dir():
        count = len(list(folder.rglob("*.mat")))
        total += count
        print(f"{folder.name}: {count}개")

print(f"\n전체 MAT 파일: {total}개")

from collections import Counter
from scipy.io import loadmat
import numpy as np

shapes = Counter()
sampling_rates = Counter()
issues = []

all_files = sorted(data_dir.rglob("*.mat"))

for i, path in enumerate(all_files, start=1):
    try:
        data = loadmat(path)
        signals = data["signals"]
        fs = float(data["fs"].item())

        shapes[signals.shape] += 1
        sampling_rates[fs] += 1

        if not np.isfinite(signals).all():
            issues.append((str(path.relative_to(data_dir)), "NaN 또는 무한대 있음"))

    except Exception as e:
        issues.append((str(path.relative_to(data_dir)), str(e)))

    if i % 100 == 0 or i == len(all_files):
        print(f"검사 진행: {i}/{len(all_files)}")

print("\n데이터 모양별 파일 수:", dict(shapes))
print("샘플링 주파수별 파일 수:", dict(sampling_rates))
print("문제 있는 파일 수:", len(issues))

for name, reason in issues:
    print(name, "→", reason)


print("\n문제 파일의 채널별 상세 확인")

for name, _ in issues:
    path = data_dir / name
    signals = loadmat(path)["signals"]

    print(f"\n파일: {name}")

    for ch in range(signals.shape[0]):
        values = signals[ch]
        nan_count = np.isnan(values).sum()
        inf_count = np.isinf(values).sum()

        if nan_count + inf_count > 0:
            print(
                f"  CH {ch + 1}: "
                f"NaN {nan_count}개, "
                f"무한대 {inf_count}개 / 전체 {values.size}개"
            )


print("\nNaN 보정 확인")

for name, _ in issues:
    signals = loadmat(data_dir / name)["signals"].copy()

    for ch in range(signals.shape[0]):
        values = signals[ch]
        missing = np.isnan(values)

        if missing.any():
            positions = np.arange(values.size)
            valid = ~missing

            values[missing] = np.interp(
                positions[missing], positions[valid], values[valid]
            )

    remaining = np.count_nonzero(~np.isfinite(signals))
    print(f"{name}: 보정 후 비정상 숫자 {remaining}개")
