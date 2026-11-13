# 보관된 초기 실험 코드: inspect_problem. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from scipy.io import loadmat
import numpy as np

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "20260916_900RPM"

# 센서의 이상 판정 기준이 아니라,
# 극단적인 숫자 손상 후보를 찾기 위한 임시 기준
CHECK_LIMIT = 1e100

problem_files = 0

for path in sorted(data_dir.rglob("*.mat")):
    signals = loadmat(path)["signals"]

    nonfinite = ~np.isfinite(signals)
    extreme = np.isfinite(signals) & (np.abs(signals) > CHECK_LIMIT)
    bad = nonfinite | extreme

    if not bad.any():
        continue

    problem_files += 1
    print(f"\n파일: {path.relative_to(data_dir)}")
    print("NaN·무한대:", np.count_nonzero(nonfinite))
    print("극단값:", np.count_nonzero(extreme))

    # 8개 구간 중 어느 구간에 문제가 있는지 확인
    segments = bad.reshape(6, 8, 8192)
    bad_segments = np.flatnonzero(segments.any(axis=(0, 2)))

    print("문제가 있는 구간 번호:", (bad_segments + 1).tolist())

print(f"\n문제 후보 파일: {problem_files}개")
