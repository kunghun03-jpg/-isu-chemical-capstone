# 보관된 초기 실험 코드: evaluate_combined_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

# 정상 데이터로 이미 정한 보조 점수 기준 사용
with (output_dir / "reference_20hz_1200rpm.json").open(encoding="utf-8") as f:
    reference = json.load(f)

mean = np.array(reference["mean"], dtype=np.float64)
std = np.array(reference["std"], dtype=np.float64)
aux_threshold = float(reference["threshold"])

channels = np.array(reference["channels"]) - 1
low, high = reference["band_hz"]

# 기존 오토엔코더 파일 단위 결과
with (output_dir / "evaluation_file_1200rpm.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    ae_rows = list(csv.DictReader(f))

# 분할 목록과 평가 대상 일치 여부 확인
with (project_dir / "outputs_clean" / "1200rpm" / "first" / "split_1200rpm.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    expected = {
        row["file"]: (row["condition"], row["split"])
        for row in csv.DictReader(f)
        if row["split"] in ("test_normal", "test_anomaly")
    }

actual_files = [row["file"] for row in ae_rows]

if len(actual_files) != len(set(actual_files)) or set(actual_files) != set(expected):
    raise ValueError("분할 목록과 오토엔코더 평가 파일이 다릅니다.")

fs = 25600.0
n = 65536
window = np.hanning(n)
freq = np.fft.rfftfreq(n, d=1 / fs)
band = (freq >= low) & (freq <= high)

results = []

for i, row in enumerate(ae_rows, start=1):
    if expected[row["file"]] != (row["condition"], row["split"]):
        raise ValueError(f"파일 정보 불일치: {row['file']}")

    data = loadmat(project_dir / row["file"])
    signals = data["signals"].copy()

    if signals.shape != (6, n) or float(data["fs"].item()) != fs:
        raise ValueError(f"데이터 형식 확인 필요: {row['file']}")

    if not np.isfinite(signals).all() or (np.abs(signals) > 1e100).any():
        raise ValueError(f"데이터 값 확인 필요: {row['file']}")

    signals -= signals.mean(axis=1, keepdims=True)

    amplitude = np.abs(np.fft.rfft(signals * window, axis=1)) / window.sum()
    amplitude[:, 1:-1] *= 2

    peaks = amplitude[channels][:, band].max(axis=1)
    aux_score = float(np.mean(((peaks - mean) / std) ** 2))

    if not np.isfinite(aux_score):
        raise ValueError(f"보조 점수 확인 필요: {row['file']}")

    ae_score = float(row["score"])
    ae_threshold = float(row["threshold"])
    ae_alarm = int(ae_score > ae_threshold)
    aux_alarm = int(aux_score > aux_threshold)
    combined_alarm = int(ae_alarm or aux_alarm)

    results.append(
        {
            "file": row["file"],
            "condition": row["condition"],
            "split": row["split"],
            "ae_score": ae_score,
            "ae_threshold": ae_threshold,
            "aux_score": aux_score,
            "aux_threshold": aux_threshold,
            "ae_alarm": ae_alarm,
            "aux_alarm": aux_alarm,
            "combined_alarm": combined_alarm,
        }
    )

    if i % 100 == 0 or i == len(ae_rows):
        print(f"계산 진행: {i}/{len(ae_rows)}", flush=True)

# 결과 저장
with (output_dir / "evaluation_combined_1200rpm.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# 같은 파일에서 세 방법 비교
methods = [
    ("ae_alarm", "오토엔코더"),
    ("aux_alarm", "20Hz 보조 점수"),
    ("combined_alarm", "결합"),
]

for split in ("test_normal", "test_anomaly"):
    selected = [r for r in results if r["split"] == split]
    metric = "오탐률" if split == "test_normal" else "탐지율"

    print(f"\n[{split}] {len(selected)}개 파일")

    for key, name in methods:
        count = sum(r[key] for r in selected)
        print(
            f"{name} {metric}: "
            f"{count}/{len(selected)}개 "
            f"({count / len(selected):.2%})"
        )

print("\n상태별 결합 결과")

for condition, count in sorted(Counter(r["condition"] for r in results).items()):
    selected = [r for r in results if r["condition"] == condition]
    alarms = sum(r["combined_alarm"] for r in selected)

    print(f"{condition}: 이상 판정 {alarms}/{count}개 " f"({alarms / count:.2%})")

print("\n결합 평가 결과 저장 완료!")
