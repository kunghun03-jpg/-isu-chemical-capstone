# 보관된 초기 실험 코드: prepare_split. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
from collections import Counter

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data" / "20260916_900RPM"

rows = []

for folder in sorted(data_dir.iterdir()):
    if not folder.is_dir():
        continue

    files = sorted(folder.rglob("*.mat"))

    if folder.name in ("Normal1", "Normal2"):
        n_train = int(len(files) * 0.6)
        n_val = int(len(files) * 0.2)

        for i, path in enumerate(files):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "validation"
            else:
                split = "test_normal"

            rows.append([path.relative_to(project_dir).as_posix(), folder.name, split])
    else:
        for path in files:
            rows.append(
                [path.relative_to(project_dir).as_posix(), folder.name, "test_anomaly"]
            )

output_dir = project_dir / "outputs_clean" / "900rpm" / "first"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "split_900rpm.csv"

with output_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "condition", "split"])
    writer.writerows(rows)

for split, count in sorted(Counter(row[2] for row in rows).items()):
    print(f"{split}: {count}개")

print("목록 저장 완료:", output_path)
