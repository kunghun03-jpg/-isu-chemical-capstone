# 보관된 초기 실험 코드: prepare_split_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import Counter
import csv

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data" / "20260916_1200RPM"
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

# 검사에서 발견한 문제 후보 파일 목록
with (output_dir / "quality_check_1200rpm.csv").open(
    encoding="utf-8-sig", newline=""
) as f:
    excluded = {row["file"] for row in csv.DictReader(f)}

rows = []
excluded_count = 0

for folder in sorted(data_dir.iterdir()):
    if not folder.is_dir():
        continue

    # 시간 순서로 정렬하고 문제 후보 제외
    files = []

    for path in sorted(folder.rglob("*.mat")):
        relative = path.relative_to(project_dir).as_posix()

        if relative in excluded:
            excluded_count += 1
            continue

        files.append(path)

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

output_path = output_dir / "split_1200rpm.csv"

with output_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "condition", "split"])
    writer.writerows(rows)

for split, count in sorted(Counter(row[2] for row in rows).items()):
    print(f"{split}: {count}개")

print("제외 파일:", excluded_count, "개")
print("사용 파일:", len(rows), "개")
print("목록 저장:", output_path)
