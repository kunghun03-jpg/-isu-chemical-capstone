# 보관된 초기 실험 코드: evaluate_file_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
import torch

from train_autoencoder_1200rpm import Autoencoder

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

torch.set_num_threads(min(4, torch.get_num_threads()))

checkpoint = torch.load(
    output_dir / "autoencoder_1200rpm.pt", map_location="cpu", weights_only=True
)

model = Autoencoder()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def get_file_scores(split):
    with np.load(output_dir / f"{split}_normalized.npz") as data:
        X = data["X"].copy()
        files = data["source_file"].copy()
        conditions = data["condition"].copy()
        segment_ids = data["segment_id"].copy()

    scores = []

    with torch.no_grad():
        for start in range(0, len(X), 32):
            batch = torch.from_numpy(X[start : start + 32]).float()

            reconstructed = model(batch)
            error = (reconstructed.double() - batch.double()).square().mean(dim=(1, 2))

            scores.extend(error.numpy().tolist())

    if not np.isfinite(scores).all():
        raise ValueError(f"{split}: 점수 확인 필요")

    grouped = defaultdict(list)

    for file, condition, segment_id, score in zip(
        files, conditions, segment_ids, scores
    ):
        grouped[str(file)].append((str(condition), int(segment_id), float(score)))

    result = []

    for file, entries in sorted(grouped.items()):
        if (
            len(entries) != 8
            or {entry[1] for entry in entries} != set(range(8))
            or len({entry[0] for entry in entries}) != 1
        ):
            raise ValueError(f"파일별 구간 구성 확인 필요: {file}")

        result.append(
            {
                "file": file,
                "condition": entries[0][0],
                "split": split,
                "score": float(np.mean([entry[2] for entry in entries])),
            }
        )

    return result


# 검증 파일도 동일하게 구간 점수의 평균 사용
validation = get_file_scores("validation")
threshold = float(np.percentile([row["score"] for row in validation], 95))

print("정상 검증 파일:", len(validation), "개")
print(f"파일 단위 임계값: {threshold:.6f}")

results = []

for split in ("test_normal", "test_anomaly"):
    file_results = get_file_scores(split)

    for row in file_results:
        row["threshold"] = threshold
        row["predicted_anomaly"] = int(row["score"] > threshold)

    predictions = np.array([row["predicted_anomaly"] for row in file_results])

    print(f"\n[{split}] 파일 {len(file_results)}개")

    if split == "test_normal":
        print(f"정상 파일 오탐률: {predictions.mean():.2%}")
    else:
        print(f"이상 파일 탐지율: {predictions.mean():.2%}")

    for condition in sorted({row["condition"] for row in file_results}):
        selected = [row for row in file_results if row["condition"] == condition]

        rate = np.mean([row["predicted_anomaly"] for row in selected])

        print(f"  {condition}: {len(selected)}개 파일, " f"이상 판정 비율 {rate:.2%}")

    results.extend(file_results)

with (output_dir / "evaluation_file_1200rpm.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n파일 단위 평가 저장 완료!")
