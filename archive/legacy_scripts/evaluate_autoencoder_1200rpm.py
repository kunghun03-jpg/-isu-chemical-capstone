# 보관된 초기 실험 코드: evaluate_autoencoder_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import json
import numpy as np
import torch

from train_autoencoder_1200rpm import Autoencoder


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))

    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

    # 저장된 최적 모델 불러오기
    checkpoint = torch.load(
        output_dir / "autoencoder_1200rpm.pt", map_location="cpu", weights_only=True
    )

    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def calculate_scores(split):
        with np.load(output_dir / f"{split}_normalized.npz") as data:
            X = data["X"].copy()
            files = data["source_file"].copy()
            conditions = data["condition"].copy()
            segments = data["segment_id"].copy()

        scores = []

        with torch.no_grad():
            for start in range(0, len(X), 32):
                batch = torch.from_numpy(X[start : start + 32]).float()

                reconstructed = model(batch)

                # 각 구간의 6채널·전체 주파수 평균 제곱 오차
                error = (
                    (reconstructed.double() - batch.double()).square().mean(dim=(1, 2))
                )

                scores.append(error.numpy())

        scores = np.concatenate(scores)

        if not np.isfinite(scores).all():
            raise ValueError(f"{split}: 점수 계산 결과 확인 필요")

        return scores, files, conditions, segments

    # 정상 검증 점수로만 임계값 설정
    val_scores, _, _, _ = calculate_scores("validation")
    threshold = float(np.percentile(val_scores, 95))

    print(f"불러온 모델 Epoch: {checkpoint['epoch']}")
    print(f"임시 임계값: {threshold:.6f}")

    results = []

    for split in ("test_normal", "test_anomaly"):
        scores, files, conditions, segments = calculate_scores(split)
        predictions = scores > threshold

        print(f"\n[{split}]")

        if split == "test_normal":
            false_alarm = float(predictions.mean())
            print(f"정상 구간 오탐률: {false_alarm:.2%}")
            print(f"정상 구간 정상 판정률: {1 - false_alarm:.2%}")
        else:
            print(f"이상 구간 탐지율: {predictions.mean():.2%}")

        # 상태별로 결과 확인
        for condition in sorted(np.unique(conditions)):
            mask = conditions == condition
            rate = predictions[mask].mean()

            print(
                f"  {condition}: {mask.sum()}개 구간, "
                f"이상 판정 비율 {rate:.2%}, "
                f"점수 중앙값 {np.median(scores[mask]):.6f}"
            )

        for i in range(len(scores)):
            results.append(
                {
                    "split": split,
                    "file": str(files[i]),
                    "condition": str(conditions[i]),
                    "segment_id": int(segments[i]),
                    "score": float(scores[i]),
                    "threshold": threshold,
                    "predicted_anomaly": int(predictions[i]),
                }
            )

    # 구간별 평가 결과 저장
    with (output_dir / "evaluation_1200rpm.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with (output_dir / "threshold_1200rpm.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "validation_segment_score_percentile",
                "percentile": 95,
                "threshold": threshold,
                "model_epoch": checkpoint["epoch"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n평가 결과 저장 완료!")


if __name__ == "__main__":
    main()
