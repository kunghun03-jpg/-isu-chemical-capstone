# 1800RPM 정상 파일 점수를 계산하고 시간 순서 그래프를 저장합니다.

from pathlib import Path
import csv
import subprocess
import sys
import argparse

import numpy as np
from vibration.channel import validate_spectra
from vibration.experiment import load_experiment, check_fingerprint


# 점수 계산: PyTorch를 이 함수에서 불러와 그래프 프로세스와 분리합니다.
def main(run_name="ch3_runs"):
    import torch
    from vibration.model import Autoencoder

    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "outputs" / "1800rpm" / run_name
    datasets, data_fingerprint = load_experiment(output_dir, 1800)
    del datasets
    analysis_dir = output_dir / "normal_analysis"
    analysis_dir.mkdir(exist_ok=True)

    torch.set_num_threads(min(4, torch.get_num_threads()))

    checkpoint = torch.load(
        output_dir / "autoencoder_1800rpm.pt",
        map_location="cpu",
        weights_only=True,
    )
    check_fingerprint(checkpoint, data_fingerprint)
    model = Autoencoder()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 기존 파일 단위 임계값 그대로 사용
    with (output_dir / "evaluation_file_1800rpm.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        threshold = float(next(csv.DictReader(f))["threshold"])

    results = []

    for split in ("train", "validation", "test_normal"):
        with np.load(output_dir / f"{split}_normalized.npz") as data:
            X = data["X"]
            validate_spectra(X)
            files = data["source_file"].astype(str)
            conditions = data["condition"].astype(str)
            segment_ids = data["segment_id"]

        scores = []

        with torch.no_grad():
            for start in range(0, len(X), 32):
                batch = torch.from_numpy(X[start : start + 32]).float()
                reconstructed = model(batch)

                errors = (
                    (reconstructed.double() - batch.double()).square().mean(dim=(1, 2))
                )
                scores.append(errors.numpy())

        scores = np.concatenate(scores)

        if not np.isfinite(scores).all():
            raise ValueError(f"{split}: 점수에 비정상 숫자가 있습니다.")

        for file in sorted(np.unique(files)):
            mask = files == file

            if (
                mask.sum() != 8
                or set(segment_ids[mask].tolist()) != set(range(8))
                or len(np.unique(conditions[mask])) != 1
            ):
                raise ValueError(f"구간 또는 상태 정보 확인 필요: {file}")

            score = float(scores[mask].mean())
            filename = Path(file).stem
            time_text = filename.split("_")[1]

            results.append(
                {
                    "file": file,
                    "condition": str(conditions[mask][0]),
                    "split": split,
                    "time": (
                        f"{time_text[:2]}:" f"{time_text[2:4]}:" f"{time_text[4:]}"
                    ),
                    "score": score,
                    "threshold": threshold,
                    "above_threshold": int(score > threshold),
                }
            )

        print(f"{split}: 정상 파일 {len(np.unique(files))}개 계산 완료")

    csv_path = analysis_dir / "normal_scores_all_splits_1800rpm.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # Keep PyTorch and plotting libraries in separate processes on Windows.
    subprocess.run(
        [sys.executable, "-m", "scripts.analyze_normal_1800rpm", "--plot-only", "--run-name", run_name],
        cwd=project_dir,
        check=True,
    )


# 그래프 생성: --plot-only에서는 저장된 CSV를 읽어 추론을 생략합니다.
def plot_results(run_name="ch3_runs"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "1800rpm"
        / run_name
        / "normal_analysis"
    )
    csv_path = analysis_dir / "normal_scores_all_splits_1800rpm.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        results = list(csv.DictReader(f))
    for row in results:
        row["score"] = float(row["score"])
        row["threshold"] = float(row["threshold"])
    threshold = results[0]["threshold"]

    colors = {
        "train": "tab:blue",
        "validation": "tab:orange",
        "test_normal": "tab:green",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    print(f"\n기존 파일 단위 임계값: {threshold:.6f}")

    for ax, condition in zip(axes, ("Normal1", "Normal2")):
        rows = sorted(
            [row for row in results if row["condition"] == condition],
            key=lambda row: row["file"],
        )

        for split, color in colors.items():
            indices = [i for i, row in enumerate(rows) if row["split"] == split]
            values = [rows[i]["score"] for i in indices]

            ax.scatter(indices, values, color=color, label=split)

            if values:
                above = sum(value > threshold for value in values)
                print(
                    f"{condition} / {split}: "
                    f"{len(values)}개, "
                    f"점수 중앙값 {np.median(values):.6f}, "
                    f"임계값 초과 {above}개"
                )

        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            label=f"Threshold = {threshold:.6f}",
        )

        ticks = list(range(0, len(rows), 5))
        if len(rows) - 1 not in ticks:
            ticks.append(len(rows) - 1)

        ax.set_xticks(ticks)
        ax.set_xticklabels([rows[i]["time"] for i in ticks], rotation=45)
        ax.set_title(f"1800RPM - {condition}")
        ax.set_xlabel("Measurement time (file order)")
        ax.set_ylabel("Mean reconstruction error")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    graph_path = analysis_dir / "normal_scores_over_time_1800rpm.png"
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    print("\n분석 완료!")
    print("CSV:", csv_path)
    print("그래프:", graph_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1800RPM CH3 정상 점수 분석")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--run-name", default="ch3_runs")
    args = parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("실험 이름에는 영문·숫자·밑줄·하이픈만 사용하세요.")
    if args.plot_only:
        plot_results(args.run_name)
    else:
        main(args.run_name)
