"""저장된 AE 파일 평가 결과로 정상·결함 점수 분포를 시각화한다."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/ae_test_score_boxplots"

CONDITIONS = [
    "Normal1",
    "Normal2",
    "IR",
    "OR",
    "Roller",
    "Looseness0.05",
    "Looseness0.1",
    "LoosenessFull",
    "Unbalance_-1",
    "Unbalance_-2",
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for rpm in (900, 1200, 1800):
        path = (
            ROOT / f"outputs/{rpm}rpm/date_prelim_20261106"
            / f"evaluation_file_{rpm}rpm.csv"
        )

        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [
                r for r in csv.DictReader(f)
                if r["split"] in ("test_normal", "test_anomaly")
            ]

        thresholds = {float(r["threshold"]) for r in rows}
        if len(thresholds) != 1:
            raise ValueError(f"{rpm}RPM: 임계값 불일치")

        threshold = thresholds.pop()

        values = []
        labels = []

        for condition in CONDITIONS:
            scores = [
                float(r["score"])
                for r in rows
                if r["condition"] == condition
            ]
            if scores:
                values.append(scores)
                labels.append(condition)

        fig, ax = plt.subplots(
            figsize=(15, 7),
            constrained_layout=True,
        )

        ax.boxplot(
            values,
            tick_labels=labels,
            showfliers=True,
        )

        ax.axhline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Normal Validation Q95 = {threshold:.4f}",
        )

        ax.set_title(
            f"{rpm}RPM | AE Test File Anomaly Score Distribution"
        )
        ax.set_ylabel("File reconstruction MSE")
        ax.set_xlabel("Condition")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        target = OUTPUT / f"ae_test_score_boxplot_{rpm}rpm.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)

        print(f"{rpm}RPM 그래프 저장: {target}")

    print("\nAE 정상·결함 점수 분포 그래프 3개 생성 완료")


if __name__ == "__main__":
    main()
