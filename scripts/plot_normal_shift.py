from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

for rpm in (900, 1200, 1800):
    folder = ROOT / f"outputs/{rpm}rpm/date_prelim_20261106"
    path = folder / "normal_shift_analysis/normal_score_summary.csv"

    if not path.exists():
        print(f"{rpm}RPM: 분석 CSV 없음")
        continue

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    models = ("AE", "OCSVM", "IF")
    splits = ("train", "validation", "test_normal")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, model in zip(axes, models):
        selected = {
            row["split"]: row
            for row in rows
            if row["model"] == model
        }

        medians = [
            float(selected[s]["median"])
            for s in splits
        ]
        q25 = [
            float(selected[s]["q25"])
            for s in splits
        ]
        q75 = [
            float(selected[s]["q75"])
            for s in splits
        ]

        x = np.arange(3)
        ax.errorbar(
            x,
            medians,
            yerr=[
                np.array(medians) - np.array(q25),
                np.array(q75) - np.array(medians),
            ],
            fmt="o",
            capsize=6,
        )

        threshold = float(selected["validation"]["threshold"])
        ax.axhline(
            threshold,
            linestyle="--",
            label="Validation Q95",
        )

        ax.set_xticks(x, ["Train", "Validation", "Test Normal"])
        ax.set_title(f"{model} - {rpm} RPM")
        ax.set_ylabel("Anomaly score")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle(
        f"Normal Score Distribution Shift - {rpm} RPM\n"
        "Median and IQR | Date-based preliminary experiment"
    )
    fig.tight_layout()

    output = folder / "normal_shift_analysis" / f"normal_shift_{rpm}rpm.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("저장:", output)
