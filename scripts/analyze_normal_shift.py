from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibration.model import Autoencoder
from vibration.experiment import load_experiment, check_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ae_scores(folder, rpm, split, model):
    path = folder / f"{split}_normalized.npz"
    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        files = data["source_file"].astype(str)
        conditions = data["condition"].astype(str)

    if X.ndim != 3 or X.shape[1:] != (1, 3201):
        raise ValueError(f"예상하지 못한 AE 입력 형상: {X.shape}")

    grouped = {}
    with torch.no_grad():
        for start in range(0, len(X), 32):
            batch = torch.from_numpy(X[start:start + 32])
            output = model(batch)
            scores = (
                (output.double() - batch.double())
                .square()
                .mean(dim=(1, 2))
                .numpy()
            )

            for offset, score in enumerate(scores):
                i = start + offset
                key = str(files[i])
                grouped.setdefault(key, []).append(
                    (str(conditions[i]), float(score))
                )

    rows = []
    for file, entries in sorted(grouped.items()):
        if len(entries) != 8 or len({c for c, _ in entries}) != 1:
            raise ValueError(f"파일 구간 구성 오류: {file}")
        rows.append({
            "file": file,
            "condition": entries[0][0],
            "split": split,
            "score": float(np.mean([score for _, score in entries])),
        })
    return rows


def summarize(model_name, rpm, rows, threshold):
    result = []
    for split in ("train", "validation", "test_normal"):
        selected = [
            float(r["score"])
            for r in rows
            if r["split"] == split
        ]
        if not selected:
            raise ValueError(f"{model_name} {rpm}RPM: {split} 점수 없음")

        values = np.asarray(selected)
        if not np.isfinite(values).all():
            raise ValueError("비유한 점수 발견")

        result.append({
            "rpm": rpm,
            "model": model_name,
            "split": split,
            "files": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "q95": float(np.percentile(values, 95)),
            "threshold": threshold,
            "threshold_exceedance_rate": float(np.mean(values > threshold)),
        })
    return result


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))

    for rpm in (900, 1200, 1800):
        folder = ROOT / f"outputs/{rpm}rpm/date_prelim_20261106"
        if not folder.is_dir():
            print(f"{rpm}RPM: 실험 폴더 없음, 건너뜀")
            continue

        print(f"\n========== {rpm} RPM ==========")

        _, fingerprint = load_experiment(folder, rpm)
        checkpoint = torch.load(
            folder / f"autoencoder_{rpm}rpm.pt",
            map_location="cpu",
            weights_only=True,
        )
        check_fingerprint(checkpoint, fingerprint)

        model = Autoencoder()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        ae_test = read_csv(folder / f"evaluation_file_{rpm}rpm.csv")
        ae_rows = []
        for split in ("train", "validation"):
            ae_rows.extend(ae_scores(folder, rpm, split, model))
        ae_rows.extend(ae_test)

        ae_thresholds = {float(r["threshold"]) for r in ae_test}
        if len(ae_thresholds) != 1:
            raise ValueError("AE 파일 임계값 불일치")

        all_summary = []
        all_summary.extend(
            summarize("AE", rpm, ae_rows, ae_thresholds.pop())
        )

        for name, dirname in (("OCSVM", "ocsvm"), ("IF", "iforest")):
            rows = read_csv(
                folder / "comparisons" / dirname / "scores_file.csv"
            )
            thresholds = {float(r["threshold"]) for r in rows}
            if len(thresholds) != 1:
                raise ValueError(f"{name} 임계값 불일치")
            all_summary.extend(
                summarize(name, rpm, rows, thresholds.pop())
            )

        destination = folder / "normal_shift_analysis"
        destination.mkdir(exist_ok=True)
        path = destination / "normal_score_summary.csv"

        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_summary[0].keys())
            )
            writer.writeheader()
            writer.writerows(all_summary)

        for row in all_summary:
            print(
                f"{row['model']:5s} "
                f"{row['split']:12s} "
                f"N={row['files']:3d} "
                f"중앙값={row['median']:.6f} "
                f"Q95={row['q95']:.6f} "
                f"임계값 초과={row['threshold_exceedance_rate']:.2%}"
            )

        print("저장:", path)


if __name__ == "__main__":
    main()
