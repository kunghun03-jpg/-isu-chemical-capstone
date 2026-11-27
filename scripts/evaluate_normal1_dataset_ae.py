"""데이터셋별 Normal1 기반 탐색적 이상 탐지 실험."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    f1_score,
)

from vibration.model import Autoencoder
from vibration.preprocessing import (
    apply_normalization,
    frequency_bins,
)
from vibration.dataset import path_metadata


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_spectra(path, rpm, normal):

    with np.load(path, allow_pickle=False) as data:

        X = data["X"].copy()

        meta = pd.DataFrame({
            "file": data["source_file"].astype(str),
            "condition": data["condition"].astype(str),
            "segment_id": data["segment_id"],
            "date": data["date"].astype(str),
            "rpm": data["rpm"].astype(int),
        })

        require(
            np.array_equal(
                data["frequency"], frequency_bins()
            ),
            "주파수 구성 불일치",
        )

    require(
        X.ndim == 3 and X.shape[1:] == (1, 3201),
        "스펙트럼 입력 형태 오류",
    )

    require(
        len(X) == len(meta)
        and np.isfinite(X).all()
        and (X >= 0).all(),
        "스펙트럼 데이터 오류",
    )

    require(
        meta["rpm"].eq(rpm).all()
        and meta["date"].eq("20260919").all(),
        "데이터셋 정보 불일치",
    )

    for file, group in meta.groupby("file"):

        require(
            len(group) == 8
            and set(group["segment_id"]) == set(range(8))
            and group["condition"].nunique() == 1,
            f"파일별 구간 오류: {file}",
        )

        info = path_metadata(file)

        require(
            info["date"] == "20260919"
            and int(info["rpm"]) == rpm
            and info["condition"]
                == group["condition"].iloc[0],
            f"파일 메타데이터 불일치: {file}",
        )

    allowed_faults = {
        "IR", "OR", "Roller",
        "Looseness0.05",
        "Looseness0.1",
        "LoosenessFull",
        "Unbalance_-1",
        "Unbalance_-2",
    }

    allowed = (
        {"Normal1", "Normal2"}
        if normal else allowed_faults
    )

    require(
        meta["condition"].isin(allowed).all(),
        "정상·결함 조건 오류",
    )

    return X, meta


def aggregate_scores(scores, meta):

    require(
        len(scores) == len(meta)
        and np.isfinite(scores).all(),
        "구간 점수 오류",
    )

    frame = meta.copy()
    frame["score"] = scores

    return (
        frame.groupby(
            ["file", "condition", "rpm"],
            as_index=False,
        )["score"].mean()
    )


def ae_scores(model, X, meta):

    scores = []

    model.eval()

    with torch.no_grad():

        for start in range(0, len(X), 32):

            batch = torch.from_numpy(
                X[start:start + 32]
            ).float()

            reconstructed = model(batch)

            error = (
                reconstructed.double()
                - batch.double()
            ).square().mean(dim=(1, 2))

            scores.extend(error.numpy().tolist())

    return aggregate_scores(
        np.asarray(scores), meta
    )


def distance_scores(X, meta, reference):

    scores = np.mean(
        (X.astype(np.float64) - reference) ** 2,
        axis=(1, 2),
    )

    return aggregate_scores(scores, meta)


def train_ae(train_X, val_X):

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X)),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X)),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = Autoencoder()

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, 31):

        model.train()
        total = 0.0

        for (batch,) in train_loader:

            optimizer.zero_grad()

            loss = loss_fn(model(batch), batch)

            require(
                torch.isfinite(loss).item(),
                "학습 오차 오류",
            )

            loss.backward()
            optimizer.step()

            total += loss.item() * len(batch)

        train_loss = total / len(train_X)

        model.eval()
        total = 0.0

        with torch.no_grad():

            for (batch,) in val_loader:

                loss = loss_fn(
                    model(batch), batch
                )

                total += loss.item() * len(batch)

        val_loss = total / len(val_X)

        require(
            np.isfinite(val_loss),
            "검증 오차 오류",
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        if val_loss < best_loss:

            best_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(
                model.state_dict()
            )

            wait = 0

        else:
            wait += 1

        print(
            f"Epoch {epoch:02d}/30 | "
            f"Train={train_loss:.6f} | "
            f"Val={val_loss:.6f}",
            flush=True,
        )

        if wait >= 8:
            print("Early stopping")
            break

    require(
        best_state is not None,
        "최적 모델 저장 실패",
    )

    model.load_state_dict(best_state)
    model.eval()

    return model, history, best_epoch, best_loss


def metrics(frame, method):

    y = (
        ~frame["condition"].isin(
            ["Normal1", "Normal2"]
        )
    ).astype(int)

    pred = frame["predicted_anomaly"]

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)

    return {
        "method": method,
        "files": len(frame),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "FPR": fpr,
        "Recall": recall,
        "Specificity": 1 - fpr,
        "Balanced_Accuracy":
            (recall + 1 - fpr) / 2,
        "Precision": precision_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "AUROC": roc_auc_score(
            y, frame["score"]
        ),
        "AUPRC_AP": average_precision_score(
            y, frame["score"]
        ),
        "evaluation_status": "exploratory_only",
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rpm",
        type=int,
        required=True,
        choices=[900, 1200, 1800],
    )

    args = parser.parse_args()
    rpm = args.rpm

    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    root = Path("outputs")

    source = (
        root / f"{rpm}rpm"
        / "date_prelim_20261106"
    )

    normal_path = (
        source / "test_normal_spectra.npz"
    )

    anomaly_path = (
        source / "test_anomaly_spectra.npz"
    )

    old_score_path = (
        root / "date_prelim_20261106_score_analysis"
        / f"{rpm}_autoencoder_file_scores.csv"
    )

    out = (
        root / "experimental_normal1_dataset_ae"
        / "20260919" / f"{rpm}rpm"
    )

    require(
        not out.exists(),
        f"기존 실험 결과 존재: {out}",
    )

    source_paths = [
        normal_path,
        anomaly_path,
        old_score_path,
    ]

    source_hashes = {
        str(p): sha256(p)
        for p in source_paths
    }

    print("===== Load existing spectra =====")

    X_normal, normal = load_spectra(
        normal_path, rpm, True
    )

    X_anomaly, anomaly = load_spectra(
        anomaly_path, rpm, False
    )

    require(
        not set(normal["file"])
        & set(anomaly["file"]),
        "정상·결함 파일 중복",
    )

    normal1_files = sorted(
        normal.loc[
            normal["condition"] == "Normal1",
            "file",
        ].unique(),
        key=lambda p: Path(p).name,
    )

    require(
        len(normal1_files) > 40,
        "Normal1 파일 부족",
    )

    require(
        len({Path(p).name for p in normal1_files})
        == len(normal1_files),
        "Normal1 파일명 중복",
    )

    train_files = set(normal1_files[:20])
    val_files = set(normal1_files[20:40])
    test_normal1 = set(normal1_files[40:])

    require(
        not train_files & val_files
        and not train_files & test_normal1
        and not val_files & test_normal1,
        "파일 분할 중복",
    )

    train_mask = normal["file"].isin(
        train_files
    ).to_numpy()

    val_mask = normal["file"].isin(
        val_files
    ).to_numpy()

    test_mask = ~(train_mask | val_mask)

    require(
        train_mask.sum() == 160
        and val_mask.sum() == 160,
        "학습·검증 구간 수 오류",
    )

    train_raw = X_normal[train_mask]
    val_raw = X_normal[val_mask]

    train_meta = normal.loc[
        train_mask
    ].reset_index(drop=True)

    val_meta = normal.loc[
        val_mask
    ].reset_index(drop=True)

    test_meta = pd.concat(
        [normal.loc[test_mask], anomaly],
        ignore_index=True,
    )

    test_raw = np.concatenate(
        [X_normal[test_mask], X_anomaly],
        axis=0,
    )

    test_files = set(test_meta["file"])

    require(
        not test_files & (train_files | val_files),
        "학습·검증·시험 파일 중복",
    )

    print("\n===== Split =====")
    print("Train Normal1:", len(train_files))
    print("Validation Normal1:", len(val_files))
    print("Test Normal1:", len(test_normal1))
    print("Test files:", len(test_files))

    print("\n===== Train-only normalization =====")

    train64 = train_raw.astype(np.float64)

    mean = train64.mean(axis=0, keepdims=True)

    std = np.maximum(
        train64.std(axis=0, keepdims=True),
        1e-6,
    )

    train_X = apply_normalization(
        train_raw, mean, std
    )

    val_X = apply_normalization(
        val_raw, mean, std
    )

    test_X = apply_normalization(
        test_raw, mean, std
    )

    # 기존 점수와 시험 파일의 일치 여부 확인
    old = pd.read_csv(
        old_score_path,
        dtype={"date": str},
    )

    old = old[
        old["split"].isin(
            ["test_normal", "test_anomaly"]
        )
    ].copy()

    require(
        old["file"].is_unique
        and set(old["file"])
            == set(normal["file"]) | set(anomaly["file"]),
        "기존 평가 파일 불일치",
    )

    old = old.set_index("file")

    expected = pd.concat(
        [normal, anomaly],
        ignore_index=True,
    ).drop_duplicates("file").set_index("file")

    aligned = old.loc[expected.index]

    require(
        np.array_equal(
            aligned["condition"], expected["condition"]
        )
        and aligned["rpm"].eq(rpm).all()
        and aligned["date"].eq("20260919").all()
        and np.isfinite(aligned["score"]).all()
        and np.isfinite(aligned["threshold"]).all()
        and aligned["threshold"].nunique() == 1,
        "기존 점수 메타데이터 오류",
    )

    require(
        np.array_equal(
            aligned["predicted_anomaly"],
            (aligned["score"] >
             aligned["threshold"]).astype(int),
        ),
        "기존 판정 불일치",
    )

    old_threshold = float(
        old["threshold"].iloc[0]
    )

    original = old.loc[
        sorted(test_files)
    ].reset_index()

    # C와 동일한 Validation 파일로 B 보정
    calibrated_threshold = float(
        np.percentile(
            old.loc[sorted(val_files), "score"],
            95,
        )
    )

    # 이전 STEP 11-3 보정 방식도 별도 유지
    previous_threshold = float(
        np.percentile(
            old.loc[sorted(train_files), "score"],
            95,
        )
    )

    print("\n===== Normal1-only AE training =====")

    model, history, epoch, loss = train_ae(
        train_X, val_X
    )

    # Validation 결과만 이용해 기준 결정
    val_ae = ae_scores(
        model, val_X, val_meta
    )

    ae_threshold = float(
        np.percentile(val_ae["score"], 95)
    )

    reference = train_X.mean(
        axis=0,
        dtype=np.float64,
    )

    val_distance = distance_scores(
        val_X, val_meta, reference
    )

    distance_threshold = float(
        np.percentile(
            val_distance["score"], 95
        )
    )

    print("\n===== Frozen thresholds =====")
    print("Original AE:", old_threshold)
    print("Calibrated AE:", calibrated_threshold)
    print("Normal1 AE:", ae_threshold)
    print("Normal distance:", distance_threshold)

    # 임계값 확정 이후 시험 점수 계산
    test_ae = ae_scores(
        model, test_X, test_meta
    )

    test_distance = distance_scores(
        test_X, test_meta, reference
    )

    experiments = {
        "A_original": (
            original, old_threshold
        ),
        "B_normal1_q95": (
            original, calibrated_threshold
        ),
        "B_previous_first20": (
            original, previous_threshold
        ),
        "C_normal1_ae": (
            test_ae, ae_threshold
        ),
        "D_normal_distance": (
            test_distance, distance_threshold
        ),
    }

    results = []
    groups = []
    predictions = []

    for method, (frame, threshold) in (
        experiments.items()
    ):

        frame = frame.copy()

        require(
            frame["file"].is_unique
            and set(frame["file"]) == test_files,
            f"시험 파일 불일치: {method}",
        )

        frame["threshold"] = threshold

        frame["predicted_anomaly"] = (
            frame["score"] > threshold
        ).astype(int)

        frame["method"] = method
        frame["evaluation_status"] = "exploratory_only"

        results.append(metrics(frame, method))
        predictions.append(frame)

        for condition, group in frame.groupby(
            "condition"
        ):

            groups.append({
                "method": method,
                "condition": condition,
                "files": len(group),
                "metric": (
                    "FPR" if condition in
                    ("Normal1", "Normal2")
                    else "Recall"
                ),
                "rate_pct": (
                    group["predicted_anomaly"].mean()
                    * 100
                ),
            })

    result_df = pd.DataFrame(results)
    group_df = pd.DataFrame(groups)

    # 원본 산출물이 변경되지 않았는지 확인
    require(
        all(
            sha256(p) == source_hashes[str(p)]
            for p in source_paths
        ),
        "기존 입력 산출물 변경 감지",
    )

    out.mkdir(parents=True, exist_ok=False)

    split_rows = []

    for name, metadata in (
        ("train", train_meta),
        ("validation", val_meta),
        ("test", test_meta),
    ):

        rows = metadata[
            ["file", "condition", "rpm", "date"]
        ].drop_duplicates().copy()

        rows["split"] = name
        rows["run_independence_verified"] = False

        split_rows.append(rows)

    pd.concat(
        split_rows, ignore_index=True
    ).to_csv(
        out / "split.csv", index=False
    )

    np.savez_compressed(
        out / "normalization.npz",
        mean=mean,
        std=std,
    )

    np.savez_compressed(
        out / "normal_reference.npz",
        reference=reference,
    )

    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": epoch,
        "validation_loss": loss,
        "threshold": ae_threshold,
        "input_shape": [1, 3201],
        "evaluation_status": "exploratory_only",
        "approved_for_production": False,
    }, out / "autoencoder.pt")

    pd.DataFrame(history).to_csv(
        out / "training_history.csv",
        index=False,
    )

    validation_rows = []

    for name, frame, threshold in (
        ("C_normal1_ae", val_ae, ae_threshold),
        ("D_normal_distance",
         val_distance, distance_threshold),
    ):

        frame = frame.copy()
        frame["method"] = name
        frame["threshold"] = threshold
        validation_rows.append(frame)

    pd.concat(
        validation_rows,
        ignore_index=True,
    ).to_csv(
        out / "validation_scores.csv",
        index=False,
    )

    pd.concat(
        predictions,
        ignore_index=True,
    ).to_csv(
        out / "test_scores.csv",
        index=False,
    )

    result_df.to_csv(
        out / "metrics_comparison.csv",
        index=False,
    )

    group_df.to_csv(
        out / "condition_metrics.csv",
        index=False,
    )

    metadata = {
        "date": "20260919",
        "rpm": rpm,
        "seed": 42,
        "max_epochs": 30,
        "patience": 8,
        "batch_size": 32,
        "learning_rate": 0.001,
        "training_condition": "Normal1",
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "test_files": len(test_files),
        "normalization": "per_frequency",
        "std_floor": 1e-6,
        "threshold_method": "Q95",
        "thresholds": {
            name: threshold
            for name, (_, threshold)
            in experiments.items()
        },
        "source_sha256": source_hashes,
        "evaluation_status": "exploratory_only",
        "test_previously_examined": True,
        "run_independence_verified": False,
        "approved_for_production": False,
    }

    (out / "experiment_metadata.json").write_text(
        json.dumps(
            metadata, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== Overall comparison =====")

    print(
        result_df.round(6)
        .to_string(index=False)
    )

    print("\n===== Condition comparison (%) =====")

    print(
        group_df.pivot(
            index=["condition", "files", "metric"],
            columns="method",
            values="rate_pct",
        ).round(2).to_string()
    )

    print("\n===== Final verification =====")
    print("Best epoch:", epoch)
    print("Best validation loss:", loss)
    print("Train / validation / test overlap: 0")
    print("Common evaluation files:", len(test_files))
    print("Training condition: Normal1 only")
    print("Original source hashes: PASS")
    print("Production model changes: NONE")
    print("Physical Run independence: UNVERIFIED")
    print("Evaluation scope: exploratory_only")
    print("Output:", out)
    print("Normal1 dataset experiment: PASS")


if __name__ == "__main__":
    main()
