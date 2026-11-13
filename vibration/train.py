# 정상 데이터로 학습하고 검증 오차가 가장 낮은 모델을 저장합니다.

from pathlib import Path
import csv
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from vibration.model import Autoencoder
from vibration.experiment import load_experiment, experiment_metadata


def run(project_dir, output_dir, rpm):
    # 모델 수식·학습 설정은 유지하고, 실행 전 확정 Run 및 데이터 일치만 검사합니다.
    datasets, data_fingerprint = load_experiment(output_dir, rpm)
    del datasets

    # 1. 재현성을 위해 난수 시드를 고정하고 CPU 스레드 수를 제한합니다.
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    # 2. 배열의 모양과 유한값을 검사한 뒤 텐서로 바꿉니다.
    def load_tensor(split):
        path = output_dir / f"{split}_normalized.npz"
        with np.load(path) as data:
            X = data["X"].copy()
        if X.ndim != 3 or X.shape[1:] != (1, 3201):
            raise ValueError(f"{split}: 데이터 모양 확인 필요")
        if not np.isfinite(X).all():
            raise ValueError(f"{split}: NaN 또는 무한대 발견")
        return torch.from_numpy(X).float()

    train_X = load_tensor("train")
    val_X = load_tensor("validation")
    train_loader = DataLoader(
        TensorDataset(train_X), batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_X), batch_size=32, shuffle=False, num_workers=0
    )

    # 3. 모델, 평균 제곱 오차 손실과 Adam 최적화 설정을 준비합니다.
    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 30
    patience = 8
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    model_path = output_dir / f"autoencoder_{rpm}rpm.pt"
    print("학습 데이터:", tuple(train_X.shape))
    print("검증 데이터:", tuple(val_X.shape))
    print("CPU 학습 시작", flush=True)

    # 4. 학습을 반복하고 매 회차 별도 검증 데이터로 성능을 확인합니다.
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            if not torch.isfinite(loss).item():
                raise ValueError("학습 오차에 NaN 또는 무한대 발생")
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.size(0)
        train_loss = train_total / len(train_X)

        # 검증에서는 가중치를 갱신하지 않습니다.
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                reconstructed = model(batch)
                loss = loss_fn(reconstructed, batch)
                val_total += loss.item() * batch.size(0)
        val_loss = val_total / len(val_X)
        if not math.isfinite(val_loss):
            raise ValueError("검증 오차에 NaN 또는 무한대 발생")
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
        )
        improved = val_loss < best_val

        # 검증 오차가 개선될 때만 최적 모델을 저장합니다.
        if improved:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "input_shape": [1, 3201],
                    "channels": [3],
                    "data_fingerprint": data_fingerprint,
                    **experiment_metadata(output_dir, rpm),
                },
                model_path,
            )
        else:
            wait += 1
        mark = " ← 최적 모델 저장" if improved else ""
        print(
            f"Epoch {epoch:02d}/{max_epochs} | 학습: {train_loss:.6f} | 검증: {val_loss:.6f}{mark}",
            flush=True,
        )

        # 연속으로 개선되지 않으면 조기 종료합니다.
        if wait >= patience:
            print("검증 오차 개선이 없어 조기 종료")
            break
    with (output_dir / f"training_history_{rpm}rpm.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "validation_loss"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"\n학습 완료! 최적 Epoch: {best_epoch}")
    print(f"최저 검증 오차: {best_val:.6f}")
    print("모델 저장:", model_path)
