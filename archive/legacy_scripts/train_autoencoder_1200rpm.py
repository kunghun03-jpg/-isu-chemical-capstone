# 보관된 초기 실험 코드: train_autoencoder_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 주파수 방향으로 압축
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        )

        # 압축된 데이터에서 스펙트럼 복원
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 6, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        reconstructed = self.decoder(self.encoder(x))

        # 입력과 출력의 주파수 개수 맞추기
        return reconstructed[:, :, : x.shape[-1]]


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 노트북 CPU에서 사용할 스레드 수
    torch.set_num_threads(min(4, torch.get_num_threads()))

    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"

    def load_tensor(split):
        path = output_dir / f"{split}_normalized.npz"

        with np.load(path) as data:
            X = data["X"].copy()

        if X.ndim != 3 or X.shape[1:] != (6, 3201):
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

    model = Autoencoder()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_epochs = 30
    patience = 8
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    history = []

    model_path = output_dir / "autoencoder_1200rpm.pt"

    print("학습 데이터:", tuple(train_X.shape))
    print("검증 데이터:", tuple(val_X.shape))
    print("CPU 학습 시작", flush=True)

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

        if improved:
            best_val = val_loss
            best_epoch = epoch
            wait = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "input_shape": [6, 3201],
                },
                model_path,
            )
        else:
            wait += 1

        mark = " ← 최적 모델 저장" if improved else ""
        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"학습: {train_loss:.6f} | "
            f"검증: {val_loss:.6f}{mark}",
            flush=True,
        )

        if wait >= patience:
            print("검증 오차 개선이 없어 조기 종료")
            break

    with (output_dir / "training_history_1200rpm.csv").open(
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


if __name__ == "__main__":
    main()
