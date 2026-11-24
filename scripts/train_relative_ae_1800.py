import csv
import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vibration.model import Autoencoder
from vibration.relative_preprocessing import ch3_relative_spectra

ROOT = Path.cwd()
OUT = ROOT / "outputs/experimental_relative_ae/1800rpm"
OUT.mkdir(parents=True, exist_ok=True)
SPLIT = Path(
    "/home/kungh/ae_normal1_normal2_dev_1800/"
"split.csv"
)

MODEL = OUT / "model.pt"
STATS = OUT / "normalization.npz"
HISTORY = OUT / "history.csv"
SCORES = OUT / "validation_scores.csv"

for path in (MODEL, STATS, HISTORY, SCORES):
    if path.exists():
        raise FileExistsError(f"기존 결과 보호: {path}")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)

with SPLIT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 100:
    raise ValueError("분할 파일 수 불일치")

def prepare(split):
    arrays = []
    files = []
    conditions = []

    for row in rows:
        if row["split"] != split:
            continue

        if (
            row["date"] != "20260912"
            or row["condition"] not in ("Normal1", "Normal2")
            or int(row["rpm"]) != 1800
        ):
            raise ValueError("측정 조건 불일치")

        mat = loadmat(ROOT / row["file"])
        x = np.asarray(
            ch3_relative_spectra(mat["signals"], mat["fs"]),
            dtype=np.float32
        )

        if x.shape != (8, 1, 3201):
            raise ValueError(f"입력 크기 오류: {row['file']}")

        if not np.isfinite(x).all():
            raise ValueError("비유한 스펙트럼")

        arrays.append(x)
        files.append(row["file"])
        conditions.append(row["condition"])

    return np.concatenate(arrays), files, conditions

train_raw, train_files, train_conditions = prepare("train")
val_raw, val_files, val_conditions = prepare("validation")

counts = {'Normal1': 50, 'Normal2': 50}

if len(train_files) != 70 or len(val_files) != 30:
    raise ValueError("Train/Validation 개수 불일치")

if (
    len(set(train_files)) != 70
    or len(set(val_files)) != 30
    or set(train_files) & set(val_files)
):
    raise ValueError("파일 중복 발견")

for condition in ("Normal1", "Normal2"):
    if train_conditions.count(condition) != 35:
        raise ValueError("Train 조건별 파일 수 불일치")
    if val_conditions.count(condition) != (counts[condition] - 35):
        raise ValueError("Validation 조건별 파일 수 불일치")

# 통합 Normal Train만으로 정규화
mean = train_raw.mean(axis=0, keepdims=True)
std = np.maximum(
    train_raw.std(axis=0, keepdims=True), 1e-6
)

train_x = ((train_raw - mean) / std).astype(np.float32)
val_x = ((val_raw - mean) / std).astype(np.float32)

if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
    raise ValueError("정규화 오류")

train_tensor = torch.from_numpy(train_x)
val_tensor = torch.from_numpy(val_x)

loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=32,
    shuffle=True
)

model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

best_loss = float("inf")
best_epoch = 0
best_state = None
history = []
stale = 0

print("===== 1800 RPM Normal1·Normal2 통합 AE =====")
print("Train:", len(train_files), "파일 /", len(train_x), "구간")
print("Validation:", len(val_files), "파일 /", len(val_x), "구간")

for epoch in range(1, 61):
    model.train()
    total_loss = 0.0

    for (batch,) in loader:
        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)

    train_loss = total_loss / len(train_x)

    model.eval()
    with torch.inference_mode():
        val_loss = criterion(model(val_tensor), val_tensor).item()

    if not np.isfinite([train_loss, val_loss]).all():
        raise ValueError("학습 손실값 오류")

    improved = val_loss < best_loss

    if improved:
        best_loss = val_loss
        best_epoch = epoch
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
        stale = 0
    else:
        stale += 1

    history.append((epoch, train_loss, val_loss))

    print(
        f"Epoch {epoch:02d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f}"
        + (" *" if improved else "")
    )

    if stale >= 8:
        print("Early stopping")
        break

model.load_state_dict(best_state)
model.eval()

with torch.inference_mode():
    segment_scores = (
        (model(val_tensor).double() - val_tensor.double())
        .square()
        .mean(dim=(1, 2))
        .numpy()
    )

file_scores = segment_scores.reshape(len(val_files), 8).mean(axis=1)
threshold = float(np.percentile(file_scores, 95))

torch.save({
    "model_state_dict": best_state,
    "epoch": best_epoch,
    "validation_loss": best_loss,
    "experiment": "1800rpm_relative_l2_experimental",
}, MODEL)

np.savez_compressed(STATS, mean=mean, std=std)

with HISTORY.open("x", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    writer.writerows(history)

with SCORES.open("x", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "file", "condition", "score",
        "threshold", "predicted_anomaly"
    ])
    for name, condition, score in zip(
        val_files, val_conditions, file_scores
    ):
        writer.writerow([
            name, condition, float(score),
            threshold, int(score > threshold)
        ])

print("\n===== 최종 결과 =====")
print("최적 Epoch:", best_epoch)
print(f"Validation MSE: {best_loss:.6f}")
print(f"Validation Q95: {threshold:.6f}")

for condition in ("Normal1", "Normal2"):
    selected = np.array([
        score for score, c in zip(file_scores, val_conditions)
        if c == condition
    ])
    above = int((selected > threshold).sum())
    print(
        f"{condition}: 중앙값={np.median(selected):.6f}, "
        f"임계값 초과={above}/{len(selected)}"
    )

print("모델 저장:", MODEL)
print("기존 운영 모델 변경 없음")
print("9/19 Test 사용 없음")
print("물리적 Run 독립성 미확인")
