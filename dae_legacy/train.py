import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

# 화면이 없는 WSL 환경에서도 그래프 저장 가능
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ConvAutoencoder

# 1. 기본 설정
SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NOISE_STD = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

# 2. 정규화 데이터 불러오기
data = np.load("normal_dataset_normalized.npz")

X_train = data["X_train"].astype(np.float32)
X_val = data["X_val"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)

print("\n===== 데이터 크기 =====")
print("학습:", X_train.shape)
print("검증:", X_val.shape)
print("시험:", X_test.shape)

# 데이터에 NaN, 무한대가 있는지 확인
for name, X in [
    ("학습", X_train),
    ("검증", X_val),
    ("시험", X_test)
]:
    if not np.isfinite(X).all():
        raise ValueError(
            f"{name} 데이터에 NaN 또는 무한대가 있습니다."
        )

# Conv1d 입력 형태: (샘플 수, 채널 수, 특성 수)
X_train = torch.from_numpy(X_train).unsqueeze(1)
X_val = torch.from_numpy(X_val).unsqueeze(1)
X_test = torch.from_numpy(X_test).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 3. 모델 생성
model = ConvAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\n===== 모델 학습 시작 =====")

# 4. 학습 및 검증
train_losses = []
val_losses = []

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # ---------- 학습 ----------
    model.train()

    train_loss_sum = 0.0
    train_count = 0

    for (clean_x,) in train_loader:
        clean_x = clean_x.to(device)

        # 학습 입력에만 노이즈 추가
        noisy_x = clean_x + (
            torch.randn_like(clean_x) * NOISE_STD
        )

        optimizer.zero_grad()

        reconstructed = model(noisy_x)

        # 원래 정상 신호를 복원하도록 학습
        loss = criterion(reconstructed, clean_x)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * clean_x.size(0)
        train_count += clean_x.size(0)

    train_loss = train_loss_sum / train_count
    train_losses.append(train_loss)

    # ---------- 검증 ----------
    model.eval()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for (clean_x,) in val_loader:
            clean_x = clean_x.to(device)

            # 검증할 때는 노이즈를 추가하지 않음
            reconstructed = model(clean_x)

            loss = criterion(reconstructed, clean_x)

            val_loss_sum += loss.item() * clean_x.size(0)
            val_count += clean_x.size(0)

    val_loss = val_loss_sum / val_count
    val_losses.append(val_loss)

    # 검증 손실이 가장 낮은 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "autoencoder_model.pth"
        )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

# 5. 학습 손실 그래프 저장
plt.figure(figsize=(9, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "training_loss.png",
    dpi=150
)
plt.close()

# 6. 가장 좋은 모델 불러오기
model.load_state_dict(
    torch.load(
        "autoencoder_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# 7. 정상 데이터 복원 결과 확인
with torch.no_grad():
    sample = X_val[:1].to(device)

    reconstructed = model(sample)

original = sample.squeeze().cpu().numpy()
restored = reconstructed.squeeze().cpu().numpy()

plt.figure(figsize=(11, 5))

plt.plot(
    original,
    label="Original Normal Spectrum",
    linewidth=1.2
)

plt.plot(
    restored,
    label="Reconstructed Spectrum",
    linewidth=1.2
)

plt.xlabel("Frequency Bin (0-799)")
plt.ylabel("Normalized Log Spectrum")
plt.title("Normal Spectrum Reconstruction")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reconstruction.png",
    dpi=150
)

plt.close()

# 8. 최종 결과 출력
print("\n===== 학습 완료 =====")

print("최종 학습 손실:", train_losses[-1])
print("최종 검증 손실:", val_losses[-1])
print("최저 검증 손실:", best_val_loss)

print("\n저장된 파일:")
print("- autoencoder_model.pth")
print("- training_loss.png")
print("- reconstruction.png")