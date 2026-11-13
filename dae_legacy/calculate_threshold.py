import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ConvAutoencoder

# 1. 기본 설정
device = torch.device("cpu")

print("===== 정상 데이터 이상 점수 계산 =====")

# 2. 정규화 데이터 불러오기
data = np.load("normal_dataset_normalized.npz")

X_train = data["X_train"].astype(np.float32)
X_val = data["X_val"].astype(np.float32)
X_test = data["X_test"].astype(np.float32)

print("학습 데이터:", X_train.shape)
print("검증 데이터:", X_val.shape)
print("시험 데이터:", X_test.shape)

# 3. 학습된 모델 불러오기
model = ConvAutoencoder().to(device)

model.load_state_dict(
    torch.load(
        "autoencoder_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("학습된 모델 불러오기 완료!")

# 4. 샘플별 복원 오차(MSE) 계산
def calculate_mse(X):
    X_tensor = torch.from_numpy(X).unsqueeze(1).to(device)

    with torch.no_grad():
        reconstructed = model(X_tensor)

        # 각 샘플의 800개 주파수 빈에 대해 평균 제곱 오차
        mse = torch.mean(
            (X_tensor - reconstructed) ** 2,
            dim=(1, 2)
        )

    return mse.cpu().numpy()


train_mse = calculate_mse(X_train)
val_mse = calculate_mse(X_val)
test_mse = calculate_mse(X_test)

# 5. 정상 검증 데이터 기준 임계값 설정
# 정상 검증 데이터 MSE의 95번째 백분위수
threshold = np.percentile(val_mse, 95)

print("\n===== 복원 오차 통계 =====")

for name, scores in [
    ("학습", train_mse),
    ("검증", val_mse),
    ("시험", test_mse)
]:
    print(f"\n{name} 데이터")
    print(f"평균 MSE: {scores.mean():.6f}")
    print(f"최소 MSE: {scores.min():.6f}")
    print(f"최대 MSE: {scores.max():.6f}")

print("\n===== 이상 탐지 임계값 =====")
print(f"95% 기준 임계값: {threshold:.6f}")


# 6. 정상 데이터의 이상 판정 비율 확인
val_abnormal = val_mse > threshold
test_abnormal = test_mse > threshold

print("\n===== 정상 데이터 이상 판정 결과 =====")

print(
    "검증 데이터 이상 판정:",
    f"{val_abnormal.sum()} / {len(val_mse)}",
    f"({val_abnormal.mean() * 100:.2f}%)"
)

print(
    "시험 데이터 이상 판정:",
    f"{test_abnormal.sum()} / {len(test_mse)}",
    f"({test_abnormal.mean() * 100:.2f}%)"
)

# 7. MSE 분포 그래프 저장
plt.figure(figsize=(10, 6))

plt.hist(
    train_mse,
    bins=30,
    alpha=0.5,
    label="Train Normal"
)

plt.hist(
    val_mse,
    bins=30,
    alpha=0.5,
    label="Validation Normal"
)

plt.hist(
    test_mse,
    bins=30,
    alpha=0.5,
    label="Test Normal"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"95% Threshold = {threshold:.4f}"
)

plt.xlabel("Reconstruction MSE")
plt.ylabel("Number of Samples")
plt.title("Normal Data Reconstruction Error Distribution")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "normal_mse_distribution.png",
    dpi=150
)

plt.close()

# 8. 계산 결과 저장
np.savez(
    "normal_threshold.npz",
    threshold=threshold,
    train_mse=train_mse,
    val_mse=val_mse,
    test_mse=test_mse
)

print("\n===== 계산 완료 =====")
print("저장 파일:")
print("- normal_threshold.npz")
print("- normal_mse_distribution.png")