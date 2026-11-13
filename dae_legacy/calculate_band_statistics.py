from pathlib import Path
import numpy as np
import torch
from model import ConvAutoencoder

BASE_DIR = Path(__file__).resolve().parent

# 1. 정상 검증 데이터 불러오기
data = np.load(
    BASE_DIR / "normal_dataset_normalized.npz"
)

X_val = data["X_val"].astype(np.float32)

if X_val.ndim != 2 or X_val.shape[1] != 800:
    raise ValueError(
        f"정상 검증 데이터 크기 오류: {X_val.shape}"
    )

if not np.isfinite(X_val).all():
    raise ValueError("정상 검증 데이터에 이상값이 있습니다.")

print("정상 검증 데이터:", X_val.shape)

# 2. 기존 DAE 모델 불러오기
model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        BASE_DIR / "autoencoder_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 3. 정상 검증 데이터 복원
X_tensor = torch.from_numpy(
    X_val
).unsqueeze(1)

reconstructed_batches = []

with torch.no_grad():
    for batch in torch.split(X_tensor, 64):
        reconstructed = model(batch)
        reconstructed_batches.append(
            reconstructed.squeeze(1).numpy()
        )

X_reconstructed = np.concatenate(
    reconstructed_batches,
    axis=0
)

# 4. 주파수축 생성
fs = 25600
nfft = 8192

freq_bins = np.arange(3200) * (fs / nfft)
freq_800 = freq_bins.reshape(800, 4).mean(axis=1)

# 5. 대역별 MSE 계산
squared_error = (
    X_val - X_reconstructed
) ** 2

frequency_bands = {
    "low": (0, 100),
    "mid": (100, 1000),
    "high": (1000, 10000)
}

band_names = []
band_means = []
band_stds = []

for band_name, (low, high) in frequency_bands.items():
    mask = (
        (freq_800 >= low)
        & (freq_800 < high)
    )

    if not np.any(mask):
        raise ValueError(
            f"{band_name} 대역에 데이터가 없습니다."
        )

    # 검증 샘플마다 해당 대역의 MSE 계산
    band_errors = np.mean(
        squared_error[:, mask],
        axis=1
    )

    mean_error = float(
        np.mean(band_errors)
    )

    std_error = float(
        np.std(band_errors)
    )

    if std_error <= 0:
        raise ValueError(
            f"{band_name} 대역의 표준편차가 0입니다."
        )

    band_names.append(band_name)
    band_means.append(mean_error)
    band_stds.append(std_error)

    print(
        f"{band_name}: "
        f"평균={mean_error:.6f}, "
        f"표준편차={std_error:.6f}"
    )

# 6. 별도 파일로 저장
output_path = BASE_DIR / "band_statistics.npz"

np.savez(
    output_path,
    band_names=np.array(band_names),
    band_means=np.array(band_means),
    band_stds=np.array(band_stds)
)

print("\n대역별 정상 기준 통계 계산 완료!")
print("저장 파일:", output_path)
