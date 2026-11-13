from pathlib import Path
import numpy as np
import torch
from model import ConvAutoencoder
from web_preprocessing import preprocess_uploaded_mat

BASE_DIR = Path(__file__).resolve().parent

def run_dae_inference(uploaded_file):
    # 1. 기존 학습 결과 불러오기
    normal_data = np.load(
        BASE_DIR / "normal_dataset_normalized.npz"
    )

    threshold_data = np.load(
        BASE_DIR / "normal_threshold.npz"
    )

    mean = normal_data["mean"]
    std = normal_data["std"]
    threshold = float(threshold_data["threshold"])

    if np.any(std <= 0):
        raise ValueError("정규화 표준편차가 올바르지 않습니다.")

    # 2. MAT 파일 전처리
    log_spectrum = preprocess_uploaded_mat(
        uploaded_file
    )

    # 3. 정상 학습 데이터 기준 Z-score
    normalized = (log_spectrum - mean) / std

    if not np.isfinite(normalized).all():
        raise ValueError("정규화 결과에 이상값이 있습니다.")

    # 4. DAE 입력 형태: (1, 1, 800)
    input_tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).reshape(1, 1, 800)

    # 5. 학습된 모델 불러오기
    model = ConvAutoencoder()

    model.load_state_dict(
        torch.load(
            BASE_DIR / "autoencoder_model.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    # 6. 복원 및 MSE 계산
    with torch.no_grad():
        reconstructed = model(input_tensor)

        mse = torch.mean(
            (input_tensor - reconstructed) ** 2,
            dim=(1, 2)
        ).item()

    # 7. 정상·이상 판정
    is_anomaly = mse > threshold

    return {
        "prediction": "이상" if is_anomaly else "정상",
        "mse": mse,
        "threshold": threshold,
        "input_spectrum": normalized,
        "reconstructed_spectrum": (
            reconstructed.squeeze().numpy()
        )
    }
