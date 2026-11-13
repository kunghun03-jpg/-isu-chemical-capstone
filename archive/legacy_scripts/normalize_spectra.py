# 보관된 초기 실험 코드: normalize_spectra. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]
output_dir = project_dir / "outputs_clean" / "900rpm" / "first"

# 정상 학습 데이터만으로 정규화 기준 계산
with np.load(output_dir / "train_spectra.npz") as data:
    train = data["X"].astype(np.float64)

# 채널별·주파수별 평균과 표준편차
mean = train.mean(axis=0, keepdims=True)
std = train.std(axis=0, keepdims=True)

# 변동이 거의 없는 주파수에서 과도하게 나누는 것 방지
std = np.maximum(std, 1e-6)

np.savez_compressed(output_dir / "normalization_900rpm.npz", mean=mean, std=std)

del train

# 모든 데이터에 동일한 기준 적용
for split in ("train", "validation", "test_normal", "test_anomaly"):
    source_path = output_dir / f"{split}_spectra.npz"

    with np.load(source_path) as data:
        X = data["X"].astype(np.float64)
        X = ((X - mean) / std).astype(np.float32)

        if not np.isfinite(X).all():
            raise ValueError(f"{split}: 정규화 결과 확인 필요")

        np.savez_compressed(
            output_dir / f"{split}_normalized.npz",
            X=X,
            frequency=data["frequency"],
            source_file=data["source_file"],
            condition=data["condition"],
            segment_id=data["segment_id"],
        )

    print(f"{split}: {X.shape}, 유한값 확인 완료")
    del X

print("정규화 완료!")
