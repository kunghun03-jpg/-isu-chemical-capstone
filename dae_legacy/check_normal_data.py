import numpy as np

data = np.load("normal_dataset_normalized.npz")

print("저장된 배열 이름:", data.files)

for name in data.files:
    print(f"{name}: {data[name].shape}")