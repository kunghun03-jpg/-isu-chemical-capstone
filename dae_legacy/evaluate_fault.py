import numpy as np
import torch
import matplotlib.pyplot as plt
from model import ConvAutoencoder

# 1. 데이터 불러오기
fault_data = np.load("fault_dataset.npz")
normal_data = np.load("normal_dataset_normalized.npz")
threshold_data = np.load("normal_threshold.npz")

X_fault = fault_data["X"]
labels = fault_data["labels"]
groups = fault_data["groups"]

# 정상 학습 데이터에서 계산한 평균과 표준편차
mean = normal_data["mean"]
std = normal_data["std"]

threshold = float(threshold_data["threshold"])

print("고장 데이터 크기:", X_fault.shape)
print("이상 탐지 임계값:", threshold)

# 2. 고장 데이터 정규화
# 고장 데이터 자체의 평균/표준편차 사용 금지
X_fault_norm = (X_fault - mean) / std

if not np.isfinite(X_fault_norm).all():
    raise ValueError("정규화 데이터에 NaN 또는 Inf 존재")

X_tensor = torch.tensor(
    X_fault_norm,
    dtype=torch.float32
).unsqueeze(1)

# 3. 학습된 모델 불러오기
model = ConvAutoencoder()

model.load_state_dict(
    torch.load(
        "autoencoder_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# 4. 고장 데이터 복원 오차 계산
mse_list = []

with torch.no_grad():
    for batch in torch.split(X_tensor, 64):
        reconstructed = model(batch)

        mse = torch.mean(
            (batch - reconstructed) ** 2,
            dim=(1, 2)
        )

        mse_list.extend(mse.numpy())

mse_list = np.asarray(mse_list)

# 임계값 초과 → 이상 판정
predicted_fault = mse_list > threshold

# 5. 전체 고장 탐지율
total = len(predicted_fault)
detected = int(predicted_fault.sum())

detection_rate = detected / total * 100

print("\n===== 전체 고장 탐지 결과 =====")
print("전체 고장 데이터:", total)
print("이상으로 탐지:", detected)
print("탐지 실패:", total - detected)
print(f"고장 탐지율: {detection_rate:.2f}%")

# 6. 고장 유형별 탐지율
print("\n===== 고장 유형별 탐지율 =====")

for fault_type in np.unique(labels):
    mask = labels == fault_type

    count = int(mask.sum())
    found = int(predicted_fault[mask].sum())

    rate = found / count * 100

    print(
        f"{fault_type:18s} "
        f"전체 {count:4d}개 / "
        f"탐지 {found:4d}개 / "
        f"탐지율 {rate:6.2f}%"
    )

# 7. RPM별 탐지율
print("\n===== RPM별 탐지율 =====")

for rpm in [900, 1200, 1800]:
    mask = np.array([
        f"_{rpm}RPM" in group
        for group in groups
    ])

    count = int(mask.sum())

    if count == 0:
        continue

    found = int(predicted_fault[mask].sum())

    print(
        f"{rpm}RPM: "
        f"{found}/{count} "
        f"({found/count*100:.2f}%)"
    )

# 8. 정상/고장 MSE 분포 비교
normal_test_mse = threshold_data["test_mse"]

plt.figure(figsize=(10, 6))

plt.hist(
    normal_test_mse,
    bins=35,
    alpha=0.6,
    label="Normal Test"
)

plt.hist(
    mse_list,
    bins=35,
    alpha=0.6,
    label="Fault"
)

plt.axvline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.4f}"
)

plt.xlabel("Reconstruction MSE")
plt.ylabel("Number of Samples")
plt.title("Normal vs Fault Reconstruction Error")
plt.legend()
plt.tight_layout()

plt.savefig(
    "normal_vs_fault_mse.png",
    dpi=300
)

plt.close()

# 9. 결과 저장
np.savez(
    "fault_evaluation_results.npz",
    mse=mse_list,
    labels=labels,
    groups=groups,
    predicted_fault=predicted_fault,
    threshold=threshold
)

print("\n평가 완료!")
print("그래프: normal_vs_fault_mse.png")
print("결과: fault_evaluation_results.npz")

# 10. ROC 곡선 및 AUC 평가
from sklearn.metrics import roc_curve, roc_auc_score

# 정상 시험 데이터: 0
# 고장 데이터: 1
y_true = np.concatenate([
    np.zeros(len(normal_test_mse)),
    np.ones(len(mse_list))
])

# 복원 오차가 클수록 이상일 가능성이 높음
y_score = np.concatenate([
    normal_test_mse,
    mse_list
])

auc_score = roc_auc_score(y_true, y_score)

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

print("\n===== ROC 평가 결과 =====")
print(f"ROC AUC: {auc_score:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Autoencoder (AUC={auc_score:.4f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)
plt.scatter(
    19 / 189,
    1747 / 3120,
    color="red",
    s=70,
    label="Current Threshold"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.close()
print("ROC 그래프 저장: roc_curve.png")


# 11. ROC 곡선 및 AUC 계산
normal_scores = np.asarray(normal_test_mse)
fault_scores = np.asarray(mse_list)

# 정상: 0, 고장: 1
y_true = np.concatenate([
    np.zeros(len(normal_scores), dtype=int),
    np.ones(len(fault_scores), dtype=int)
])

scores = np.concatenate([
    normal_scores,
    fault_scores
])

# 복원 오차가 큰 순서대로 정렬
order = np.argsort(-scores, kind="mergesort")
sorted_scores = scores[order]
sorted_labels = y_true[order]

# 같은 MSE 점수는 한 번에 처리
group_ends = np.r_[
    np.where(np.diff(sorted_scores) != 0)[0],
    len(sorted_scores) - 1
]

tp = np.cumsum(sorted_labels)[group_ends]
fp = (group_ends + 1) - tp

tpr = np.r_[0, tp / len(fault_scores)]
fpr = np.r_[0, fp / len(normal_scores)]

# ROC 곡선 아래 면적
auc_score = np.trapezoid(tpr, fpr)

print("\n===== ROC 평가 결과 =====")
print(f"ROC AUC: {auc_score:.4f}")

# ROC 그래프
plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Autoencoder (AUC={auc_score:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)

# 현재 임계값에서의 성능 표시
current_fpr = np.mean(normal_scores > threshold)
current_tpr = np.mean(fault_scores > threshold)

plt.scatter(
    current_fpr,
    current_tpr,
    color="red",
    s=80,
    label="Current Threshold"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("roc_curve.png", dpi=300)
plt.close()

print("ROC 그래프 저장 완료: roc_curve.png")

# 12. 고장 유형별 MSE 분포 분석
fault_types = [
    "IR",
    "OR",
    "Roller",
    "Looseness0.05",
    "Looseness0.1",
    "LoosenessFull",
    "Unbalance_-1",
    "Unbalance_-2"
]

# 고장 유형별 MSE 데이터
fault_mse_groups = [
    mse_list[labels == fault_type]
    for fault_type in fault_types
]

# ======================================
# 1. 고장 유형별 박스플롯
# ======================================

plt.figure(figsize=(13, 6))

plt.boxplot(
    fault_mse_groups,
    tick_labels=fault_types,
    showfliers=True
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.4f}"
)

plt.xticks(rotation=30, ha="right")
plt.ylabel("Reconstruction MSE")
plt.title("Reconstruction Error by Fault Type")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "fault_type_mse_boxplot.png",
    dpi=300
)

plt.close()

# ======================================
# 2. 고장 유형별 MSE 통계
# ======================================

print("\n===== 고장 유형별 MSE 통계 =====")

for fault_type, errors in zip(
    fault_types,
    fault_mse_groups
):
    if len(errors) == 0:
        continue

    print(
        f"{fault_type:18s} | "
        f"평균: {np.mean(errors):.5f} | "
        f"중앙값: {np.median(errors):.5f} | "
        f"최솟값: {np.min(errors):.5f} | "
        f"최댓값: {np.max(errors):.5f}"
    )

# ======================================
# 3. 정상 + 고장 유형별 MSE 분포
# ======================================

all_names = ["Normal"] + fault_types

all_mse_groups = [
    normal_test_mse
] + fault_mse_groups

plt.figure(figsize=(14, 6))

plt.boxplot(
    all_mse_groups,
    tick_labels=all_names,
    showfliers=False
)

plt.axhline(
    threshold,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {threshold:.4f}"
)

plt.xticks(rotation=30, ha="right")
plt.ylabel("Reconstruction MSE")
plt.title("Normal vs Fault Type Reconstruction Error")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "normal_fault_type_boxplot.png",
    dpi=300
)

plt.close()

print("\n고장 유형별 MSE 분석 완료!")
print("저장된 그래프:")
print("1. fault_type_mse_boxplot.png")
print("2. normal_fault_type_boxplot.png")