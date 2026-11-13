"""정상 학습·검증 기반 Isolation Forest 비교 모델 실행."""

from vibration.one_class import run_model


def run(project_dir, output_dir, rpm):
    run_model(output_dir, rpm, "iforest")
