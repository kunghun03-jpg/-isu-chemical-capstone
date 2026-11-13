"""Optional DAE web adapter; no changes to the three-model result/DB schema."""

from io import BytesIO
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
import streamlit as st

from vibration.dae_bundle import check_rpm, load_manifest
from vibration.dae_inference import SavedDAEBundleInference
from vibration.dae_preprocessing import preprocess_mat_bytes
from vibration.web_inference import MAX_UPLOAD_BYTES

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "outputs/dae_bundle"


def bundle_status(rpm):
    folder = Path(os.environ.get("ISU_CHEMICAL_DAE_BUNDLE", str(DEFAULT_BUNDLE))).expanduser()
    try:
        manifest = load_manifest(folder)
    except FileNotFoundError:
        return {"label": "모델 미준비", "ready": False, "folder": str(folder), "id": None}
    except Exception as error:
        return {"label": "모델 검증 실패", "ready": False, "folder": str(folder),
                "id": None, "reason": str(error)}
    state = {"folder": str(folder), "id": manifest["bundle_id"], "ready": False}
    if manifest["supported_rpms"] is None:
        return dict(state, label="RPM 적용 범위 미확인")
    try:
        check_rpm(manifest, rpm)
    except ValueError:
        return dict(state, label="선택 RPM 미지원")
    return dict(state, label="사용 가능", ready=True)


def preview_dae_mat(raw):
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("비어 있지 않은 20MB 이하 MAT 파일을 업로드해 주세요.")
    # DAE-only preview must not run the existing FFT/log1p preprocessing.
    preprocess_mat_bytes(raw)
    data = loadmat(BytesIO(raw), variable_names=("signals", "fs"))
    signal = data["signals"][2].copy()
    return {"signal": signal, "time": np.arange(65536) / 25600.,
            "fs": 25600., "samples": 65536, "duration": 2.56}


def predict_dae(raw, rpm, expected_id):
    folder = Path(os.environ.get("ISU_CHEMICAL_DAE_BUNDLE", str(DEFAULT_BUNDLE))).expanduser()
    # Recheck actual artifacts and RPM at execution time; never use the legacy bypass.
    engine = SavedDAEBundleInference(folder)
    if engine.manifest["bundle_id"] != expected_id:
        raise ValueError("DAE 번들이 변경되었습니다. 화면을 갱신한 뒤 다시 분석하세요.")
    return engine.predict_bytes(raw, rpm=rpm)


def render_dae_result(result):
    st.subheader("DAE 분석 결과")
    st.dataframe(pd.DataFrame([{
        "모델": "DAE", "파일 MSE": result["mse"], "저장 임계값": result["threshold"],
        "판정": result["prediction"],
    }]), hide_index=True, column_config={
        "파일 MSE": st.column_config.NumberColumn(format="%.12g"),
        "저장 임계값": st.column_config.NumberColumn(format="%.12g"),
    })
    st.caption("DAE는 파일 전체 Welch 800특성의 복원 MSE를 사용합니다. 8구간 평균 점수가 아닙니다.")
    original = np.asarray(result["input_spectrum"]).reshape(800)
    restored = np.asarray(result["reconstructed_spectrum"]).reshape(800)
    frequency = (np.arange(3200) * (25600. / 8192)).reshape(800, 4).mean(axis=1)
    st.line_chart(pd.DataFrame({"주파수 (Hz)": frequency,
                               "DAE 입력": original, "DAE 복원": restored}),
                  x="주파수 (Hz)", y=["DAE 입력", "DAE 복원"])
    st.caption("Welch PSD 제곱근 → 4개 bin 평균 → log10 → 저장 Z-score 적용값입니다. "
               "다른 모델의 점수와 절댓값을 직접 비교하지 마세요.")
    st.info("DAE 결과의 이력 DB 저장은 아직 지원하지 않습니다.")
