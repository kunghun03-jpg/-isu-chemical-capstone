"""P3 물리적 진동 특징 기반 실험용 웹 분석."""

import hashlib
import json

import pandas as pd
import streamlit as st

from vibration.physical_bundle import (
    verify_bundle,
    analyze_bundled_physical_mat,
)

from vibration.physical_inference import (
    SUPPORTED_DATES,
    SUPPORTED_RPMS,
    PhysicalInferenceError,
)

from vibration.web_inference import (
    MAX_UPLOAD_BYTES,
    preview_mat,
)


# 배포 패키지의 고정된 신뢰 기준
MANIFEST_SHA256 = (
    "dd0f655a7dd1f113f84bf7a4fa4f45e"
    "53374007f1dde4e0b09e6881aee426aad"
)


FEATURE_LABELS = {
    "rms": "RMS",
    "peak": "Peak",
    "kurtosis": "Kurtosis",
    "crest_factor": "Crest Factor",
    "amplitude_1x": "1X 진폭",
    "amplitude_2x": "2X 진폭",
    "amplitude_3x": "3X 진폭",
    "energy_low": "저주파 에너지",
    "energy_mid": "중주파 에너지",
    "energy_high": "고주파 에너지",
}


def predict_physical(raw, date, rpm):

    if not isinstance(raw, bytes):

        raise PhysicalInferenceError(
            "MAT 파일 bytes가 필요합니다."
        )

    if not raw or len(raw) > MAX_UPLOAD_BYTES:

        raise PhysicalInferenceError(
            "허용되지 않는 파일 크기입니다."
        )

    # 기존 웹 입력 검증 재사용
    preview_mat(raw)

    return analyze_bundled_physical_mat(
        raw,
        date,
        rpm,
        MANIFEST_SHA256,
    )


def clear_physical_result():

    st.session_state.pop(
        "physical_p3_result", None
    )


def reset_physical_ui():

    clear_physical_result()

    st.session_state.pop(
        "physical_p3_identity", None
    )

    st.session_state.pop(
        "physical_p3_confirm", None
    )


def render_physical_result(result):

    if (
        result.get("evaluation_status")
            != "exploratory_only"
        or result.get("approved_for_production")
            is not False
        or result.get("bundle_integrity_verified")
            is not True
    ):

        st.error(
            "P3 결과의 검증 상태가 올바르지 않습니다."
        )

        return

    st.markdown("#### P3 실험 분석 결과")

    cols = st.columns(3)

    cols[0].metric(
        "이상 점수",
        f"{result['score']:.6f}",
    )

    cols[1].metric(
        "저장 임계값",
        f"{result['threshold']:.6f}",
    )

    cols[2].metric(
        "임계값 대비 비율",
        f"{result['score_ratio']:.3f}배",
    )

    if result["predicted_anomaly"]:

        st.warning(
            "실험 판정: Abnormal (이상)"
        )

    else:

        st.info(
            "실험 판정: Normal (정상)"
        )

    st.caption(
        f"선택한 측정 조건: "
        f"{result['date']} / "
        f"{result['rpm']} RPM"
    )

    dominant = result["dominant_feature"]

    st.write(
        "가장 큰 이탈 특징:",
        FEATURE_LABELS.get(
            dominant, dominant
        ),
    )

    st.markdown("#### 특징별 정상 기준 이탈 점수")

    rows = []

    for name, value in (
        result["features"].items()
    ):

        rows.append({
            "특징": FEATURE_LABELS.get(
                name, name
            ),
            "측정값": value,
            "이탈 점수":
                result["feature_deviations"][name],
        })

    frame = pd.DataFrame(rows)

    st.bar_chart(
        frame.set_index("특징")[
            "이탈 점수"
        ]
    )

    st.dataframe(
        frame,
        hide_index=True,
    )

    st.caption(
        "이탈 점수는 정상 기준과의 차이를 "
        "MAD 기반 스케일로 변환한 값입니다. "
        "특정 특징만으로 결함 종류를 "
        "확정할 수 없습니다."
    )

    st.success(
        "P3 정상 기준 파일 무결성 검증 완료"
    )

    st.warning(
        "독립 Run 검증이 완료되지 않은 "
        "실험용 판정입니다. 실제 측정 조건은 "
        "사용자가 선택한 값이며 자동 검증되지 "
        "않았습니다."
    )

    st.download_button(
        "P3 실험 결과 JSON 다운로드",
        data=json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        file_name="physical_p3_result.json",
        mime="application/json",
        key="physical_p3_download",
    )


def render_physical_analysis(raw, rpm):

    st.divider()

    st.subheader(
        "물리적 진동 특징 분석 (P3)"
    )

    st.caption(
        "실험용 보조 알고리즘 · "
        "기존 AE 판정을 변경하지 않습니다."
    )

    st.write(
        "RMS, Kurtosis, 회전 주파수 진폭, "
        "대역별 에너지 등 10개 특징을 "
        "이용하여 정상 기준 이탈 정도를 "
        "분석합니다."
    )

    if not raw:

        reset_physical_ui()
        st.info("MAT 파일을 먼저 업로드하세요.")
        return

    date = st.selectbox(
        "P3 실제 측정 날짜",
        ("",) + SUPPORTED_DATES,
        format_func=lambda value: (
            f"{value[:4]}-{value[4:6]}-{value[6:]}"
            if value
            else "실제 측정 날짜를 선택하세요"
        ),
        key="physical_p3_date",
    )

    identity = (
        hashlib.sha256(raw).hexdigest(),
        date,
        rpm,
        MANIFEST_SHA256,
    )

    if (
        st.session_state.get(
            "physical_p3_identity"
        ) != identity
    ):

        clear_physical_result()

        st.session_state[
            "physical_p3_identity"
        ] = identity

        st.session_state[
            "physical_p3_confirm"
        ] = False

    confirmed = st.checkbox(
        "실제 측정 날짜와 RPM이 "
        "선택한 조건과 일치함을 확인했습니다.",
        key="physical_p3_confirm",
    )

    if not confirmed:

        clear_physical_result()

    try:

        verify_bundle(MANIFEST_SHA256)

        bundle_ready = True

    except PhysicalInferenceError:

        bundle_ready = False

        clear_physical_result()

        st.error(
            "P3 정상 기준 파일 검증 실패. "
            "실험 분석을 실행할 수 없습니다."
        )

    ready = (
        bundle_ready
        and date in SUPPORTED_DATES
        and rpm in SUPPORTED_RPMS
        and confirmed
    )

    if st.button(
        "P3 실험 분석 실행",
        disabled=not ready,
        key="physical_p3_run",
    ):

        clear_physical_result()

        try:

            with st.spinner(
                "물리적 진동 특징 분석 중…"
            ):

                result = predict_physical(
                    raw, date, rpm
                )

            st.session_state[
                "physical_p3_result"
            ] = result

        except Exception:

            clear_physical_result()

            st.error(
                "P3 분석 실패: 입력 파일 또는 "
                "정상 기준을 확인하세요."
            )

    result = st.session_state.get(
        "physical_p3_result"
    )

    if ready and result is not None:

        render_physical_result(result)
