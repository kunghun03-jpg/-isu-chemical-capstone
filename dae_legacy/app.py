
import io
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
import streamlit as st

from dae_inference import run_dae_inference


# ==================================================
# 1. 기본 설정
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="회전기기 진동 이상 탐지 시스템",
    layout="wide"
)

st.sidebar.title("VIBRA AI")

menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "대시보드",
        "데이터 입력",
        "AI 이상 탐지",
        "모델 결과 비교",
        "상세 분석 및 결함 추정",
        "분석 이력"
    ]
)

st.title(menu)


# ==================================================
# 2. 정상 기준 대역별 통계 불러오기
# ==================================================

def load_band_statistics():
    file_path = BASE_DIR / "band_statistics.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            "band_statistics.npz 파일이 없습니다. "
            "calculate_band_statistics.py를 먼저 실행하세요."
        )

    with np.load(file_path) as data:
        names = data["band_names"].astype(str)
        means = data["band_means"].astype(float)
        stds = data["band_stds"].astype(float)

    expected_names = ["low", "mid", "high"]

    if names.tolist() != expected_names:
        raise ValueError(
            "대역별 통계의 순서가 예상과 다릅니다. "
            f"현재 순서: {names.tolist()}"
        )

    if (
        means.shape != (3,)
        or stds.shape != (3,)
    ):
        raise ValueError(
            "대역별 통계 데이터 크기가 올바르지 않습니다."
        )

    if (
        not np.isfinite(means).all()
        or not np.isfinite(stds).all()
    ):
        raise ValueError(
            "대역별 통계에 NaN 또는 무한대가 있습니다."
        )

    if np.any(stds <= 0):
        raise ValueError(
            "대역별 표준편차는 0보다 커야 합니다."
        )

    return means, stds


# ==================================================
# 3. 대시보드
# ==================================================

if menu == "대시보드":

    st.write(
        "회전기기 진동 모니터링 대시보드"
    )


# ==================================================
# 4. 데이터 입력
# ==================================================

elif menu == "데이터 입력":

    st.subheader("진동 데이터 입력")

    uploaded_file = st.file_uploader(
        "MAT 파일 업로드",
        type=["mat"]
    )

    if uploaded_file is not None:

        try:
            mat_data = loadmat(
                io.BytesIO(
                    uploaded_file.getvalue()
                )
            )

            signals = np.asarray(
                mat_data["signals"]
            )

            fs = float(
                np.asarray(
                    mat_data["fs"]
                ).squeeze()
            )

            if (
                signals.ndim != 2
                or signals.shape[0] < 3
            ):
                raise ValueError(
                    "CH3 채널을 찾을 수 없습니다."
                )

            if not np.isfinite(fs) or fs <= 0:
                raise ValueError(
                    "샘플링 주파수가 올바르지 않습니다."
                )

            ch3 = signals[2, :]

            time = (
                np.arange(len(ch3)) / fs
            )

            st.success(
                f"파일 읽기 성공: {uploaded_file.name}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "분석 채널",
                "CH3"
            )

            col2.metric(
                "샘플링 주파수",
                f"{fs:,.0f} Hz"
            )

            col3.metric(
                "측정 시간",
                f"{len(ch3) / fs:.2f}초"
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=ch3,
                    mode="lines",
                    name="CH3 진동 신호",
                    line=dict(
                        color="#19A7A0",
                        width=1
                    )
                )
            )

            fig.update_layout(
                title="CH3 시간 영역 진동 파형",
                xaxis_title="시간 (초)",
                yaxis_title="가속도 (원본 데이터 단위)",
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:
            st.error(
                f"MAT 파일 읽기 오류: {e}"
            )


# ==================================================
# 5. AI 이상 탐지
# ==================================================

elif menu == "AI 이상 탐지":

    st.subheader(
        "DAE 기반 진동 이상 탐지"
    )

    dae_file = st.file_uploader(
        "분석할 MAT 파일 업로드",
        type=["mat"],
        key="dae_upload"
    )

    if dae_file is not None:

        st.success(
            f"분석 파일: {dae_file.name}"
        )

        if st.button(
            "DAE 이상 탐지 실행"
        ):

            try:

                # ----------------------------------
                # 5-1. DAE 이상 탐지 실행
                # ----------------------------------

                with st.spinner(
                    "DAE 모델 분석 중..."
                ):
                    result = run_dae_inference(
                        dae_file
                    )

                st.subheader(
                    "이상 탐지 결과"
                )

                if result["prediction"] == "이상":
                    st.error(
                        "분석 결과: 이상"
                    )
                else:
                    st.success(
                        "분석 결과: 정상"
                    )

                col1, col2 = st.columns(2)

                col1.metric(
                    "복원 오차 (MSE)",
                    f'{result["mse"]:.6f}'
                )

                col2.metric(
                    "이상 판단 임계값",
                    f'{result["threshold"]:.6f}'
                )

                st.caption(
                    "전체 주파수 스펙트럼의 복원 MSE가 "
                    "기존 임계값보다 크면 이상으로 판정합니다."
                )

                # ----------------------------------
                # 5-2. 주파수축 생성
                # ----------------------------------

                fs = 25600
                nfft = 8192

                freq_bins = (
                    np.arange(3200)
                    * (fs / nfft)
                )

                freq_800 = (
                    freq_bins
                    .reshape(800, 4)
                    .mean(axis=1)
                )

                input_spectrum = np.asarray(
                    result["input_spectrum"]
                ).reshape(-1)

                reconstructed_spectrum = np.asarray(
                    result["reconstructed_spectrum"]
                ).reshape(-1)

                if (
                    len(input_spectrum) != 800
                    or len(reconstructed_spectrum) != 800
                ):
                    raise ValueError(
                        "스펙트럼 데이터 크기가 800이 아닙니다."
                    )

                if (
                    not np.isfinite(input_spectrum).all()
                    or not np.isfinite(
                        reconstructed_spectrum
                    ).all()
                ):
                    raise ValueError(
                        "스펙트럼에 NaN 또는 무한대가 있습니다."
                    )

                # ----------------------------------
                # 5-3. 입력 및 복원 스펙트럼
                # ----------------------------------

                st.subheader(
                    "입력 및 복원 주파수 스펙트럼"
                )

                fig_spectrum = go.Figure()

                fig_spectrum.add_trace(
                    go.Scatter(
                        x=freq_800,
                        y=input_spectrum,
                        mode="lines",
                        name="입력 스펙트럼",
                        line=dict(
                            color="#19A7A0",
                            width=1.5
                        )
                    )
                )

                fig_spectrum.add_trace(
                    go.Scatter(
                        x=freq_800,
                        y=reconstructed_spectrum,
                        mode="lines",
                        name="DAE 복원 스펙트럼",
                        line=dict(
                            color="#F39C4A",
                            width=1.5
                        )
                    )
                )

                fig_spectrum.update_layout(
                    title="DAE 입력 및 복원 스펙트럼 비교",
                    xaxis_title="주파수 (Hz)",
                    yaxis_title=(
                        "Z-score 정규화된 로그 스펙트럼"
                    ),
                    height=450,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0
                    )
                )

                st.plotly_chart(
                    fig_spectrum,
                    use_container_width=True
                )

                st.caption(
                    "청록색은 실제 입력 스펙트럼, "
                    "주황색은 DAE 복원 스펙트럼입니다. "
                    "두 값의 차이가 클수록 해당 주파수 "
                    "구간의 복원 오차가 큽니다."
                )

                # ----------------------------------
                # 5-4. 대역별 MSE 계산
                # ----------------------------------

                st.subheader(
                    "주파수 대역별 복원 오차"
                )

                squared_error = (
                    input_spectrum
                    - reconstructed_spectrum
                ) ** 2

                calculated_mse = float(
                    np.mean(squared_error)
                )

                frequency_bands = {
                    "저주파 (0~100 Hz)": (
                        0, 100
                    ),
                    "중주파 (100~1000 Hz)": (
                        100, 1000
                    ),
                    "고주파 (1000~10000 Hz)": (
                        1000, 10000
                    )
                }

                band_names = []
                band_mse_values = []

                for band_name, (
                    low, high
                ) in frequency_bands.items():

                    mask = (
                        (freq_800 >= low)
                        & (freq_800 < high)
                    )

                    if not np.any(mask):
                        raise ValueError(
                            f"{band_name} 대역에 "
                            "데이터가 없습니다."
                        )

                    band_mse = float(
                        np.mean(
                            squared_error[mask]
                        )
                    )

                    band_names.append(
                        band_name
                    )

                    band_mse_values.append(
                        band_mse
                    )

                # ----------------------------------
                # 5-5. 대역별 MSE 그래프
                # ----------------------------------

                fig_band = go.Figure()

                fig_band.add_trace(
                    go.Bar(
                        x=band_names,
                        y=band_mse_values,
                        marker_color=[
                            "#19A7A0",
                            "#4B8BD4",
                            "#F39C4A"
                        ],
                        text=[
                            f"{value:.6f}"
                            for value in band_mse_values
                        ],
                        textposition="outside"
                    )
                )

                fig_band.update_layout(
                    title="저·중·고주파 대역별 복원 MSE",
                    xaxis_title="주파수 대역",
                    yaxis_title="평균제곱오차 (MSE)",
                    height=420,
                    yaxis=dict(
                        rangemode="tozero"
                    )
                )

                st.plotly_chart(
                    fig_band,
                    use_container_width=True
                )

                band_cols = st.columns(3)

                for col, name, value in zip(
                    band_cols,
                    band_names,
                    band_mse_values
                ):
                    col.metric(
                        name,
                        f"{value:.6f}"
                    )

                max_mse_index = int(
                    np.argmax(
                        band_mse_values
                    )
                )

                st.info(
                    "이번 데이터에서 복원 MSE가 "
                    "가장 큰 대역은 "
                    f"**{band_names[max_mse_index]}**입니다."
                )

                st.caption(
                    "대역별 MSE는 입력·복원 차이를 "
                    "나타내는 참고 지표입니다. "
                    "MSE가 가장 크다고 해서 "
                    "특정 결함이 확인된 것은 아닙니다."
                )

                # ----------------------------------
                # 5-6. 대역별 Z-score 계산
                # ----------------------------------

                st.subheader(
                    "정상 기준 대역별 Z-score 분석"
                )

                band_means, band_stds = (
                    load_band_statistics()
                )

                band_mse_array = np.asarray(
                    band_mse_values,
                    dtype=np.float64
                )

                band_zscores = (
                    (band_mse_array - band_means)
                    / band_stds
                )

                # ----------------------------------
                # 5-7. Z-score 그래프
                # ----------------------------------

                fig_zscore = go.Figure()

                fig_zscore.add_trace(
                    go.Bar(
                        x=band_names,
                        y=band_zscores,
                        marker_color=[
                            "#19A7A0",
                            "#4B8BD4",
                            "#F39C4A"
                        ],
                        text=[
                            f"{value:+.2f}"
                            for value in band_zscores
                        ],
                        textposition="outside",
                        name="대역별 Z-score"
                    )
                )

                fig_zscore.add_hline(
                    y=0,
                    line_color="gray",
                    line_width=1
                )

                fig_zscore.update_layout(
                    title=(
                        "정상 검증 데이터 기준 "
                        "대역별 복원 오차 Z-score"
                    ),
                    xaxis_title="주파수 대역",
                    yaxis_title="Z-score",
                    height=440,
                    yaxis=dict(
                        zeroline=True,
                        autorange=True
                    )
                )

                st.plotly_chart(
                    fig_zscore,
                    use_container_width=True
                )

                # ----------------------------------
                # 5-8. Z-score 수치 표시
                # ----------------------------------

                zscore_cols = st.columns(3)

                for col, name, value in zip(
                    zscore_cols,
                    band_names,
                    band_zscores
                ):
                    col.metric(
                        name,
                        f"{value:+.2f}"
                    )

                # ----------------------------------
                # 5-9. 정상 기준 통계 표시
                # ----------------------------------

                with st.expander(
                    "정상 기준 통계 확인"
                ):

                    for (
                        name,
                        mean_value,
                        std_value
                    ) in zip(
                        band_names,
                        band_means,
                        band_stds
                    ):

                        st.write(
                            f"**{name}**"
                        )

                        st.write(
                            "정상 MSE 평균: "
                            f"{mean_value:.6f}"
                        )

                        st.write(
                            "정상 MSE 표준편차: "
                            f"{std_value:.6f}"
                        )

                    st.caption(
                        "정상 검증 데이터 179개를 "
                        "기준으로 계산한 통계입니다."
                    )

                # ----------------------------------
                # 5-10. Z-score 해석
                # ----------------------------------

                max_zscore_index = int(
                    np.argmax(
                        band_zscores
                    )
                )
                max_zscore_name = (
                    band_names[
                        max_zscore_index
                    ]
                )
                max_zscore_value = float(
                    band_zscores[
                        max_zscore_index
                    ]
                )
                st.info(
                    "정상 기준 대비 복원 오차의 "
                    "상대적 증가가 가장 큰 대역은 "
                    f"**{max_zscore_name}**입니다. "
                    f"(Z-score: {max_zscore_value:+.2f})"
                )
                if max_zscore_value <= 0:
                    st.caption(
                        "모든 대역의 복원 MSE가 "
                        "정상 검증 평균 이하입니다."
                    )
                st.caption(
                    "Z-score가 양수이면 해당 대역의 "
                    "복원 MSE가 정상 검증 평균보다 "
                    "높다는 뜻이며, 음수이면 "
                    "평균보다 낮다는 뜻입니다."
                )
                st.warning(
                    "대역별 Z-score는 정상 데이터 "
                    "기준의 상대적인 복원 오차를 "
                    "보여주는 보조 지표입니다. "
                    "Z-score가 가장 높은 대역만으로 "
                    "불평형, 풀림, 베어링 결함 등의 "
                    "원인을 확정할 수 없습니다."
                )
                st.caption(
                    "정상·이상 판정은 기존 전체 MSE "
                    "임계값을 그대로 사용합니다. "
                    "대역별 Z-score에는 아직 별도의 "
                    "이상 판정 임계값을 적용하지 않습니다."
                )

                # 5-11. 전체 MSE 일치 여부 확인
              
                if not np.isclose(
                    calculated_mse,
                    result["mse"],
                    rtol=1e-4,
                    atol=1e-6
                ):
                    st.warning(
                        "스펙트럼에서 재계산한 전체 MSE와 "
                        "모델이 반환한 MSE에 "
                        "차이가 있습니다."
                    )
            except Exception as e:
                st.error(
                    f"DAE 분석 오류: {e}"
                )

# 6. 아직 개발 중인 메뉴

else:
    st.info(
        "해당 기능은 개발 중입니다."
    )
