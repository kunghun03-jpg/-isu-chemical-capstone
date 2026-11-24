"""Read-only AE reconstruction visualization. Operational inference is unchanged."""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from vibration.inference import SavedModelInference
from vibration.preprocessing import frequency_bins


def reconstruction_data(raw, rpm, config, result):
    """Reconstruct once and verify parity with the operational AE segment scores."""
    import torch

    engine = SavedModelInference(
        config.bundle_root / f"{rpm}rpm",
        models=("autoencoder",),
        selected_artifacts_only=True,
    )
    if result["input"]["rpm"] != rpm:
        raise ValueError("분석 결과의 RPM이 현재 선택한 RPM과 다릅니다.")
    import hashlib
    if hashlib.sha256(raw).hexdigest() != result["input"]["sha256"]:
        raise ValueError("현재 업로드 파일과 AE 분석 파일이 다릅니다.")
    if engine.manifest["bundle_id"] != result["bundle_id"]:
        raise ValueError("현재 AE 번들과 분석 당시 번들이 다릅니다.")

    _, normalized = engine.preprocess(raw)
    model = engine._estimators["autoencoder"]
    with torch.inference_mode():
        batch = torch.from_numpy(normalized).float()
        reconstructed = model(batch)
        scores = ((reconstructed.double() - batch.double())
                  .square().mean(dim=(1, 2)).numpy())

    item = result["models"]["autoencoder"]
    expected = np.asarray(
        [s["score"] for s in sorted(item["segments"], key=lambda s: s["segment_id"])],
        dtype=np.float64,
    )
    if expected.shape != (8,) or not np.allclose(scores, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("운영 AE 구간별 점수와 복원 결과가 일치하지 않습니다.")
    if not np.isclose(scores.mean(), item["file_score"], rtol=1e-10, atol=1e-12):
        raise ValueError("운영 AE 파일 점수와 복원 결과가 일치하지 않습니다.")

    return {
        "frequency": frequency_bins(),
        "input": normalized[:, 0, :].copy(),
        "reconstruction": reconstructed.numpy()[:, 0, :].copy(),
        "scores": scores.copy(),
    }


def render_reconstruction(data, result):
    st.subheader("AE 입력·복원 Spectrum 비교")
    if data is None:
        st.info("MAT 파일 분석 후 복원 결과를 확인할 수 있습니다.")
        return

    frequency = np.asarray(data["frequency"], dtype=np.float64)
    inputs = np.asarray(data["input"], dtype=np.float64)
    reconstructions = np.asarray(data["reconstruction"], dtype=np.float64)
    scores = np.asarray(data["scores"], dtype=np.float64)
    if (frequency.shape != (3201,) or inputs.shape != (8, 3201)
            or reconstructions.shape != inputs.shape or scores.shape != (8,)
            or not np.isfinite(inputs).all() or not np.isfinite(reconstructions).all()
            or not np.isfinite(scores).all()):
        st.error("AE 복원 시각화 데이터의 크기 또는 수치가 올바르지 않습니다.")
        return

    segment = st.selectbox(
        "분석 구간 선택",
        options=list(range(8)),
        index=int(np.argmax(scores)),
        format_func=lambda i: f"구간 {i + 1}",
        key="v3_ae_reconstruction_segment",
    )
    range_option = st.radio(
        "주파수 표시 범위",
        ("전체 0–10kHz", "저주파 0–500Hz", "중간 대역 500–5000Hz"),
        horizontal=True,
        key="v3_ae_frequency_range",
    )
    if range_option == "저주파 0–500Hz":
        mask = frequency <= 500
    elif range_option == "중간 대역 500–5000Hz":
        mask = (frequency >= 500) & (frequency <= 5000)
    else:
        mask = np.ones(frequency.shape, dtype=bool)

    original = inputs[segment]
    reconstructed = reconstructions[segment]
    squared_error = (original - reconstructed) ** 2

    frame = pd.DataFrame({
        "주파수 (Hz)": frequency[mask],
        "입력 Spectrum": original[mask],
        "복원 Spectrum": reconstructed[mask],
    })
    long_frame = frame.melt(
        id_vars="주파수 (Hz)", var_name="구분", value_name="정규화 Spectrum"
    )
    chart = alt.Chart(long_frame).mark_line().encode(
        x=alt.X("주파수 (Hz):Q", title="주파수 (Hz)"),
        y=alt.Y("정규화 Spectrum:Q", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "구분:N",
            scale=alt.Scale(
                domain=["입력 Spectrum", "복원 Spectrum"],
                range=["#2563eb", "#ea580c"],
            ),
        ),
        tooltip=[
            alt.Tooltip("주파수 (Hz):Q", format=".1f"),
            "구분:N",
            alt.Tooltip("정규화 Spectrum:Q", format=".5f"),
        ],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")

    scale_option = st.radio(
        "제곱오차 표시 축",
        ("선형", "로그"),
        horizontal=True,
        key="v3_ae_error_scale",
    )
    # Log display requires positive values; do not modify scores or original error.
    display_error = (np.maximum(squared_error[mask], 1e-12)
                     if scale_option == "로그" else squared_error[mask])
    error_frame = pd.DataFrame({
        "주파수 (Hz)": frequency[mask],
        "표시용 제곱오차": display_error,
    })
    error_chart = alt.Chart(error_frame).mark_line(color="#d97757").encode(
        x=alt.X("주파수 (Hz):Q", title="주파수 (Hz)"),
        y=alt.Y(
            "표시용 제곱오차:Q",
            title="제곱오차",
            scale=alt.Scale(type="log" if scale_option == "로그" else "linear"),
        ),
        tooltip=[
            alt.Tooltip("주파수 (Hz):Q", format=".1f"),
            alt.Tooltip("표시용 제곱오차:Q", format=".6g"),
        ],
    ).properties(height=220)
    st.altair_chart(error_chart, width="stretch")
    if scale_option == "로그":
        st.caption("로그 축에서는 0인 제곱오차를 표시용으로만 1e-12로 대체합니다.")

    item = result["models"]["autoencoder"]
    a, b, c = st.columns(3)
    a.metric("선택 구간 MSE", f"{scores[segment]:.6f}")
    b.metric("파일 평균 MSE", f"{item['file_score']:.6f}")
    c.metric("파일 Q95 임계값", f"{item['file_threshold']:.6f}")

    top_indices = np.argsort(squared_error)[-10:][::-1]
    with st.expander("복원오차 상위 10개 주파수 bin · 전체 대역 기준"):
        st.dataframe(pd.DataFrame({
            "주파수 (Hz)": frequency[top_indices],
            "제곱오차": squared_error[top_indices],
        }), hide_index=True, width="stretch")

    st.caption(
        "Train 기준 주파수별 Z-score 정규화 Spectrum입니다. "
        "표시 범위와 로그 축은 AE 점수·임계값·판정에 영향을 주지 않습니다."
    )
    st.caption(
        "파일 판정은 8개 구간 평균 MSE와 저장된 파일 임계값으로 결정합니다. "
        "주파수별 복원오차만으로 결함 원인을 확정할 수 없으며, "
        "실제 측정 Run 간 독립성은 검증되지 않았습니다."
    )
