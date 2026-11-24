"""Read-only, condition-labeled comparison of real MAT CH3 signals.

This page is independent of AE/RF/DAE decisions and never changes model artifacts.
"""
from pathlib import Path
from io import BytesIO
import json
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from vibration.web_inference import preview_mat
from scripts.analyze_fault_signatures import analyze_signal

DEFAULT_EDA = Path(__file__).resolve().parents[1] / "outputs/web_artifacts/fault_signatures"
CLASSES = {
    "Normal": ("Normal1", "Normal2"),
    "Bearing": ("IR", "OR", "Roller"),
    "Looseness": ("Looseness0.05", "Looseness0.1", "LoosenessFull"),
    "Unbalance": ("Unbalance_-1", "Unbalance_-2"),
}
PALETTE = ["#1870d5", "#b05cc4", "#e38b27", "#168e7a"]


@st.cache_data(show_spinner=False)
def load_index(csv_path, mtime):
    frame = pd.read_csv(csv_path, dtype={"date": str, "condition": str, "path": str})
    needed = {"path", "rpm", "date", "condition"}
    if not needed.issubset(frame.columns):
        raise ValueError("file_features.csv 필수 열 누락: " + ", ".join(sorted(needed - set(frame.columns))))
    frame["rpm"] = pd.to_numeric(frame["rpm"], errors="coerce")
    frame = frame[frame["condition"].isin(sum((list(v) for v in CLASSES.values()), []))].copy()
    frame["class"] = frame["condition"].map(
        {condition: label for label, conditions in CLASSES.items() for condition in conditions}
    )
    return frame


def locate_mat(recorded, mat_root):
    """Prefer the recorded path; optional root only resolves by filename, not by label."""
    p = Path(str(recorded)).expanduser()
    if p.is_file():
        return p
    if mat_root:
        root = Path(mat_root).expanduser()
        if root.is_dir():
            matches = list(root.rglob(p.name))
            if len(matches) == 1:
                return matches[0]
    return None


@st.cache_data(show_spinner=False, max_entries=12)
def calculate(path, size, mtime, rpm):
    raw = Path(path).read_bytes()
    preview = preview_mat(raw)  # same strict MAT/CH3 validation as inference
    fs = float(preview["fs"])
    x = np.asarray(preview["signal"], dtype=np.float64)
    observed = analyze_signal(x, fs, int(rpm))  # same EDA FFT and envelope contract
    return {
        "signal": x,
        "time": np.asarray(preview["time"]),
        "frequency": np.asarray(observed["frequency"]),
        "amplitude": np.asarray(observed["amplitude"]),
        "envelope_frequency": np.asarray(observed["envelope_frequency"]),
        "envelope_amplitude": np.asarray(observed["envelope_amplitude"]),
        "fs": fs,
    }


def compress_max(f, y, limit=1300):
    """Keep high spectral peaks while reducing browser chart size."""
    f, y = np.asarray(f), np.asarray(y)
    if len(f) <= limit:
        return f, y
    edges = np.linspace(0, len(f), limit + 1, dtype=int)
    ix = [i + int(np.argmax(y[i:j])) for i, j in zip(edges[:-1], edges[1:]) if j > i]
    return f[ix], y[ix]


def compress_wave(t, y, limit=1800):
    """Min/max envelope downsampling keeps transients visible."""
    if len(t) <= limit:
        return t, y
    edges = np.linspace(0, len(t), limit // 2 + 1, dtype=int)
    ix = []
    for i, j in zip(edges[:-1], edges[1:]):
        if j > i:
            ix.extend((i + int(np.argmin(y[i:j])), i + int(np.argmax(y[i:j]))))
    ix = np.array(sorted(set(ix)))
    return t[ix], y[ix]


def velocity_spectrum(f, acceleration_amplitude, unit, low_cutoff=2.0):
    """Amplitude of integrated acceleration spectrum, excluding near-DC bins."""
    f = np.asarray(f)
    a = np.asarray(acceleration_amplitude)
    factor = 9.80665 if unit == "g" else 1.0
    valid = f >= low_cutoff
    return f[valid], a[valid] * factor / (2 * np.pi * f[valid]) * 1000.0


def order_rules(rpm, upper):
    """Nominal shaft orders; not measured rotational speed."""
    points = [
        {"frequency": order * float(rpm) / 60.0, "order": f"{order}X"}
        for order in (1, 2, 3)
        if order * float(rpm) / 60.0 <= float(upper)
    ]
    base = alt.Chart(pd.DataFrame(points))
    rules = base.mark_rule(color="#e38b27", strokeDash=[5, 4], opacity=0.75).encode(
        x=alt.X("frequency:Q"),
        tooltip=[alt.Tooltip("order:N", title="회전 차수"), alt.Tooltip("frequency:Q", title="명목 Hz", format=".2f")],
    )
    return rules


def plot_lines(items, kind, upper, y_title, height=290, rpm=None, show_orders=False):
    chunks = []
    for label, data in items:
        if kind == "wave":
            f, a = compress_wave(data["time"], data["signal"])
        elif kind == "envelope":
            mask = data["envelope_frequency"] <= upper
            f, a = compress_max(data["envelope_frequency"][mask], data["envelope_amplitude"][mask])
        elif kind == "velocity":
            f, a = velocity_spectrum(data["frequency"], data["amplitude"], data["unit"])
            mask = f <= upper
            f, a = compress_max(f[mask], a[mask])
        else:
            mask = data["frequency"] <= upper
            f, a = compress_max(data["frequency"][mask], data["amplitude"][mask])
        chunks.append(pd.DataFrame({"x": f, "y": a, "Class": label}))
    frame = pd.concat(chunks, ignore_index=True)
    chart = alt.Chart(frame).mark_line(strokeWidth=1.25).encode(
        x=alt.X("x:Q", title="시간 (초)" if kind == "wave" else "주파수 (Hz)"),
        y=alt.Y("y:Q", title=y_title),
        color=alt.Color("Class:N", scale=alt.Scale(domain=list(CLASSES), range=PALETTE)),
        tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".5g")],
    ).properties(height=height).interactive()
    if show_orders and rpm is not None and kind != "wave":
        chart = chart + order_rules(rpm, upper)
    st.altair_chart(chart, use_container_width=True)



def render_feature_table(selections, loaded_labels):
    """Show existing EDA feature values, not new measurements or RF outputs."""
    fields = {
        "rms_ac": "RMS (AC, raw)",
        "peak_ac": "Peak (AC, raw)",
        "crest_factor_ac": "Crest Factor",
        "one_x_amplitude": "1X 진폭 (raw)",
        "two_x_amplitude": "2X 진폭 (raw)",
    }
    records = []
    for label, row in selections:
        if label not in loaded_labels:
            continue
        record = {"Class": label, "원본 조건": row["condition"], "파일": Path(str(row["path"])).name}
        for key, title in fields.items():
            val = pd.to_numeric(row.get(key), errors="coerce")
            record[title] = round(float(val), 6) if pd.notna(val) else None
        records.append(record)
    if records:
        table = pd.DataFrame(records)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button(
            "Class별 특징 비교표 CSV 다운로드",
            data=table.to_csv(index=False).encode("utf-8-sig"),
            file_name="isu_class_feature_comparison.csv",
            mime="text/csv",
            key="class_features_csv_download",
        )
        st.caption("EDA file_features.csv의 선택 파일별 특징값입니다. "
                   "RMS/Peak는 AC 기준이며 단위는 CH3 원본 단위입니다.")


def render_individual_charts(items, kind, upper, title, y_title, shared_scale, rpm=None, show_orders=False):
    """Small multiples; optional identical y-domain for honest magnitude comparison."""
    st.markdown("##### " + title)
    arrays = []
    for label, data in items:
        if kind == "wave":
            x, y = compress_wave(data["time"], data["signal"])
        elif kind == "envelope":
            mask = data["envelope_frequency"] <= upper
            x, y = compress_max(data["envelope_frequency"][mask], data["envelope_amplitude"][mask])
        else:
            mask = data["frequency"] <= upper
            x, y = compress_max(data["frequency"][mask], data["amplitude"][mask])
        arrays.append((label, x, y))
    min_y = min(float(np.min(y)) for _, _, y in arrays)
    max_y = max(float(np.max(y)) for _, _, y in arrays)
    padding = max((max_y - min_y) * 0.04, 1e-9)
    domain = [min_y - padding, max_y + padding]
    columns = st.columns(2)
    for i, (label, x, y) in enumerate(arrays):
        df = pd.DataFrame({"x": x, "y": y})
        with columns[i % 2]:
            st.markdown("**" + label + "**")
            y_axis = alt.Y("y:Q", title=y_title,
                           scale=alt.Scale(domain=domain) if shared_scale else alt.Scale(zero=False))
            chart = (alt.Chart(df).mark_line(color=PALETTE[list(CLASSES).index(label)],
                                              strokeWidth=1.1)
                     .encode(x=alt.X("x:Q", title="시간 (초)" if kind == "wave" else "주파수 (Hz)"),
                             y=y_axis,
                             tooltip=[alt.Tooltip("x:Q", format=".2f"),
                                      alt.Tooltip("y:Q", format=".5g")])
                     .properties(height=230))
            if show_orders and rpm is not None and kind != "wave":
                chart = chart + order_rules(rpm, upper)
            st.altair_chart(chart, use_container_width=True)


def _report_font():
    """Use an installed Korean font without bundling or modifying font files."""
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        if name in available:
            return name
    return "DejaVu Sans"


def report_png(items, mode, upper, env_upper, title, show_orders=True,
               class_rpm=None, individual=False, metadata=None):
    """High-resolution PNG from the selected MAT signals; no model decisions."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.font_manager import FontProperties

    font = FontProperties(family=_report_font())
    colors = ({900: "#2475db", 1200: "#e38b27", 1800: "#168e7a"}
              if mode == "rpm" else dict(zip(CLASSES, PALETTE)))
    rows = []
    for entry in items:
        if mode == "rpm":
            rpm, data, row = entry
            label = f"{rpm} RPM"
            condition = str(row["condition"])
            filename = Path(str(row["path"])).name
            color = colors[rpm]
        else:
            label, data = entry
            rpm = class_rpm
            info = (metadata or {}).get(label, {})
            condition = str(info.get("condition", label))
            filename = Path(str(info.get("path", ""))).name
            color = colors[label]
        rows.append((rpm, label, data, color, condition, filename))

    if individual and mode == "class":
        fig = Figure(figsize=(13.8, 16.5), dpi=120, facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(4, 3, squeeze=False)
        fig.suptitle(title, fontsize=17, fontproperties=font, y=.994)
        for row_index, (rpm, label, data, color, condition, filename) in enumerate(rows):
            plots = (
                ("wave", 0, "시간 영역 파형", "CH3 진폭 (raw)"),
                ("fft", upper, "FFT 주파수 스펙트럼", "FFT 진폭 (raw)"),
                ("envelope", env_upper, "포락 주파수 스펙트럼", "포락 FFT 진폭 (raw)"),
            )
            for col_index, (kind, limit, chart_title, ylabel) in enumerate(plots):
                ax = axes[row_index][col_index]
                _report_draw(ax, data, kind, limit, color)
                if show_orders and kind != "wave":
                    _report_orders(ax, rpm, limit, "#d98a3b")
                ax.set_title(f"{label} · {condition} | {chart_title}", fontsize=10, fontproperties=font)
                ax.set_xlabel("시간 (초)" if kind == "wave" else "주파수 (Hz)", fontproperties=font)
                ax.set_ylabel(ylabel, fontproperties=font)
                ax.grid(alpha=.18)
                ax.tick_params(labelsize=8)
        fig.text(.5, .037,
                 "선택 파일별 탐색적 비교 · raw 진폭(물리 단위 미확인) · 명목 RPM · 확정 진단 아님",
                 ha="center", fontsize=9, fontproperties=font, color="#555555")
        fig.subplots_adjust(top=.955, bottom=.075, left=.07, right=.985, hspace=.58, wspace=.36)
    else:
        fig = Figure(figsize=(12.8, 10.5), dpi=120, facecolor="white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(3, 1)
        for rpm, label, data, color, condition, filename in rows:
            for ax, kind, limit in (
                (axes[0], "wave", 0),
                (axes[1], "fft", upper),
                (axes[2], "envelope", env_upper),
            ):
                _report_draw(ax, data, kind, limit, color, label=label)
            if show_orders and mode == "rpm":
                for ax, limit in ((axes[1], upper), (axes[2], env_upper)):
                    _report_orders(ax, rpm, limit, color, count=1)
        if show_orders and mode == "class":
            for ax, limit in ((axes[1], upper), (axes[2], env_upper)):
                _report_orders(ax, class_rpm, limit, "#d98a3b")
        settings = (
            ("CH3 시간 영역 파형", "시간 (초)", "CH3 진폭 (raw)"),
            ("Hann 단측 FFT 주파수 스펙트럼", "주파수 (Hz)", "FFT 진폭 (raw)"),
            ("포락 주파수 스펙트럼 (2–8kHz)", "주파수 (Hz)", "포락 FFT 진폭 (raw)"),
        )
        for ax, (heading, xlabel, ylabel) in zip(axes, settings):
            ax.set_title(heading, fontsize=12, fontproperties=font)
            ax.set_xlabel(xlabel, fontproperties=font)
            ax.set_ylabel(ylabel, fontproperties=font)
            ax.grid(alpha=.2)
            ax.legend(loc="upper right", ncol=min(len(rows), 4), fontsize=9)
        if mode == "class":
            info_text = " / ".join(f"{label}: {condition}" for _, label, _, _, condition, _ in rows)
        else:
            info_text = " / ".join(f"{rpm}RPM: {condition}" for rpm, _, _, _, condition, _ in rows)
        fig.suptitle(title, fontsize=16, fontproperties=font, y=.993)
        fig.text(.5, .046, info_text, ha="center", fontsize=9, fontproperties=font)
        fig.text(.5, .022,
                 "선택한 실제 MAT의 탐색적 비교 · raw 진폭(물리 단위 미확인) · 명목 RPM · 확정 진단 아님",
                 ha="center", fontsize=8, fontproperties=font, color="#555555")
        fig.subplots_adjust(top=.93, bottom=.105, left=.10, right=.97, hspace=.55)
    output = BytesIO()
    fig.savefig(output, format="png", dpi=220, facecolor="white")
    return output.getvalue()


def _report_draw(ax, data, kind, upper, color, label=None):
    if kind == "wave":
        x, y = compress_wave(data["time"], data["signal"], limit=2400)
    else:
        prefix = "envelope_" if kind == "envelope" else ""
        f = data[prefix + "frequency"]
        a = data[prefix + "amplitude"]
        mask = (f >= 0) & (f <= upper)
        x, y = compress_max(f[mask], a[mask], limit=1800)
        ax.set_xlim(0, upper)
    ax.plot(x, y, lw=.8, color=color, alpha=.9, label=label)


def _report_orders(ax, rpm, upper, color, count=3):
    if rpm is None:
        return
    for order in range(1, count + 1):
        frequency = float(rpm) * order / 60.0
        if frequency <= upper:
            ax.axvline(frequency, ls="--", lw=.9, color=color, alpha=.65)

def render_rpm_comparison(frame):
    st.markdown("### 동일 조건 · RPM별 신호 비교")
    st.caption("동일한 원본 condition과 측정 날짜를 선택해 900·1200·1800RPM 파일을 비교합니다. 각 RPM의 파일은 서로 다른 측정 기록입니다.")
    conditions = sorted(set(frame["condition"].dropna()))
    condition = st.selectbox("원본 측정 조건", conditions, key="rpm_compare_condition")
    subset = frame[frame["condition"] == condition].copy()
    all_dates = sorted(set(subset["date"].dropna()))
    date = st.selectbox("측정 날짜", all_dates, key="rpm_compare_date")
    subset = subset[subset["date"] == date]
    root = st.text_input("MAT 원본 폴더 (기록 경로가 없을 때)", "", key="rpm_compare_root")
    selected = []
    columns = st.columns(3)
    for col, rpm in zip(columns, (900, 1200, 1800)):
        with col:
            rows = subset[subset["rpm"] == rpm].sort_values("path")
            if rows.empty:
                st.warning(f"{rpm}RPM: 선택 날짜에 파일 없음")
                continue
            idx = st.selectbox(
                f"{rpm}RPM 실제 파일", list(rows.index),
                format_func=lambda i: Path(str(rows.loc[i, "path"])).name,
                key=f"rpm_compare_pick_{rpm}"
            )
            selected.append((rpm, rows.loc[idx]))
    if st.button("RPM별 실제 MAT 비교", type="primary", key="rpm_compare_run"):
        st.session_state["rpm_compare_ready"] = True
    if not st.session_state.get("rpm_compare_ready", False):
        st.info("비교할 파일을 고른 뒤 실행해 주세요.")
        return
    items = []
    missing = []
    for rpm, row in selected:
        path = locate_mat(row["path"], root)
        if path is None:
            missing.append(f"{rpm}RPM: {row['path']}")
            continue
        try:
            stat = path.stat()
            data = calculate(str(path), stat.st_size, stat.st_mtime_ns, rpm)
            items.append((rpm, data, row))
        except Exception as exc:
            st.warning(f"{rpm}RPM 분석 실패: {exc}")
    if missing:
        st.warning("일부 원본 MAT 파일을 찾지 못했습니다.")
        with st.expander("누락 파일 경로"):
            st.code("\\n".join(missing))
    if len(items) < 2:
        st.info("서로 다른 RPM의 실제 MAT 파일이 최소 2개 필요합니다.")
        return

    records = []
    for rpm, data, row in items:
        records.append({
            "RPM": rpm, "명목 1X (Hz)": rpm / 60,
            "명목 2X (Hz)": rpm / 30,
            "원본 조건": row["condition"],
            "RMS (AC, raw)": row.get("rms_ac"),
            "Peak (AC, raw)": row.get("peak_ac"),
            "Crest Factor": row.get("crest_factor_ac"),
            "1X 진폭 (raw)": row.get("one_x_amplitude"),
            "2X 진폭 (raw)": row.get("two_x_amplitude"),
            "파일": Path(str(row["path"])).name
        })
    st.markdown("#### RPM별 특징 비교")
    rpm_table = pd.DataFrame(records)
    st.dataframe(rpm_table, hide_index=True, use_container_width=True)
    st.download_button(
        "RPM별 특징 비교표 CSV 다운로드",
        data=rpm_table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"isu_rpm_comparison_{condition}_{date}.csv",
        mime="text/csv",
        key="rpm_features_csv_download",
    )
    report = {
        "report_type": "exploratory_rpm_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "date": date,
        "source": "EDA file_features.csv and selected MAT signals",
        "rows": rpm_table.where(pd.notna(rpm_table), None).to_dict(orient="records"),
        "limitations": [
            "One selected file per RPM; not an estimate of population performance.",
            "Nominal RPM is not measured shaft speed.",
            "CH3 amplitudes are raw and physical calibration is unverified.",
            "Not a fault diagnosis or an AE/RF performance evaluation.",
        ],
    }
    st.download_button(
        "RPM별 비교 기록 JSON 다운로드",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name=f"isu_rpm_comparison_{condition}_{date}.json",
        mime="application/json",
        key="rpm_features_json_download",
    )
    st.caption("RMS·Peak·1X·2X는 EDA 특징 CSV의 값입니다. 실제 계측 RPM이 아닌 명목 RPM을 사용합니다.")

    st.markdown("#### 주파수 스펙트럼 · 1X 피크 이동 비교")
    upper = st.radio("주파수 표시 범위", [100, 200, 1000], horizontal=True, key="rpm_compare_upper")
    colors = {900: "#2475db", 1200: "#e38b27", 1800: "#168e7a"}
    pieces = []
    rules = []
    for rpm, data, _ in items:
        mask = data["frequency"] <= upper
        f, a = compress_max(data["frequency"][mask], data["amplitude"][mask])
        pieces.append(pd.DataFrame({"주파수": f, "진폭": a, "RPM": f"{rpm}RPM"}))
        rules.append({"주파수": rpm / 60, "RPM": f"{rpm}RPM"})
    chart = alt.Chart(pd.concat(pieces, ignore_index=True)).mark_line(strokeWidth=1.4).encode(
        x=alt.X("주파수:Q", title="주파수 (Hz)", scale=alt.Scale(domain=[0, upper])),
        y=alt.Y("진폭:Q", title="FFT 진폭 (raw)"),
        color=alt.Color("RPM:N", scale=alt.Scale(
            domain=["900RPM", "1200RPM", "1800RPM"],
            range=[colors[900], colors[1200], colors[1800]]
        )),
        tooltip=["RPM:N", alt.Tooltip("주파수:Q", format=".2f"), alt.Tooltip("진폭:Q", format=".5g")]
    ).properties(height=330)
    rules_chart = alt.Chart(pd.DataFrame(rules)).mark_rule(strokeDash=[5, 4], opacity=.65).encode(
        x="주파수:Q", color=alt.Color("RPM:N", scale=alt.Scale(
            domain=["900RPM", "1200RPM", "1800RPM"],
            range=[colors[900], colors[1200], colors[1800]]
        ))
    )
    st.altair_chart(chart + rules_chart, use_container_width=True)
    st.caption("점선은 RPM별 명목 1X(900:15Hz, 1200:20Hz, 1800:30Hz)입니다. 측정된 피크와 일치한다고 가정하지 않습니다.")

    st.markdown("#### 포락 주파수 스펙트럼 · RPM별 비교")
    env_pieces = []
    for rpm, data, _ in items:
        mask = data["envelope_frequency"] <= upper
        f, a = compress_max(data["envelope_frequency"][mask], data["envelope_amplitude"][mask])
        env_pieces.append(pd.DataFrame({"주파수": f, "진폭": a, "RPM": f"{rpm}RPM"}))
    env_chart = alt.Chart(pd.concat(env_pieces, ignore_index=True)).mark_line(strokeWidth=1.4).encode(
        x=alt.X("주파수:Q", title="주파수 (Hz)", scale=alt.Scale(domain=[0, upper])),
        y=alt.Y("진폭:Q", title="포락 FFT 진폭 (raw)"),
        color=alt.Color("RPM:N", scale=alt.Scale(
            domain=["900RPM", "1200RPM", "1800RPM"],
            range=[colors[900], colors[1200], colors[1800]]
        )),
        tooltip=["RPM:N", alt.Tooltip("주파수:Q", format=".2f"), alt.Tooltip("진폭:Q", format=".5g")]
    ).properties(height=330)
    st.altair_chart(env_chart + rules_chart, use_container_width=True)
    st.caption("선택한 동일 조건의 각 RPM 대표 파일을 비교한 탐색적 결과입니다. Class 전체 특성이나 독립 진단 성능을 의미하지 않습니다.")
    st.markdown("#### 보고서용 그래프 PNG 저장")
    st.caption("선택된 실제 MAT 신호의 시간파형·FFT·포락 스펙트럼을 한 장의 고해상도 PNG로 저장합니다.")
    try:
        png = report_png(items, "rpm", upper, upper,
                         f"회전속도별 진동 신호 비교 | {condition} | {date}")
        st.download_button(
            "RPM별 비교 그래프 PNG 다운로드",
            data=png, file_name=f"isu_rpm_graph_{condition}_{date}.png",
            mime="image/png", key="rpm_graph_png_download",
        )
    except Exception as exc:
        st.warning(f"PNG 생성 실패: {exc}")



def render_class_comparison():
    st.subheader("Class별 신호 비교 · 실제 MAT")
    st.caption("원본 데이터의 condition 레이블 기준 비교입니다. RF 예측이나 확정 진단 결과가 아닙니다.")
    eda_dir = Path(st.text_input("EDA 특징 데이터 폴더", str(DEFAULT_EDA), key="class_eda_dir")).expanduser()
    csv_path = eda_dir / "file_features.csv"
    if not csv_path.is_file():
        st.warning("file_features.csv를 찾지 못했습니다. EDA 폴더 경로를 확인하세요.")
        return
    try:
        frame = load_index(str(csv_path), csv_path.stat().st_mtime_ns)
    except Exception as exc:
        st.error("EDA 목록을 읽을 수 없습니다: " + str(exc))
        return
    mode = st.radio(
        "비교 방식", ["Class별 비교 (동일 RPM)", "RPM별 비교 (동일 조건)"],
        horizontal=True, key="class_comparison_mode"
    )
    if mode.startswith("RPM별"):
        render_rpm_comparison(frame)
        return
    rpm = st.selectbox("비교 RPM", [900, 1200, 1800], key="class_rpm")
    dates = sorted(frame.loc[frame["rpm"] == rpm, "date"].dropna().unique().tolist())
    if not dates:
        st.warning("해당 RPM의 기록이 없습니다.")
        return
    date = st.selectbox("비교 날짜 (동일 날짜·RPM 권장)", dates, key="class_date")
    mat_root = st.text_input("MAT 원본 폴더 (기록된 경로가 없을 때만 입력)", "", key="class_mat_root")
    st.caption("동일 RPM·날짜에서 Class별 실제 파일 하나씩 선택합니다. 파일명이 중복되는 경우 원본 경로를 확인하세요.")
    subset = frame[(frame["rpm"] == rpm) & (frame["date"] == date)]
    selections = []
    cols = st.columns(4)
    for col, label in zip(cols, CLASSES):
        with col:
            rows = subset[subset["class"] == label].sort_values(["condition", "path"])
            if rows.empty:
                st.info(label + ": 해당 날짜 파일 없음")
                continue
            options = list(rows.index)

            option_labels = {
                idx: (
                    str(rows.loc[idx, "condition"])
                    + " · "
                    + Path(str(rows.loc[idx, "path"])).name
                )
                for idx in options
            }

            choice = st.selectbox(
                label,
                options,
                format_func=lambda idx, labels=option_labels: labels.get(
                    idx, str(idx)
                ),
                key="class_pick_" + label,
            )

            if choice not in rows.index:
                st.warning(label + ": 파일을 다시 선택해 주세요.")
                continue

            selections.append((label, rows.loc[choice]))
    if len(selections) < 2:
        st.info("비교할 Class가 두 개 이상 있어야 합니다.")
        return
    if st.button("선택한 실제 MAT 신호 비교", type="primary", key="class_compare"):
        st.session_state["class_comparison_requested"] = True
    if not st.session_state.get("class_comparison_requested", False):
        st.info("파일을 선택한 뒤 비교 버튼을 눌러 주세요.")
        return
    items = []
    missing = []
    for label, row in selections:
        mat_path = locate_mat(row["path"], mat_root)
        if mat_path is None:
            missing.append(label + ": " + str(row["path"]))
            continue
        try:
            stat = mat_path.stat()
            data = calculate(str(mat_path), stat.st_size, stat.st_mtime_ns, rpm)
            items.append((label, data))
        except Exception as exc:
            st.warning(label + " MAT 분석 불가: " + str(exc))
    if missing:
        st.warning("원본 MAT를 찾지 못했습니다. MAT 원본 폴더를 지정하거나 EDA의 path를 확인하세요.")
        with st.expander("찾지 못한 파일 경로"):
            st.code("\\n".join(missing))
    if len(items) < 2:
        st.info("실제 MAT를 읽은 Class가 두 개 미만이어서 비교를 표시하지 않습니다.")
        return
    st.success("실제 MAT 신호 로드: " + ", ".join(label for label, _ in items))
    st.markdown("#### 회전 주파수 기준선")
    show_orders = st.checkbox("FFT·포락 그래프에 1X·2X·3X 표시", value=True, key="class_show_orders")
    base_hz = rpm / 60.0
    st.caption(f"선택한 {rpm}RPM의 명목 기준: 1X {base_hz:g}Hz · 2X {2*base_hz:g}Hz · 3X {3*base_hz:g}Hz. 실제 측정 회전속도는 아닙니다.")
    st.markdown("#### ① CH3 시간 영역 파형")
    plot_lines(items, "wave", 0, "CH3 진폭 (raw)")
    st.markdown("#### ② CH3 주파수 스펙트럼 · Hann 단측 FFT")
    upper = st.radio("스펙트럼 범위", [200, 1000, 10000], horizontal=True, key="class_fft_upper")
    plot_lines(items, "fft", upper, "FFT 진폭 (raw)", rpm=rpm, show_orders=show_orders)
    st.markdown("#### ③ 포락 주파수 스펙트럼 · 2–8kHz 대역")
    env_upper = st.radio("포락 스펙트럼 범위", [200, 1000], horizontal=True, key="class_env_upper")
    plot_lines(items, "envelope", env_upper, "포락 FFT 진폭 (raw)", rpm=rpm, show_orders=show_orders)
    st.caption("기존 EDA 계산 사용: 4차 Butterworth 2–8kHz, 양방향 필터, Hilbert, 양끝 0.1초 제외.")
    st.markdown("#### ④ 속도 주파수 스펙트럼 · 단위 검증 필요")
    confirmed = st.checkbox(
        "CH3가 교정된 가속도 신호이며 아래 단위가 맞음을 확인했습니다",
        value=False, key="class_accel_confirm"
    )
    if confirmed:
        unit = st.selectbox("원본 가속도 단위", ["m/s²", "g"], key="class_accel_unit")
        calibrated = [(label, {**data, "unit": unit}) for label, data in items]
        plot_lines(calibrated, "velocity", upper, "속도 진폭 (mm/s)")
        st.caption("주파수 영역에서 a/(2πf) 적분, 2Hz 미만 제외. DC/저주파 드리프트 및 센서 교정 확인 필요.")
    else:
        st.info("현재 CH3는 raw 진폭입니다. 가속도 단위와 센서 감도가 확인되기 전에는 속도(mm/s)를 계산·표시하지 않습니다.")
    st.divider()
    st.markdown("#### ⑤ Class별 진동 특징 비교표")
    render_feature_table(selections, {label for label, _ in items})
    class_records = []
    for label, row in selections:
        if label in {name for name, _ in items}:
            class_records.append({
                "class": label, "condition": str(row["condition"]),
                "file": str(row["path"]), "rpm": rpm, "date": date,
                **{key: (float(row[key]) if pd.notna(row.get(key)) else None)
                   for key in ("rms_ac", "peak_ac", "crest_factor_ac", "one_x_amplitude", "two_x_amplitude")}
            })
    report = {
        "report_type": "exploratory_class_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "EDA file_features.csv and selected MAT signals",
        "rows": class_records,
        "limitations": [
            "One selected file per class; not representative of the whole class.",
            "Class is original measurement condition, not an RF prediction.",
            "CH3 amplitudes are raw and physical calibration is unverified.",
            "Not a fault diagnosis or an AE/RF performance evaluation.",
        ],
    }
    st.download_button(
        "Class별 비교 기록 JSON 다운로드",
        data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
        file_name=f"isu_class_comparison_{rpm}rpm_{date}.json",
        mime="application/json",
        key="class_features_json_download",
    )

    st.markdown("#### ⑥ Class별 개별 그래프")
    st.caption("중첩 그래프에서 작은 신호가 가려지는 문제를 줄이기 위해 Class별로 분리해 표시합니다.")
    axis_mode = st.radio(
        "Y축 스케일", ["개별 스케일 (형태 비교)", "동일 스케일 (크기 비교)"],
        horizontal=True, key="class_axis_mode"
    )
    shared_scale = axis_mode.startswith("동일")
    render_individual_charts(items, "wave", 0, "CH3 시간 영역 파형", "CH3 진폭 (raw)", shared_scale)
    render_individual_charts(items, "fft", upper, "FFT 주파수 스펙트럼", "FFT 진폭 (raw)", shared_scale, rpm=rpm, show_orders=show_orders)
    render_individual_charts(items, "envelope", env_upper, "포락 주파수 스펙트럼", "포락 FFT 진폭 (raw)", shared_scale, rpm=rpm, show_orders=show_orders)
    st.caption("개별 스케일은 파형의 형태 비교용입니다. Class 사이 진폭 크기는 동일 스케일에서 비교하세요.")
    st.caption("파일별 신호의 탐색적 비교입니다. 한 파일의 차이를 Class 전체 특성이나 독립 검증 성능으로 일반화하지 마세요.")
    st.markdown("#### ⑦ 보고서용 그래프 PNG 저장")
    st.caption("선택된 실제 MAT 신호의 시간파형·FFT·포락 스펙트럼을 한 장의 고해상도 PNG로 저장합니다.")
    try:
        png = report_png(items, "class", upper, env_upper,
                         f"결함 유형별 진동 신호 비교 | {rpm} RPM | {date}",
                         show_orders=show_orders, class_rpm=rpm,
                         metadata={label: row for label, row in selections})
        st.download_button(
            "Class별 비교 그래프 PNG 다운로드",
            data=png, file_name=f"isu_class_graph_{rpm}rpm_{date}.png",
            mime="image/png", key="class_graph_png_download",
        )
    except Exception as exc:
        st.warning(f"PNG 생성 실패: {exc}")



    st.markdown("#### ⑧ Class별 개별 그래프 PNG 저장")
    st.caption("Class별로 Y축을 독립적으로 표시한 4×3 그래프입니다. 진폭 크기를 비교할 때는 중첩 그래프와 특징표를 함께 확인하세요.")
    try:
        png_individual = report_png(
            items, "class", upper, env_upper,
            f"결함 유형별 개별 신호 비교 | {rpm} RPM | {date}",
            show_orders=show_orders, class_rpm=rpm, individual=True,
            metadata={label: row for label, row in selections},
        )
        st.download_button(
            "Class별 개별 그래프 PNG 다운로드",
            data=png_individual,
            file_name=f"isu_class_individual_{rpm}rpm_{date}.png",
            mime="image/png", key="class_individual_graph_png_download",
        )
    except Exception as exc:
        st.warning(f"개별 그래프 PNG 생성 실패: {exc}")
