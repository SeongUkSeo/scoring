# app.py
import json, math, hashlib, io
from math import sqrt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from distribution import *

def is_cover(topic, source, config):
    return bool(config.get(topic, {}).get(source, {}).get("Cover", False))

def set_noncover_to_none(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty or "Topic" not in df.columns:
        return df
    out = df.copy()
    sources = [c for c in out.columns if c != "Topic"]
    for i, row in out.iterrows():
        t = row["Topic"]
        if t == "Others":
            continue
        for s in sources:
            if not is_cover(t, s, config):
                out.at[i, s] = None  # 업로드 시점에만 None으로 치환
    return out

def enforce_noncover_lock(edited_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    prev = st.session_state.input_df.copy() if "input_df" in st.session_state and not st.session_state.input_df.empty else edited_df
    out = edited_df.copy()
    if "Topic" not in out.columns: 
        return out
    sources = [c for c in out.columns if c != "Topic"]
    for i, row in out.iterrows():
        topic = row["Topic"]
        if topic == "Others":
            continue
        for s in sources:
            cover = bool(config.get(topic, {}).get(s, {}).get("Cover", False))
            if not cover:
                # 되돌리기
                try:
                    out.at[i, s] = prev.loc[prev["Topic"]==topic, s].iloc[0]
                except Exception:
                    pass
    return out

def get_effective_input(topic, source, x_covered, others_value, cfg, noncover_counts, ftw_sums):
    cell = cfg.get(topic, {}).get(source, {})
    cover = bool(cell.get("Cover", False))
    if cover and not np.isnan(x_covered):
        return x_covered, None  # 실제값 사용

    # Fallback
    if source in SOURCES_WITH_OTHERS:
        fsw = float(cell.get("FallbackSourceWeight", 0.0))
        ftw = float(cell.get("FallbackTopicWeight", 0.0))
        if ftw_sums[source] > 0:
            share = ftw / ftw_sums[source]
        else:
            share = 1.0 / max(1, noncover_counts[source])  # 균등분배
        return float(others_value) * fsw * share, None

    # Others 없는 소스(CEExercise, NowBarSports) → Cover=False면 C(0.5) 강제
    return None, 0.5  # 강제 상수 퍼센타일

def percentile_for_cell(x_eff, dist, params, const_p=None):
    # 강제 상수(C(0.5) 등) 우선 적용
    if const_p is not None:
        return float(const_p)

    d = dist.upper()
    if d.startswith("LN"):
        median = params.get("median", 1.0)
        p90    = params.get("p90",    max(1.01, median*2))
        x = 0.0 if x_eff is None else x_eff
        return float(cdf_lognorm(x, median, p90))

    if d.startswith("ZIP"):
        mean = params.get("mean", 0.0)
        p0   = params.get("p0",   0.0)
        x = 0.0 if x_eff is None else x_eff
        return float(zip_cdf(x, mean, p0))

    if d.startswith("ZINB"):
        mean = params.get("mean", 0.0)
        var  = params.get("var",  max(mean+1e-6, mean*1.5))
        p0   = params.get("p0",   0.0)
        x = 0.0 if x_eff is None else x_eff
        return float(zinb_cdf(x, mean, var, p0))

    if d.startswith("C"):
        p = params.get("p", 0.5)
        return float(p)

    return 0.0

def compute_noncover_stats(config, topics, sources):
    noncover_counts = {s:0 for s in sources}
    ftw_sums = {s:0.0 for s in sources}
    for s in sources:
        for t in topics:
            cover = bool(config.get(t, {}).get(s, {}).get("Cover", False))
            if not cover:
                noncover_counts[s] += 1
                ftw_sums[s] += float(config.get(t, {}).get(s, {}).get("FallbackTopicWeight", 0.0))
    return noncover_counts, ftw_sums

def compute_percentiles_and_scores(df_in, config):
    topics = [t for t in df_in["Topic"].tolist() if t != "Others"]
    sources = [c for c in df_in.columns if c!="Topic"]
    others_row = df_in[df_in["Topic"]=="Others"]
    others = {s: float(others_row[s].iloc[0]) if not others_row.empty and s in others_row else 0.0 for s in sources}
    noncover_counts, ftw_sums = compute_noncover_stats(config, topics, sources)

    pct = []
    for t in topics:
        row = {"Topic": t}
        for s in sources:
            x_covered = safe_float(df_in.loc[df_in["Topic"]==t, s].iloc[0]) if s in df_in.columns else np.nan
            cell_cfg  = config.get(t, {}).get(s, {})
            dist      = cell_cfg.get("Distribution", "C")
            params    = cell_cfg.get("Parameter", {})
            x_eff, const_p = get_effective_input(
                t, s, x_covered, others.get(s,0.0),
                config, noncover_counts, ftw_sums
            )
            row[s] = percentile_for_cell(x_eff, dist, params, const_p)
        pct.append(row)
    pct_df = pd.DataFrame(pct)
    pct_df["Score_sum"] = pct_df.drop(columns=["Topic"]).sum(axis=1)
    return pct_df, topics, sources

def make_stacked_bar(pct_df, sources):
    long = pct_df.melt(id_vars=["Topic","Score_sum"], value_vars=sources,
                       var_name="Source", value_name="Percentile")
    long = long.sort_values(["Score_sum"], ascending=[False])

    fig = px.bar(long, x="Percentile", y="Topic", color="Source",
                 orientation="h", barmode="stack", height=700)

    # 각 소스 segment에 값 표시
    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="inside",
        cliponaxis=False
    )

    # 우측 끝에 최종 합계 라벨
    totals = pct_df.sort_values("Score_sum", ascending=False)
    fig.add_trace(
        go.Scatter(
            x=totals["Score_sum"],
            y=totals["Topic"],
            mode="text",
            text=totals["Score_sum"].round(2).astype(str),
            textposition="middle right",
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=60, t=10, b=10)  # 오른쪽 여백 확보
    )
    return fig

def make_source_plots(topic, source, df_in, config):
    cell = config.get(topic, {}).get(source, {})
    dist = cell.get("Distribution","C").upper()
    params = cell.get("Parameter", {})

    others = float(df_in.loc[df_in["Topic"]=="Others", source].iloc[0]) if source in df_in.columns else 0.0
    x_raw = df_in.loc[df_in["Topic"]==topic, source].iloc[0] if source in df_in.columns else np.nan

    topics_all = [t for t in df_in["Topic"].tolist() if t != "Others"]
    noncover_counts, ftw_sums = compute_noncover_stats(config, topics_all, [source])

    x_eff, const_p = get_effective_input(
        topic, source, safe_float(x_raw), others, config, noncover_counts, ftw_sums
    )

    # --- 이하 동일하되 vline 계산만 x_eff로 ---
    if dist.startswith("LN"):
        median = params.get("median", 1.0); p90 = params.get("p90", max(1.01, median*2))
        xs = np.linspace(1e-6, max(1.0, (p90*4)), 300)
        pdf = [pdf_lognorm(x, median, p90) for x in xs]
        cdf = [cdf_lognorm(x, median, p90) for x in xs]
        vline = float(0.0 if x_eff is None else x_eff)
        fig_pdf = go.Figure([go.Scatter(x=xs, y=pdf, mode="lines")])
        fig_cdf = go.Figure([go.Scatter(x=xs, y=cdf, mode="lines")])
        for f in (fig_pdf, fig_cdf):
            f.add_vline(x=vline, line_dash="dot")
            f.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        return fig_pdf, fig_cdf, vline, const_p

    if dist.startswith("ZIP"):
        mean = params.get("mean", 0.0); p0 = params.get("p0", 0.0)
        lam = mean/max(1e-9, 1-p0) if p0<1.0 else 0.0
        xmax = int(max(10, math.ceil(lam*5 + 10)))
        xs = list(range(0, xmax+1))
        pmf = zip_pmf_grid(xs, mean, p0)
        cdf_vals = np.cumsum(pmf)
        vline = int(0 if x_eff is None else round(x_eff))
        fig_pdf = go.Figure([go.Bar(x=xs, y=pmf)])
        fig_cdf = go.Figure([go.Scatter(x=xs, y=cdf_vals, mode="lines+markers")])
        for f in (fig_pdf, fig_cdf):
            f.add_vline(x=vline, line_dash="dot")
            f.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        return fig_pdf, fig_cdf, vline, const_p

    if dist.startswith("ZINB"):
        mean = params.get("mean", 0.0); var = params.get("var", max(mean+1e-6, mean*1.5)); p0 = params.get("p0", 0.0)
        r,p = nb_params_from_mean_var(mean, var)
        mu_nb = r*(1-p)/p
        xmax = int(max(10, math.ceil(mu_nb + 7*sqrt(mu_nb + mu_nb**2/max(r,1e-6)))))
        xmax = min(xmax, 500)
        xs = list(range(0, xmax+1))
        pmf = zinb_pmf_grid(xs, mean, var, p0)
        cdf_vals = np.cumsum(pmf)
        vline = int(0 if x_eff is None else round(x_eff))
        fig_pdf = go.Figure([go.Bar(x=xs, y=pmf)])
        fig_cdf = go.Figure([go.Scatter(x=xs, y=cdf_vals, mode="lines+markers")])
        for f in (fig_pdf, fig_cdf):
            f.add_vline(x=vline, line_dash="dot")
            f.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
        return fig_pdf, fig_cdf, vline, const_p

    # Constant
    p = params.get("p", 0.5)
    xs = np.linspace(0,1,101)
    pdf = np.zeros_like(xs)
    cdf = np.where(xs < p, 0.0, 1.0)
    fig_pdf = go.Figure([go.Scatter(x=xs, y=pdf, mode="lines")])
    fig_cdf = go.Figure([go.Scatter(x=xs, y=cdf, mode="lines")])
    vline = float(0.0 if x_eff is None else x_eff)
    for f in (fig_pdf, fig_cdf):
        f.add_vline(x=vline, line_dash="dot")
        f.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
    return fig_pdf, fig_cdf, vline, const_p


with open("cover.json", "r") as f:
    cover_settings = json.load(f)
target_sources = list(cover_settings.keys())
target_topics = list(cover_settings[target_sources[0]].keys())

# ---------- 세션 초기화 ----------
st.set_page_config(page_title="Scoring Dashboard", layout="wide")
if "input_df" not in st.session_state:
    st.session_state.input_df = pd.DataFrame(columns=target_sources, index=target_topics+['Others'])
if "config" not in st.session_state:
    st.session_state.config = {}
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

# ---------- 상단: 업로드 + 인풋 테이블(접이식) ----------
with st.expander("Input / Config (편집 가능)", expanded=True):
    c1, c2 = st.columns(2)

    if "last_csv_hash" not in st.session_state: st.session_state.last_csv_hash = None
    if "last_json_hash" not in st.session_state: st.session_state.last_json_hash = None
    if "mask_applied"   not in st.session_state: st.session_state.mask_applied = False

    with c1:
        up_csv = st.file_uploader("aggregation.csv 업로드", type=["csv"])
        if up_csv is not None:
            csv_bytes = up_csv.getvalue()
            h = hashlib.md5(csv_bytes).hexdigest()
            if h != st.session_state.last_csv_hash:
                st.session_state.input_df = pd.read_csv(io.BytesIO(csv_bytes))
                st.session_state.last_csv_hash = h
                st.session_state.mask_applied = False  # 새 업로드 → 다시 마스킹 예정

    with c2:
        up_json = st.file_uploader("config.json 업로드", type=["json"])
        if up_json is not None:
            json_bytes = up_json.getvalue()
            h = hashlib.md5(json_bytes).hexdigest()
            if h != st.session_state.last_json_hash:
                st.session_state.config = json.loads(json_bytes.decode("utf-8"))
                st.session_state.last_json_hash = h
                st.session_state.mask_applied = False  # 새 설정 → 다시 마스킹 예정

    # 두 파일 모두 있을 때, 업로드 이벤트에 한해 non-cover를 None으로 1회 치환
    if (not st.session_state.input_df.empty) and st.session_state.config and not st.session_state.mask_applied:
        st.session_state.input_df = set_noncover_to_none(st.session_state.input_df, st.session_state.config)
        st.session_state.mask_applied = True

    st.caption("※ LAST ROW=Others, 첫 컬럼 이름은 'Topic' 가정")
    edited = st.data_editor(
        st.session_state.input_df if not st.session_state.input_df.empty else pd.DataFrame({"Topic":["Others"]}),
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor_input"
    )
    # 시스템적으로 블록하지 않음: 사용자가 non-cover 셀을 수정해도 그대로 둠
    st.session_state.input_df = edited

# 데이터/설정 체크
if st.session_state.input_df.sum().sum()==0 or not st.session_state.config:
    st.info("aggregation.csv와 config.json을 업로드하세요.")
    st.stop()

# ---------- 계산 ----------
pct_df, topics, sources = compute_percentiles_and_scores(st.session_state.input_df, st.session_state.config)
if not topics:
    st.warning("토픽 행이 없습니다.")
    st.stop()

# ---------- 레이아웃: 좌(랭킹), 우(세부/편집) ----------
left, right = st.columns([0.55, 0.45], gap="large")

with left:
    fig = make_stacked_bar(pct_df, sources)
    st.plotly_chart(fig, use_container_width=True, key="rank_bar")

    topics_list = pct_df["Topic"].tolist()
    default_topic = pct_df.sort_values("Score_sum", ascending=False)["Topic"].iloc[0]

    # 기존 선택값이 None 이거나 목록에 없으면 기본값으로
    current = st.session_state.get("selected_topic") or default_topic
    if current not in topics_list:
        current = default_topic

    sel = st.selectbox("선택 토픽", options=topics_list,
                       index=topics_list.index(current))
    st.session_state.selected_topic = sel

with right:
    st.subheader(f"세부 조정 · {st.session_state.selected_topic}")
    t = st.session_state.selected_topic
    tabs = st.tabs(sources)
    for i, s in enumerate(sources):
        with tabs[i]:
            # 분포/파라미터 에디터
            cell = st.session_state.config.get(t, {}).get(s, {})
            if t not in st.session_state.config: st.session_state.config[t]={}
            if s not in st.session_state.config[t]: st.session_state.config[t][s]={}
            cell = st.session_state.config[t][s]
            dist  = st.selectbox("Distribution", options=["LN","ZINB","ZIP","C"], index=["LN","ZINB","ZIP","C"].index(cell.get("Distribution","C")),
                                 key=f"dist_{t}_{s}")
            # 파라미터 위젯
            pbox = {}
            if dist=="LN":
                prm = cell.get("Parameter", {"median":1.0,"p90":2.0})
                pbox["median"] = st.number_input("median", value=float(prm.get("median",1.0)), min_value=1e-9, key=f"p_m_{t}_{s}")
                pbox["p90"]    = st.number_input("p90",    value=float(prm.get("p90",  2.0)), min_value=1e-9, key=f"p_p90_{t}_{s}")
            elif dist=="ZIP":
                prm = cell.get("Parameter", {"mean":0.0,"p0":0.0})
                pbox["mean"] = st.number_input("mean", value=float(prm.get("mean",0.0)), min_value=0.0, key=f"p_mean_{t}_{s}")
                pbox["p0"]   = st.number_input("p0",   value=float(prm.get("p0",  0.0)), min_value=0.0, max_value=1.0, step=0.01, key=f"p_p0_{t}_{s}")
            elif dist=="ZINB":
                prm = cell.get("Parameter", {"mean":0.0,"var":1.0,"p0":0.0})
                pbox["mean"] = st.number_input("mean", value=float(prm.get("mean",0.0)), min_value=0.0, key=f"p_mean_{t}_{s}")
                pbox["var"]  = st.number_input("var",  value=float(prm.get("var", 1.0)), min_value=0.0, key=f"p_var_{t}_{s}")
                pbox["p0"]   = st.number_input("p0",   value=float(prm.get("p0",  0.0)), min_value=0.0, max_value=1.0, step=0.01, key=f"p_p0_{t}_{s}")
            else:  # C
                prm = cell.get("Parameter", {"p":0.5})
                pbox["p"] = st.number_input("p", value=float(prm.get("p",0.5)), min_value=0.0, max_value=1.0, step=0.01, key=f"p_const_{t}_{s}")

            # 변경 적용
            cover_val = bool(cell.get("Cover", False))
            st.markdown(f"**Cover:** `{cover_val}`")
            st.session_state.config[t][s]["Distribution"] = dist
            st.session_state.config[t][s]["Parameter"] = pbox

            # PDF/CDF
            pdf_fig, cdf_fig, xmark, const_p = make_source_plots(t, s, st.session_state.input_df, st.session_state.config)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(pdf_fig, use_container_width=True, key=f"pdf_{t}_{s}")
            with c2:
                st.plotly_chart(cdf_fig, use_container_width=True, key=f"cdf_{t}_{s}")

            badge = " · Fallback C(0.5) 적용" if const_p is not None else ""
            st.caption(f"Input 표시값: {xmark}{badge}")


# ---------- 하단: 다운로드 ----------
st.divider()
conf_str = json.dumps(st.session_state.config, ensure_ascii=False, indent=2)
st.download_button("편집된 config.json 다운로드", data=conf_str, file_name="config.edited.json", mime="application/json")
