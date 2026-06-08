"""
Análise de Correlação Multivariada com Detecção de Lag — App Streamlit
----------------------------------------------------------------------
Importe uma planilha (data/hora + colunas de indicadores), escolha o
indicador-chave e descubra quais indicadores estão mais correlacionados com
ele, detectando automaticamente o atraso (lag) entre causa e efeito.

Rodar:  streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis

st.set_page_config(page_title="Análise de Correlação Multivariada", layout="wide")

st.title("📊 Análise de Correlação Multivariada")
st.caption(
    "Ferramenta para engenheiros detectarem a **causa** de aumento ou redução de "
    "um indicador-chave a partir de uma série de indicadores de processo — "
    "considerando o atraso (lag) entre causa e efeito."
)


# --------------------------------------------------------------------------- #
# Helpers de gráfico
# --------------------------------------------------------------------------- #
def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std else s - s.mean()


def fig_ranking(ranking: pd.DataFrame) -> go.Figure:
    data = ranking.dropna(subset=["|Correlação|"]).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["|Correlação|"],
            y=data["Indicador"],
            orientation="h",
            text=[
                f"{c:+.2f} @ {l:g} min"
                for c, l in zip(data["Correlação (lag ótimo)"], data["Lag ótimo (min)"])
            ],
            textposition="auto",
            marker_color=data["|Correlação|"],
            marker_colorscale="Blues",
        )
    )
    fig.update_layout(
        title="Ranking de correlação (no lag ótimo)",
        xaxis_title="|Correlação| no lag ótimo",
        yaxis_title="Indicador",
        height=max(300, 60 * len(data)),
    )
    return fig


def fig_cross_corr(res: analysis.LagResult, step_minutes: float) -> go.Figure:
    lags_min = res.lags * step_minutes
    fig = go.Figure(go.Scatter(x=lags_min, y=res.corrs, mode="lines+markers"))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.add_vline(
        x=res.best_lag_minutes,
        line_dash="dash",
        line_color="red",
        annotation_text=f"lag ótimo = {res.best_lag_minutes:g} min",
        annotation_position="top",
    )
    fig.update_layout(
        title=f"Correlação cruzada: {res.indicator} × chave",
        xaxis_title="Lag (min) — positivo = indicador antecede a chave (causa→efeito)",
        yaxis_title="Correlação de Pearson",
        height=380,
    )
    return fig


def fig_overlay(df: pd.DataFrame, key: str, res: analysis.LagResult) -> go.Figure:
    shifted = df[res.indicator].shift(res.best_lag_samples)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=_zscore(df[key]), name=f"{key} (chave)"))
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=_zscore(shifted),
            name=f"{res.indicator} (deslocado {res.best_lag_minutes:g} min)",
        )
    )
    fig.update_layout(
        title="Séries normalizadas (z-score) alinhadas pelo lag ótimo",
        xaxis_title="Tempo",
        yaxis_title="z-score",
        height=380,
    )
    return fig


def fig_impact(
    df: pd.DataFrame, key: str, res: analysis.LagResult, event: analysis.MainEvent
) -> go.Figure:
    """Causa→efeito na linha de tempo REAL (sem ajuste de lag), destacando a
    janela do maior distúrbio na causa e a janela de efeito na variável-chave."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=_zscore(df[key]), name=f"{key} (chave)",
                   line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=_zscore(df[res.indicator]),
                   name=f"{res.indicator} (causa)", line=dict(color="#d62728"))
    )

    # Janela do distúrbio na causa (laranja) e janela de efeito na chave (verde)
    fig.add_vrect(
        x0=event.cause_start, x1=event.cause_end,
        fillcolor="orange", opacity=0.20, line_width=0,
        annotation_text="distúrbio (causa)", annotation_position="top left",
    )
    fig.add_vrect(
        x0=event.effect_start, x1=event.effect_end,
        fillcolor="green", opacity=0.18, line_width=0,
        annotation_text="impacto na chave", annotation_position="top right",
    )

    # Seta ligando o pico da causa ao centro da janela de efeito
    effect_mid = event.effect_start + (event.effect_end - event.effect_start) / 2
    fig.add_annotation(
        x=effect_mid, ax=event.peak_time, xref="x", axref="x",
        y=1.05, ay=1.05, yref="paper", ayref="paper",
        showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="gray",
        text=f"lag = {event.lag_minutes:g} min", font=dict(color="gray"),
    )

    fig.update_layout(
        title=(
            f"Impacto causa→efeito na linha de tempo real (sem ajuste de lag): "
            f"{res.indicator} → {key}"
        ),
        xaxis_title="Tempo (posição real)",
        yaxis_title="z-score",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )
    return fig


def fig_scatter(df: pd.DataFrame, key: str, res: analysis.LagResult) -> go.Figure:
    pair = pd.concat(
        [df[key], df[res.indicator].shift(res.best_lag_samples)], axis=1
    ).dropna()
    pair.columns = [key, res.indicator]
    fig = go.Figure(
        go.Scatter(x=pair[res.indicator], y=pair[key], mode="markers", opacity=0.6)
    )
    fig.update_layout(
        title=f"Dispersão no lag ótimo (r = {res.corr_at_lag:+.3f})",
        xaxis_title=f"{res.indicator} (deslocado)",
        yaxis_title=key,
        height=380,
    )
    return fig


def fig_gauge(impact_pct: float, driver: str) -> go.Figure:
    """Medidor (gauge) do percentual de impacto do driver principal."""
    val = 0.0 if impact_pct is None or np.isnan(impact_pct) else impact_pct
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "%"},
            title={"text": f"Impacto de<br>{driver}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#f8d7da"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def fig_key_timeline(df: pd.DataFrame, key: str) -> go.Figure:
    """Mini série temporal do indicador-chave (contexto rápido no painel)."""
    fig = go.Figure(go.Scatter(x=df.index, y=df[key], mode="lines", name=key))
    fig.update_layout(
        title=f"Histórico do indicador-chave: {key}",
        height=260,
        margin=dict(t=40, b=10),
        xaxis_title=None,
        yaxis_title=key,
    )
    return fig


def to_excel_bytes(ranking: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ranking.to_excel(writer, index=False, sheet_name="Ranking")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# 1. Upload
# --------------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Importe a planilha (.xlsx ou .csv) — 1ª coluna = data/hora; demais = indicadores",
    type=["xlsx", "csv"],
)

if uploaded is None:
    st.info(
        "Aguardando a planilha. Formato esperado: a primeira coluna contém a "
        "data e hora (na mesma célula) e cada coluna seguinte é um indicador, "
        "com os valores alinhados ao instante da linha."
    )
    st.stop()

try:
    raw = analysis.load_table(uploaded)
except Exception as exc:  # noqa: BLE001
    st.error(f"Erro ao ler a planilha: {exc}")
    st.stop()

with st.expander("👁️ Prévia dos dados importados", expanded=False):
    st.dataframe(raw.head(20), use_container_width=True)

# --------------------------------------------------------------------------- #
# 2. Reamostragem
# --------------------------------------------------------------------------- #
col_a, col_b = st.columns([1, 1])
with col_a:
    override = st.checkbox("Definir passo manualmente", value=False)
manual_step = None
if override:
    with col_b:
        manual_step = st.number_input(
            "Passo (minutos)", min_value=0.1, value=1.0, step=0.5
        )

df, info = analysis.detect_and_resample(raw, step_minutes=manual_step)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Passo detectado", f"{info.step_minutes:g} min")
m2.metric("Amostras", info.n_rows)
m3.metric("Lacunas interpoladas", info.n_gaps_filled)
m4.metric("NaN restante", f"{info.nan_fraction*100:.1f}%")

if info.irregular:
    st.warning(
        "Os timestamps originais são irregulares — os dados foram reamostrados "
        "para uma grade fixa. Verifique se o passo detectado faz sentido."
    )
if info.nan_fraction > 0.3:
    st.warning(
        "Mais de 30% das células ficaram vazias após a reamostragem. "
        "As correlações podem ficar instáveis; considere ajustar o passo."
    )

# --------------------------------------------------------------------------- #
# 3-4. Indicador-chave + janela de lag
# --------------------------------------------------------------------------- #
st.subheader("⚙️ Configuração da análise")
c1, c2 = st.columns([2, 1])
with c1:
    key = st.selectbox("Indicador-chave (efeito a avaliar)", options=list(df.columns))
with c2:
    span_min = (info.n_rows * info.step_minutes) / 4 if info.n_rows else info.step_minutes
    default_lag = max(info.step_minutes * 5, info.step_minutes)
    max_lag_min = st.number_input(
        "Janela máxima de lag (min)",
        min_value=info.step_minutes,
        value=float(round(min(default_lag, max(span_min, default_lag)), 3)),
        step=info.step_minutes,
        help="Maior atraso causa→efeito a procurar, em minutos.",
    )

max_lag = max(1, int(round(max_lag_min / info.step_minutes)))

if st.button("🔍 Analisar", type="primary"):
    with st.spinner("Calculando correlações e lags..."):
        ranking, results = analysis.analyze_all(df, key, max_lag, info.step_minutes)
    st.session_state["ranking"] = ranking
    st.session_state["results"] = results
    st.session_state["key"] = key

STATUS_WINDOW = 3  # nº de amostras recentes para avaliar distúrbio ativo


def _dir_arrow(d: int) -> str:
    return "subindo ↑" if d > 0 else ("caindo ↓" if d < 0 else "estável →")


def _status_label(stt: analysis.CurrentStatus) -> str:
    if not stt.disturbed:
        return "🟢 Estável"
    return "🔴 Distúrbio " + ("↑" if stt.direction > 0 else "↓")


# --------------------------------------------------------------------------- #
# 5. Resultados — Painel de Decisão (principal) + Análise Detalhada
# --------------------------------------------------------------------------- #
if "ranking" in st.session_state and st.session_state.get("key") == key:
    ranking_base = st.session_state["ranking"]
    results = st.session_state["results"]

    if ranking_base.empty or ranking_base["|Correlação|"].dropna().empty:
        st.warning("Não foi possível calcular correlações (dados insuficientes).")
        st.stop()

    ranking = analysis.add_impact_columns(ranking_base)
    top = ranking.iloc[0]
    res_top = results[top["Indicador"]]

    # Status atual de cada driver (para KPIs, gatilhos e tabela)
    statuses = {ind: analysis.recent_status(df[ind], window=STATUS_WINDOW)
                for ind in ranking["Indicador"]}

    tab_dash, tab_detail = st.tabs(["🚦 Painel de Decisão", "🔬 Análise Detalhada"])

    # ===================================================================== #
    # ABA PRINCIPAL — Painel de Decisão
    # ===================================================================== #
    with tab_dash:
        key_s = df[key].dropna()
        cur_val = float(key_s.iloc[-1]) if len(key_s) else float("nan")
        ref_idx = -(STATUS_WINDOW + 1)
        prev_val = float(key_s.iloc[ref_idx]) if len(key_s) > STATUS_WINDOW else cur_val
        delta = cur_val - prev_val

        st.subheader("🎯 Diagnóstico")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{key} (atual)", f"{cur_val:.2f}", delta=f"{delta:+.2f}")
        k2.metric("Principal causa", top["Indicador"], help="Maior correlação no lag ótimo")
        k3.metric(
            "Impacto (variância explicada)",
            f"{top['Impacto (%)']:.0f}%",
            help="R² = corr² — fração da variação do indicador-chave explicada por este driver.",
        )
        k4.metric(
            "Antecedência (lag)",
            f"{top['Lag ótimo (min)']:g} min",
            delta=top["Sentido"],
            delta_color="off",
            help="Tempo entre o distúrbio na causa e o efeito no indicador-chave.",
        )

        c_gauge, c_hist = st.columns([1, 2])
        c_gauge.plotly_chart(
            fig_gauge(top["Impacto (%)"], top["Indicador"]), use_container_width=True
        )
        c_hist.plotly_chart(fig_key_timeline(df, key), use_container_width=True)

        # --- Gatilhos acionáveis ---------------------------------------- #
        st.subheader("🚨 Gatilhos acionáveis")
        triggers = []
        for _, row in ranking.iterrows():
            ind = row["Indicador"]
            stt = statuses[ind]
            corr = row["Correlação (lag ótimo)"]
            lagmin = row["Lag ótimo (min)"]
            # gatilho preditivo só faz sentido para drivers que ANTECEDEM (lag > 0)
            if stt.disturbed and lagmin > 0 and not np.isnan(corr):
                key_dir = stt.direction * (1 if corr >= 0 else -1)
                triggers.append((ind, stt, key_dir, lagmin, row["Impacto (%)"]))

        if not triggers:
            st.success(
                "Nenhum driver em distúrbio no momento — o processo está estável. "
                "Sem ações recomendadas."
            )
        else:
            st.caption(
                "Drivers que antecedem o indicador-chave e estão com nível atípico "
                "agora — janela de antecedência para agir antes do efeito se concretizar."
            )
            for ind, stt, key_dir, lagmin, impact in triggers:
                tendencia = "subir ↑" if key_dir > 0 else "cair ↓"
                msg = (
                    f"**{ind}** está **{_dir_arrow(stt.direction)}** agora "
                    f"(z = {stt.z:+.1f}) → previsão: **{key}** tende a **{tendencia}** "
                    f"em ~**{lagmin:g} min** · impacto {impact:.0f}%"
                )
                (st.error if key_dir < 0 else st.warning)(msg)

        # --- Evidência: impacto causa→efeito na linha de tempo real ----- #
        st.subheader("🔍 Evidência do impacto (causa → efeito)")
        event = analysis.detect_main_event(
            df[res_top.indicator], res_top.best_lag_samples, info.step_minutes
        )
        if event is None:
            st.info("Sem distúrbio significativo na variável principal para destacar.")
        else:
            st.caption(
                f"Maior distúrbio em **{res_top.indicator}** por volta de "
                f"**{event.peak_time:%d/%m %H:%M}** (z = {event.peak_z:+.1f}); o efeito "
                f"em **{key}** aparece ~**{event.lag_minutes:g} min** depois (região verde)."
            )
            st.plotly_chart(fig_impact(df, key, res_top, event), use_container_width=True)

        # --- Tabela de drivers ------------------------------------------ #
        st.subheader("📑 Drivers do indicador-chave")
        board = ranking[[
            "Indicador", "Sentido", "Lag ótimo (min)",
            "Impacto (%)", "Contribuição relativa (%)",
        ]].copy()
        board.insert(1, "Status atual", [_status_label(statuses[i]) for i in ranking["Indicador"]])
        st.dataframe(board, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Baixar painel (Excel)",
            data=to_excel_bytes(board),
            file_name="painel_drivers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ===================================================================== #
    # ABA — Análise Detalhada
    # ===================================================================== #
    with tab_detail:
        st.success(
            f"**Indicador mais correlacionado:** {top['Indicador']} "
            f"(r = {top['Correlação (lag ótimo)']:+.3f}, "
            f"lag = {top['Lag ótimo (min)']:g} min)"
        )

        st.subheader("📋 Ranking completo")
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Baixar ranking (Excel)",
            data=to_excel_bytes(ranking),
            file_name="ranking_correlacao.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.plotly_chart(fig_ranking(ranking), use_container_width=True)

        st.subheader("🔎 Detalhe por indicador")
        sel = st.selectbox(
            "Indicador para inspecionar", options=ranking["Indicador"].tolist()
        )
        res = results[sel]

        g1, g2 = st.columns(2)
        g1.plotly_chart(fig_cross_corr(res, info.step_minutes), use_container_width=True)
        g2.plotly_chart(fig_scatter(df, key, res), use_container_width=True)
        st.plotly_chart(fig_overlay(df, key, res), use_container_width=True)

        if res.n_points < 30:
            st.warning(
                f"Apenas {res.n_points} pontos no lag ótimo — correlação pouco "
                "confiável. Reduza a janela de lag ou use uma série mais longa."
            )
