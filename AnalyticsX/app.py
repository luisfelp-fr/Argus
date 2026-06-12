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
import auth

st.set_page_config(page_title="Argus AnalyticsX - Análise Avançada de Processo", layout="wide")

# --- Controle de acesso (login + senha) --------------------------------- #
# Bloqueia todo o app até a autenticação. Credenciais ficam em st.secrets.
auth.require_login()
auth.logout_button()   # usuário logado + botão "Sair" na barra lateral

st.title("📊 Argus AnalyticsX - Análise Avançada de Processo")

# Permite que nomes longos de indicadores quebrem linha nos KPIs (st.metric)
# em vez de serem cortados com reticências.
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.25;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    y_key = _zscore(df[key])
    y_cause = _zscore(df[res.indicator])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=y_key, name=f"{key} (chave)",
                   line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=y_cause,
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

    # Seta (horizontal, no topo) ligando o pico da causa ao centro da janela de efeito.
    # Usa coordenadas de eixo (x e y) — 'paper' não é válido para axref/ayref.
    effect_mid = event.effect_start + (event.effect_end - event.effect_start) / 2
    ytop = float(np.nanmax([y_key.max(), y_cause.max()]))
    fig.add_annotation(
        x=effect_mid, ax=event.peak_time, xref="x", axref="x",
        y=ytop, ay=ytop, yref="y", ayref="y",
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
    """Medidor (gauge) do percentual de impacto do driver principal (0–100%)."""
    val = 0.0 if impact_pct is None or np.isnan(impact_pct) else float(impact_pct)
    val = max(0.0, min(100.0, val))  # garante a escala 0–100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": "%"},
            title={"text": f"Impacto de<br>{driver}"},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "linear",
                    "dtick": 20,
                    "ticksuffix": "%",
                },
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


def fig_dual_axis(df: pd.DataFrame, var_primary: str, var_secondary: str) -> go.Figure:
    """Par de variáveis em UNIDADES REAIS (sem z-score), em eixo duplo.

    ``var_primary`` no eixo principal (esquerda) e ``var_secondary`` no eixo
    secundário (direita), na linha de tempo real (sem deslocamento por lag)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df[var_primary], name=var_primary,
                   line=dict(color="#1f77b4"))
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[var_secondary], name=var_secondary,
                   line=dict(color="#d62728"), yaxis="y2")
    )
    fig.update_layout(
        title=f"{var_primary} (eixo esq.) × {var_secondary} (eixo dir.) — unidades reais",
        xaxis_title="Tempo (posição real)",
        yaxis=dict(title=var_primary, color="#1f77b4"),
        yaxis2=dict(title=var_secondary, color="#d62728", overlaying="y", side="right"),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )
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


def fig_corr_matrix(df: pd.DataFrame) -> go.Figure:
    """Matriz de correlação (Pearson, lag 0) entre todos os indicadores + chave."""
    corr = df.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            zmin=-1, zmax=1,
            colorscale="RdBu", reversescale=True,  # +1 vermelho, −1 azul
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=11),
            colorbar=dict(title="r"),
        )
    )
    n = len(corr.columns)
    fig.update_layout(
        title="Matriz de correlação entre indicadores (sem defasagem / lag 0)",
        height=max(380, 70 * n),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=60, b=40),
    )
    return fig


def to_excel_bytes(ranking: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        ranking.to_excel(writer, index=False, sheet_name="Ranking")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Seletor de modo (Versão 2)
# --------------------------------------------------------------------------- #
modo = st.radio(
    "Modo de análise",
    ["📈 Análise Sazonal", "📊 Análise de Variáveis Contínuas"],
    horizontal=True,
    help="Análise Sazonal: correlaciona indicadores de processo minuto a minuto com um "
         "indicador sazonal (diário ou de poucas horas), com lag em minutos, modelo e SHAP. "
         "Análise de Variáveis Contínuas: análise de correlação entre indicadores minuto a "
         "minuto, com lag em minutos (modo clássico).",
)

if modo.startswith("📈"):
    import daily_ui
    daily_ui.render_seasonal_mode()
    st.stop()

st.caption(
    "Ferramenta para engenheiros detectarem a **causa** de aumento ou redução de "
    "um indicador-chave a partir de uma série de indicadores de processo — "
    "considerando o atraso (lag) entre causa e efeito."
)

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
# 2. Pré-processamento — reamostragem normal OU amostras compostas (laboratório)
# --------------------------------------------------------------------------- #
lab_mode = st.toggle(
    "Há indicadores de laboratório (amostras compostas)?",
    value=False,
    help="Variáveis medidas em laboratório por composição (ex.: a cada 4 h), cujo "
         "valor é a MÉDIA do período e não pode ser interpolado. O instante "
         "registrado é o fim do período.",
)

if lab_mode:
    lc1, lc2 = st.columns([2, 1])
    with lc1:
        lab_inds = st.multiselect(
            "Quais indicadores são variáveis de laboratório?",
            options=list(raw.columns),
        )
    with lc2:
        period_h = st.number_input(
            "Período da análise composta (horas)",
            min_value=0.5, value=4.0, step=0.5,
            help="Nº de horas que cada amostra de laboratório compõe. A janela é "
                 "(T − período, T], com T = data/hora registrada (fim do período).",
        )
    if not lab_inds:
        st.info("Selecione ao menos um indicador de laboratório para continuar.")
        st.stop()
    try:
        df, info = analysis.composite_average(raw, lab_inds, period_h)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao montar as amostras compostas: {exc}")
        st.stop()
    st.caption(
        f"🧪 **Modo laboratório:** todos os indicadores foram agregados pela média "
        f"em janelas de **{period_h:g} h**, ancoradas nas **{info.n_rows}** coletas de "
        f"laboratório. As variáveis de laboratório mantêm o valor reportado (sem interpolação)."
    )
else:
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
m1.metric("Passo" + (" (composto)" if lab_mode else " detectado"), f"{info.step_minutes:g} min")
m2.metric("Amostras", info.n_rows)
m3.metric("Lacunas interpoladas", info.n_gaps_filled)
m4.metric("NaN restante", f"{info.nan_fraction*100:.1f}%")

if info.irregular:
    st.warning(
        "As coletas de laboratório têm espaçamento irregular — verifique o agendamento."
        if lab_mode else
        "Os timestamps originais são irregulares — os dados foram reamostrados "
        "para uma grade fixa. Verifique se o passo detectado faz sentido."
    )
if info.nan_fraction > 0.3:
    st.warning(
        "Mais de 30% das células estão vazias. "
        "As correlações podem ficar instáveis; considere revisar os dados/parâmetros."
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

cfg1, cfg2 = st.columns([1, 1])
with cfg1:
    remove_collinear = st.toggle(
        "Eliminar indicadores com multicolinearidade",
        value=True,
        help="Ligado: remove indicadores redundantes entre si (|corr| alto / VIF alto), "
             "mantendo um representante por grupo. Desligado: mantém todos na atribuição "
             "(as fatias podem ficar menos estáveis; o VIF segue como alerta).",
    )
with cfg2:
    var_strong_pct = st.slider(
        "Variação forte — limiar de CV (%)",
        min_value=5, max_value=100, value=15, step=5,
        help="Um indicador é considerado de 'variação forte' (e gera gatilho) quando o "
             "coeficiente de variação no período todo (σ/|μ|) atinge este limiar.",
    )
strong_cv = var_strong_pct / 100.0

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


_STAB_EMOJI = {"Estável": "🟢", "Moderada": "🟡", "Instável": "🔴"}


def _stab_label(stab: analysis.StabilityResult) -> str:
    emoji = _STAB_EMOJI.get(stab.classification, "⚪")
    cv = "" if np.isnan(stab.cv) else f" · CV {stab.cv * 100:.0f}%"
    return f"{emoji} {stab.classification}{cv}"


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
    top = ranking.iloc[0]  # topo por correlação simples (usado na aba Detalhada)
    # Atribuição rigorosa: fatia real (não-sobreposta) da variância por driver.
    # A eliminação de multicolinearidade é controlada pelo toggle da configuração.
    attribution = analysis.robust_attribution(
        df, key, results, remove_collinear=remove_collinear
    )
    atable = attribution.table

    # Estabilidade de cada driver no PERÍODO TODO (para tabela e gatilhos)
    stabilities = {ind: analysis.overall_stability(df[ind], strong_cv=strong_cv)
                   for ind in ranking["Indicador"]}

    tab_dash, tab_detail = st.tabs(["🚦 Painel de Decisão", "🔬 Análise Detalhada"])

    # ===================================================================== #
    # ABA PRINCIPAL — Painel de Decisão (atribuição rigorosa de variância)
    # ===================================================================== #
    with tab_dash:
        if atable.empty:
            st.warning(
                "Dados insuficientes para a atribuição rigorosa de variância "
                "(poucos pontos após alinhar os lags). Veja a aba Análise Detalhada."
            )
        else:
            top_a = atable.iloc[0]
            top_ind = top_a["Indicador"]
            res_top = results[top_ind]

            key_s = df[key].dropna()
            cur_val = float(key_s.iloc[-1]) if len(key_s) else float("nan")
            ref_idx = -(STATUS_WINDOW + 1)
            prev_val = float(key_s.iloc[ref_idx]) if len(key_s) > STATUS_WINDOW else cur_val
            delta = cur_val - prev_val

            key_stab = analysis.overall_stability(df[key], strong_cv=strong_cv)

            st.subheader("🎯 Diagnóstico")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric(f"{key} (atual)", f"{cur_val:.2f}", delta=f"{delta:+.2f}")
            k2.metric(
                "Estabilidade do indicador-chave",
                f"{_STAB_EMOJI.get(key_stab.classification, '⚪')} {key_stab.classification}",
                delta=("CV —" if np.isnan(key_stab.cv) else f"CV {key_stab.cv * 100:.0f}%"),
                delta_color="off",
                help="Estabilidade do indicador-chave no período todo, pelo coeficiente "
                     "de variação (desvio padrão / |média|). CV menor = mais estável.",
            )
            k3.metric("Principal causa", top_ind, delta=top_a["Sentido"],
                      delta_color="off", help="Maior fatia da variância (Shapley/LMG).")
            k4.metric(
                "Impacto (fatia real)",
                f"{top_a['Fatia da variância (%)']:.0f}%",
                help="Fração real e não-sobreposta da variância do indicador-chave "
                     "explicada por este driver (decomposição de Shapley/LMG).",
            )
            k5.metric(
                "Variância total explicada",
                f"{attribution.model_r2 * 100:.0f}%",
                help="R² do modelo com todos os drivers (após remover redundâncias).",
            )

            c_gauge, c_hist = st.columns([1, 2])
            c_gauge.plotly_chart(
                fig_gauge(top_a["Fatia da variância (%)"], top_ind),
                use_container_width=True, key="kpi_gauge",
            )
            c_hist.plotly_chart(fig_key_timeline(df, key),
                                use_container_width=True, key="key_timeline")

            # --- Gatilhos acionáveis (forte variação no período todo) ---- #
            st.subheader("🚨 Gatilhos acionáveis")
            impact_by_ind = dict(
                zip(atable["Indicador"], atable["Fatia da variância (%)"])
            )
            triggers = []
            for ind in atable["Indicador"]:
                stab = stabilities[ind]
                res = results[ind]
                corr = res.corr_at_lag
                lagmin = res.best_lag_minutes
                # gatilho: driver com FORTE VARIAÇÃO no período que ANTECEDE a chave
                if stab.strong and lagmin > 0 and not np.isnan(corr):
                    key_dir = stab.trend * (1 if corr >= 0 else -1)
                    triggers.append((ind, stab, key_dir, lagmin, impact_by_ind[ind]))

            if not triggers:
                st.success(
                    f"Nenhum indicador apresentou variação forte (CV ≥ {var_strong_pct}%) "
                    "no período analisado, ou os que variaram não antecedem o "
                    "indicador-chave. Sem gatilhos."
                )
            else:
                st.caption(
                    f"Indicadores com **forte variação** no período todo (CV ≥ "
                    f"{var_strong_pct}%) que **antecedem** o indicador-chave — prováveis "
                    "causas a monitorar/controlar."
                )
                for ind, stab, key_dir, lagmin, impact in triggers:
                    tendencia = "subir ↑" if key_dir > 0 else "cair ↓"
                    msg = (
                        f"**{ind}** teve **forte variação** no período "
                        f"(CV {stab.cv * 100:.0f}%, {_dir_arrow(stab.trend)}) e antecede "
                        f"**{key}** em ~**{lagmin:g} min** → influência provável: "
                        f"**{key}** tende a **{tendencia}** · fatia {impact:.0f}%"
                    )
                    (st.error if key_dir < 0 else st.warning)(msg)

            # --- Evidência: chave × driver #1 (unidades reais, eixo duplo) #
            st.subheader("🔍 Evidência do impacto (causa → efeito)")
            st.plotly_chart(fig_dual_axis(df, key, top_ind),
                            use_container_width=True, key="dual_evidence")

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
                st.plotly_chart(fig_impact(df, key, res_top, event),
                                use_container_width=True, key="impact_chart")

            # --- Tabela de drivers (fatia rigorosa) --------------------- #
            st.subheader("📑 Drivers do indicador-chave")
            board = atable.copy()
            board.insert(
                1, "Estabilidade (período)",
                [_stab_label(stabilities[i]) for i in atable["Indicador"]],
            )
            st.dataframe(board, use_container_width=True, hide_index=True)
            st.caption(
                "**Estabilidade (período)**: avaliação no período todo pelo coeficiente "
                f"de variação (CV = σ/|μ|) — 🟢 Estável (CV<5%), 🟡 Moderada (5–{var_strong_pct}%), "
                f"🔴 Instável (≥{var_strong_pct}%). **Fatia da variância (%)**: parcela real "
                "e não-sobreposta da variação do indicador-chave atribuída a cada driver. "
                "**Contribuição no modelo (%)**: a mesma fatia normalizada para 100%. "
                "**VIF**: grau de redundância remanescente."
            )
            st.download_button(
                "⬇️ Baixar painel (Excel)",
                data=to_excel_bytes(board),
                file_name="painel_drivers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            if not attribution.removed.empty:
                with st.expander(
                    f"🔧 {len(attribution.removed)} indicador(es) removido(s) por redundância"
                ):
                    st.caption(
                        "Removidos por multicolinearidade alta (explicavam a mesma "
                        "variação de outro driver). A fatia ficou com o representante mantido."
                    )
                    st.dataframe(
                        attribution.removed, use_container_width=True, hide_index=True
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
        st.plotly_chart(fig_ranking(ranking), use_container_width=True, key="ranking_bar")

        st.subheader("🔎 Detalhe por indicador")
        sel = st.selectbox(
            "Indicador para inspecionar", options=ranking["Indicador"].tolist()
        )
        res = results[sel]

        g1, g2 = st.columns(2)
        g1.plotly_chart(fig_cross_corr(res, info.step_minutes),
                        use_container_width=True, key="cross_corr")
        g2.plotly_chart(fig_scatter(df, key, res),
                        use_container_width=True, key="scatter")
        st.plotly_chart(fig_overlay(df, key, res),
                        use_container_width=True, key="overlay")

        if res.n_points < 30:
            st.warning(
                f"Apenas {res.n_points} pontos no lag ótimo — correlação pouco "
                "confiável. Reduza a janela de lag ou use uma série mais longa."
            )

        # --- Matriz de correlação entre todos os indicadores ------------ #
        st.subheader("🧮 Matriz de correlação")
        st.caption(
            "Correlação de Pearson (sem defasagem) entre todos os indicadores e o "
            "indicador-chave. Útil para ver relações diretas/inversas e identificar "
            "indicadores redundantes entre si (valores próximos de ±1)."
        )
        st.plotly_chart(fig_corr_matrix(df), use_container_width=True, key="corr_matrix")

        # --- Comparação livre de pares (eixo duplo) --------------------- #
        st.subheader("📈 Comparar variáveis (eixo duplo)")
        st.caption(
            "Selecione duas variáveis para acompanhar a variação ao longo do tempo, "
            "cada uma em seu eixo (principal e secundário), em unidades reais."
        )
        all_vars = list(df.columns)
        pc1, pc2 = st.columns(2)
        with pc1:
            var_a = st.selectbox(
                "Variável — eixo principal (esq.)", options=all_vars, index=0,
                key="pair_primary",
            )
        with pc2:
            default_b = 1 if len(all_vars) > 1 else 0
            var_b = st.selectbox(
                "Variável — eixo secundário (dir.)", options=all_vars,
                index=default_b, key="pair_secondary",
            )
        if var_a == var_b:
            st.info("Selecione duas variáveis diferentes para a comparação.")
        else:
            st.plotly_chart(fig_dual_axis(df, var_a, var_b),
                            use_container_width=True, key="dual_pair")
