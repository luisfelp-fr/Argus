"""
Modo "Análise Sazonal" — Versão 2
---------------------------------
Correlaciona o comportamento dos indicadores de processo (minuto a minuto) com um
**indicador sazonal** informado periodicamente (cadência diária OU sub-diária, ex.:
a cada poucas horas). O sistema agrega o processo na janela de cada período (definido
pelos horários do indicador sazonal), considera efeitos de **lag em minutos** e aponta
os indicadores com maior EVIDÊNCIA ESTATÍSTICA de influência (prováveis causas — sem
causalidade absoluta).

Exposto por ``render_seasonal_mode()``, chamado pelo seletor de modo em ``app.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis

# Textos de ajuda (ícones "?") reutilizados nos tooltips
H_TIPO = ("Só estatística: correlações e informação mútua. "
          "Estatística + Modelo: adiciona um modelo preditivo (importância das variáveis). "
          "Completa (com SHAP): adiciona a explicabilidade SHAP.")
H_MODELO = ("RandomForest: floresta de árvores de decisão — robusto e estável. "
            "XGBoost: gradient boosting — costuma ser mais preciso, porém mais sensível ao ajuste.")
H_LAG = ("Maior atraso (em minutos) a considerar entre o processo e o indicador sazonal. "
         "O sistema cria blocos de períodos anteriores até esse limite (passo = período).")
H_PERIODO = ("Janela em que o processo é resumido para cada leitura do indicador sazonal. "
             "Detectado automaticamente pelos horários do indicador; ajuste se necessário.")
H_ESTAT = ("Pearson r: correlação LINEAR (−1 a +1). "
           "Spearman r: correlação de ORDEM, capta relações monotônicas não lineares (−1 a +1). "
           "p-value: probabilidade de a associação ser por acaso — quanto MENOR, mais significativa. "
           "Mutual Information: dependência geral, inclusive não linear (≥ 0).")
H_IMP = "Importância do modelo: o quanto cada variável reduz o erro nas árvores do modelo."
H_SHAP = ("SHAP: impacto médio (em módulo) de cada variável nas previsões do modelo — "
          "o quanto cada uma 'empurra' o resultado para cima ou para baixo.")
H_CONS = ("Score consolidado: combina Spearman, Informação Mútua, importância do modelo e SHAP "
          "(normalizados de 0 a 1) num único índice de evidência.")
H_MAE = "Erro Absoluto Médio: diferença média (nas unidades do indicador) entre previsto e real. Menor é melhor."
H_RMSE = "Raiz do Erro Quadrático Médio: como o MAE, mas penaliza mais os erros grandes. Menor é melhor."
H_MAPE = "Erro Percentual Absoluto Médio: erro médio em %. Menor é melhor."
H_R2 = "R² (coeficiente de determinação): fração da variação explicada pelo modelo (1 = perfeito; ≤ 0 = não explica)."


# --------------------------------------------------------------------------- #
# Helpers de gráfico
# --------------------------------------------------------------------------- #
def _bar_rank(series: pd.Series, title: str, xlabel: str, top_n: int = 20,
              color: str = "Blues") -> go.Figure:
    """Barra horizontal de ranking a partir de uma Series (índice = variável)."""
    s = series.dropna().head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=s.values, y=s.index, orientation="h",
            text=[f"{v:.3f}" for v in s.values], textposition="auto",
            marker_color=s.values, marker_colorscale=color,
        )
    )
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title="Variável",
        height=max(320, 26 * len(s)), margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def _fig_scatter(x: pd.Series, y: pd.Series, xname: str, yname: str) -> go.Figure:
    pair = pd.concat([x, y], axis=1).dropna()
    pair.columns = [xname, yname]
    fig = go.Figure(go.Scatter(x=pair[xname], y=pair[yname], mode="markers", opacity=0.7))
    r = pair[xname].corr(pair[yname]) if len(pair) > 2 else float("nan")
    fig.update_layout(
        title=f"{xname} × {yname} (r = {r:+.3f})",
        xaxis_title=xname, yaxis_title=yname, height=420,
    )
    return fig


def _fig_series(y: pd.Series, title: str, yname: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name=yname))
    fig.update_layout(title=title, xaxis_title="Período", yaxis_title=yname, height=360)
    return fig


def _fig_real_vs_pred(y_all: pd.Series, pred_train: pd.Series,
                      pred_test: pd.Series, yname: str = "Indicador sazonal") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_all.index, y=y_all.values, mode="lines+markers",
                             name=f"{yname} (real)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=pred_train.index, y=pred_train.values, mode="lines",
                             name="Previsto (treino)", line=dict(color="#2ca02c", dash="dot")))
    fig.add_trace(go.Scatter(x=pred_test.index, y=pred_test.values, mode="lines+markers",
                             name="Previsto (teste)", line=dict(color="#d62728")))
    if len(pred_test):
        fig.add_vline(x=pred_test.index[0], line_dash="dash", line_color="gray",
                      annotation_text="início do teste", annotation_position="top")
    fig.update_layout(title=f"{yname}: real × previsto", xaxis_title="Período",
                      yaxis_title=yname, height=420,
                      legend=dict(orientation="h", y=1.1))
    return fig


def _fig_corr_matrix(df: pd.DataFrame) -> go.Figure:
    corr = df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True, text=corr.values,
        texttemplate="%{text:.2f}", textfont=dict(size=9), colorbar=dict(title="r"),
    ))
    n = len(corr.columns)
    fig.update_layout(title="Matriz de correlação (variáveis selecionadas + indicador sazonal)",
                      height=max(400, 32 * n), yaxis=dict(autorange="reversed"),
                      margin=dict(t=50, b=40))
    return fig


def _fmt_periodo(minutes: float) -> str:
    if minutes >= 60:
        return f"{minutes:g} min (~{minutes / 60:g} h)"
    return f"{minutes:g} min"


# --------------------------------------------------------------------------- #
# Fluxo principal
# --------------------------------------------------------------------------- #
def render_seasonal_mode() -> None:
    st.header("📈 Análise Sazonal — evidência estatística de influência")
    st.caption(
        "Carregue a base de **processo minuto a minuto** e a base do **indicador "
        "sazonal** (informado periodicamente — a cada dia ou a cada poucas horas). O "
        "sistema agrega o processo na janela de cada período, considera efeitos de "
        "**lag em minutos** e aponta os indicadores com maior evidência estatística "
        "de influência sobre o indicador sazonal."
    )

    with st.expander("ℹ️ Glossário — o que cada análise e modelo significam"):
        st.markdown(
            "- **Pearson r** — correlação *linear* (−1 a +1).\n"
            "- **Spearman r** — correlação de *ordem* (capta relações monotônicas não lineares).\n"
            "- **p-value** — chance de a associação ser por acaso; quanto **menor**, mais significativa.\n"
            "- **Mutual Information** — dependência geral, inclusive não linear (≥ 0).\n"
            "- **RandomForest** — floresta de árvores de decisão; robusto e estável.\n"
            "- **XGBoost** — *gradient boosting*; costuma ser mais preciso.\n"
            "- **MAE / RMSE / MAPE** — medidas de erro do modelo (menor é melhor).\n"
            "- **R²** — fração da variação explicada pelo modelo (1 = perfeito).\n"
            "- **Importância do modelo** — quanto cada variável reduz o erro nas árvores.\n"
            "- **SHAP** — impacto médio de cada variável nas previsões (explicabilidade).\n"
            "- **Score consolidado** — combina Spearman + MI + importância + SHAP num índice único.\n"
            "- **Lag (min)** — atraso em minutos; o sistema usa blocos de períodos anteriores."
        )

    tab_cfg, tab_base, tab_stat, tab_model, tab_rank, tab_day, tab_export = st.tabs([
        "📥 Dados & Configuração", "🧱 Base por Período", "📊 Estatística",
        "🤖 Modelo & Explicabilidade", "🏆 Ranking Consolidado",
        "📅 Diagnóstico do Período", "⬇️ Exportar",
    ])

    # ===================================================================== #
    # ABA 1 — Dados & Configuração
    # ===================================================================== #
    with tab_cfg:
        uploaded = st.file_uploader(
            "Arquivo único (Excel com abas) ou CSV de processo",
            type=["xlsx", "xls", "csv"], key="v2_file",
        )
        if uploaded is None:
            st.info(
                "Aguardando o arquivo. Em Excel, use **abas distintas** para "
                "processo (DataHora + indicadores) e indicador sazonal (Data + "
                "indicador sazonal). Em CSV, carregue o processo aqui e o indicador "
                "sazonal logo abaixo."
            )
            return

        try:
            uploaded.seek(0)
            sheets = analysis.list_sheets(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erro ao abrir o arquivo: {exc}")
            return

        # --- Seleção de abas / leitura das bases ------------------------- #
        if sheets:
            cs1, cs2 = st.columns(2)
            with cs1:
                proc_sheet = st.selectbox("Aba de processo (minuto a minuto)",
                                          sheets, key="v2_proc_sheet")
            with cs2:
                prod_sheet = st.selectbox("Aba do indicador sazonal", sheets,
                                          index=min(1, len(sheets) - 1), key="v2_prod_sheet")
            try:
                uploaded.seek(0)
                proc_raw = analysis.read_sheet(uploaded, proc_sheet)
                uploaded.seek(0)
                prod_raw = analysis.read_sheet(uploaded, prod_sheet)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro ao ler as abas: {exc}")
                return
        else:
            uploaded.seek(0)
            proc_raw = analysis.read_sheet(uploaded)
            st.info("CSV não possui abas — carregue a base do **indicador sazonal** (CSV) abaixo.")
            prod_file = st.file_uploader("Base do indicador sazonal (CSV)",
                                         type=["csv"], key="v2_prod_file")
            if prod_file is None:
                return
            prod_file.seek(0)
            prod_raw = analysis.read_sheet(prod_file)

        with st.expander("👁️ Prévia das bases", expanded=False):
            st.write("**Processo (minuto a minuto):**")
            st.dataframe(proc_raw.head(10), use_container_width=True)
            st.write("**Indicador sazonal:**")
            st.dataframe(prod_raw.head(10), use_container_width=True)

        # --- Seleção de colunas ----------------------------------------- #
        st.subheader("⚙️ Colunas e parâmetros")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            dt_col = st.selectbox("Coluna de data/hora (processo)",
                                  list(proc_raw.columns), key="v2_dt_col")
        with cc2:
            prod_date_col = st.selectbox("Coluna de data/hora (indicador sazonal)",
                                         list(prod_raw.columns), key="v2_prod_date")
        with cc3:
            prod_num_cols = [c for c in prod_raw.columns if c != prod_date_col]
            target_col = st.selectbox("Coluna do indicador sazonal",
                                      prod_num_cols or list(prod_raw.columns),
                                      key="v2_target")

        # --- Detecção do período sazonal -------------------------------- #
        period_detected = None
        try:
            _ts, period_detected = analysis.parse_target_series(
                prod_raw, prod_date_col, target_col)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Não foi possível ler o indicador sazonal ainda: {exc}")

        pc0, pc1 = st.columns([1, 1])
        with pc0:
            if period_detected:
                st.metric("Período sazonal detectado", _fmt_periodo(period_detected),
                          help=H_PERIODO)
            else:
                st.metric("Período sazonal detectado", "—", help=H_PERIODO)
        with pc1:
            period_min = st.number_input(
                "Período de agregação (minutos)", min_value=1.0,
                value=float(round(period_detected)) if period_detected else 1440.0,
                step=float(max(1, int(period_detected or 60))),
                help=H_PERIODO, key="v2_period",
            )

        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            default_lag = int(period_min * 3)
            max_lag_min = st.number_input(
                "Máx. de lag (minutos)", min_value=0, value=default_lag,
                step=int(max(1, period_min)), help=H_LAG, key="v2_maxlag",
            )
        with cp2:
            tipo = st.selectbox(
                "Tipo de análise",
                ["Completa (com SHAP)", "Estatística + Modelo", "Só estatística"],
                help=H_TIPO, key="v2_tipo",
            )
        with cp3:
            model_options = ["RandomForest"] + (["XGBoost"] if analysis._HAS_XGB else [])
            model_type = st.radio("Modelo preditivo", model_options,
                                  horizontal=True, help=H_MODELO, key="v2_model")

        n_blocks = int(np.floor(max_lag_min / period_min)) if period_min else 0
        st.caption(
            f"Serão criados **{n_blocks + 1} bloco(s)** de defasagem "
            f"(lag 0 + {n_blocks} período(s) anterior(es)), rotulados em minutos."
        )
        use_turnos = st.toggle(
            "Criar variáveis por turno (00–08, 08–16, 16–24)", value=False,
            help="Útil quando o período é diário; pouco relevante para períodos curtos.",
            key="v2_turnos",
        )

        # --- Limites por indicador (tempo fora de faixa) ---------------- #
        ind_cols = [c for c in proc_raw.columns if c != dt_col]
        with st.expander("🎯 Limites por indicador (tempo fora de faixa) — opcional"):
            st.caption(
                "Preencha Mínimo e/ou Máximo apenas dos indicadores que deseja "
                "monitorar. Deixe em branco para ignorar."
            )
            limits_template = pd.DataFrame(
                {"Indicador": ind_cols, "Mínimo": [np.nan] * len(ind_cols),
                 "Máximo": [np.nan] * len(ind_cols)}
            )
            edited = st.data_editor(
                limits_template, hide_index=True, use_container_width=True,
                key="v2_limits", disabled=["Indicador"],
            )

        limits: dict = {}
        for _, r in edited.iterrows():
            lo, hi = r["Mínimo"], r["Máximo"]
            if pd.notna(lo) or pd.notna(hi):
                limits[r["Indicador"]] = (
                    float(lo) if pd.notna(lo) else None,
                    float(hi) if pd.notna(hi) else None,
                )

        run_model = tipo in ("Completa (com SHAP)", "Estatística + Modelo")
        run_shap = tipo == "Completa (com SHAP)"

        if st.button("🚀 Processar", type="primary", key="v2_run"):
            try:
                with st.spinner("Construindo a base por período e analisando..."):
                    target_series, period_det = analysis.parse_target_series(
                        prod_raw, prod_date_col, target_col)
                    period_feat, step_min = analysis.build_period_features(
                        proc_raw, dt_col, target_series.index, float(period_min),
                        limits=limits, use_turnos=use_turnos,
                    )
                    lagged = analysis.add_lag_features_min(
                        period_feat, float(max_lag_min), float(period_min))
                    merged = analysis.merge_with_target(lagged, target_series, "Sazonal")
                    if merged.empty:
                        st.error(
                            "Nenhum período em comum entre processo e indicador sazonal. "
                            "Verifique as colunas de data/hora e a sobreposição dos períodos."
                        )
                        return
                    clean, report = analysis.clean_modeling_table(merged, target="Sazonal")
                    X, y = analysis.split_features_target(clean, "Sazonal")

                    if X.shape[1] == 0:
                        st.error("Nenhuma variável explicativa restou após a limpeza.")
                        return

                    stat = analysis.statistical_ranking(X, y)

                    model_result = None
                    shap_imp = None
                    if run_model:
                        try:
                            model_result = analysis.train_model(
                                X, y, model_type=model_type)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Modelo não treinado: {exc}")
                    if run_shap and model_result is not None:
                        shap_imp = analysis.shap_importance(model_result, X)
                        if shap_imp is None:
                            st.info("SHAP indisponível — usando importância do modelo.")

                    importances = model_result.importances if model_result else None
                    consolidated = analysis.consolidated_ranking(stat, importances, shap_imp)
                    diag = analysis.generate_diagnosis(consolidated, model_result)

                st.session_state["v2_results"] = {
                    "period_feat": period_feat, "step_min": step_min,
                    "period_min": float(period_min), "report": report,
                    "X": X, "y": y, "stat": stat, "model": model_result,
                    "shap": shap_imp, "consolidated": consolidated, "diag": diag,
                    "n_periodos": len(clean), "n_features": X.shape[1],
                }
                st.success(
                    f"Base por período pronta: **{len(clean)} período(s)** × "
                    f"**{X.shape[1]} variáveis explicativas**. Veja as demais abas."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro no processamento: {exc}")
                return

    res = st.session_state.get("v2_results")
    if not res:
        with tab_base:
            st.info("Configure as colunas na aba **Dados & Configuração** e clique em 🚀 Processar.")
        return

    period_feat = res["period_feat"]
    X, y = res["X"], res["y"]
    stat = res["stat"]
    model_result = res["model"]
    shap_imp = res["shap"]
    consolidated = res["consolidated"]

    # ===================================================================== #
    # ABA 2 — Base por Período
    # ===================================================================== #
    with tab_base:
        st.subheader("🧱 Base de indicadores por período",
                     help="Cada linha é um período do indicador sazonal; as colunas são as "
                          "variáveis do processo resumidas naquela janela (antes da expansão de lag).")
        st.caption(
            f"Período sazonal: **{_fmt_periodo(res['period_min'])}** · passo do processo: "
            f"**{res['step_min']:g} min**. Total modelado: **{res['n_periodos']} período(s)** × "
            f"**{res['n_features']} variáveis** (com lag)."
        )
        st.dataframe(period_feat.round(3), use_container_width=True)

        rep = res["report"]
        with st.expander("🧹 Relatório de limpeza"):
            st.write(f"- Colunas não numéricas convertidas: **{len(rep['nao_numericas'])}**")
            st.write(f"- Colunas removidas por excesso de nulos: **{len(rep['muitos_nulos'])}**")
            if rep["muitos_nulos"]:
                st.caption(", ".join(rep["muitos_nulos"][:50]))
            st.write(f"- Colunas constantes removidas: **{len(rep['constantes'])}**")
            if rep["constantes"]:
                st.caption(", ".join(rep["constantes"][:50]))

    # ===================================================================== #
    # ABA 3 — Estatística
    # ===================================================================== #
    with tab_stat:
        if stat.empty:
            st.warning("Sem variáveis com associação calculável.")
        else:
            st.subheader("📊 Ranking estatístico de associação", help=H_ESTAT)
            st.dataframe(stat, use_container_width=True, hide_index=True)
            g1, g2 = st.columns(2)
            sp = stat.set_index("Variável")["Spearman r"]
            g1.plotly_chart(
                _bar_rank(sp.abs().sort_values(ascending=False),
                          "Ranking |Spearman|", "|Spearman r|"),
                use_container_width=True, key="v2_bar_spearman",
            )
            mi = stat.set_index("Variável")["Mutual Information"].sort_values(ascending=False)
            g2.plotly_chart(
                _bar_rank(mi, "Ranking Mutual Information", "Mutual Information",
                          color="Greens"),
                use_container_width=True, key="v2_bar_mi",
            )

            st.subheader("🔎 Dispersão: variável × indicador sazonal")
            sel = st.selectbox("Variável", list(X.columns), key="v2_scatter_var")
            st.plotly_chart(_fig_scatter(X[sel], y, sel, "Indicador sazonal"),
                            use_container_width=True, key="v2_scatter")

            if st.checkbox("Mostrar matriz de correlação (top variáveis)",
                           key="v2_show_corr"):
                top_vars = stat["Variável"].head(12).tolist()
                sub = X[top_vars].join(y.rename("Sazonal"))
                st.plotly_chart(_fig_corr_matrix(sub), use_container_width=True,
                                key="v2_corr_matrix")

    # ===================================================================== #
    # ABA 4 — Modelo & Explicabilidade
    # ===================================================================== #
    with tab_model:
        st.plotly_chart(_fig_series(y, "Série temporal do indicador sazonal", "Indicador sazonal"),
                        use_container_width=True, key="v2_prod_series")
        if model_result is None:
            st.info("Modelo não foi treinado (escolha um tipo de análise com modelo).")
        else:
            m = model_result.metrics
            st.subheader(f"🤖 Modelo {model_result.model_name} — desempenho (teste)",
                         help="Métricas calculadas no conjunto de teste (períodos mais recentes, "
                              "respeitando a ordem temporal).")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE", f"{m['MAE']:.2f}", help=H_MAE)
            k2.metric("RMSE", f"{m['RMSE']:.2f}", help=H_RMSE)
            k3.metric("MAPE", "—" if np.isnan(m["MAPE"]) else f"{m['MAPE']:.1f}%", help=H_MAPE)
            k4.metric("R²", "—" if np.isnan(m["R2"]) else f"{m['R2']:.3f}", help=H_R2)

            pred_train = pd.Series(model_result.y_pred_train, index=model_result.y_train.index)
            pred_test = pd.Series(model_result.y_pred, index=model_result.y_test.index)
            st.plotly_chart(_fig_real_vs_pred(y, pred_train, pred_test),
                            use_container_width=True, key="v2_real_pred")

            cimp, cshap = st.columns(2)
            with cimp:
                st.subheader("Importância do modelo", help=H_IMP)
                st.plotly_chart(
                    _bar_rank(model_result.importances, "Importância do modelo", "Importância"),
                    use_container_width=True, key="v2_bar_imp",
                )
            with cshap:
                st.subheader("Ranking SHAP", help=H_SHAP)
                if shap_imp is not None:
                    st.plotly_chart(
                        _bar_rank(shap_imp, "Ranking SHAP (|impacto| médio)", "|SHAP| médio",
                                  color="Oranges"),
                        use_container_width=True, key="v2_bar_shap",
                    )
                else:
                    st.info("SHAP não disponível neste processamento (escolha 'Completa (com SHAP)').")

    # ===================================================================== #
    # ABA 5 — Ranking Consolidado + Diagnóstico
    # ===================================================================== #
    with tab_rank:
        if consolidated.empty:
            st.warning("Sem ranking consolidado disponível.")
        else:
            st.subheader("🏆 Ranking consolidado", help=H_CONS)
            st.caption(
                "Combina Spearman, Mutual Information, importância do modelo e SHAP "
                "(quando disponíveis), normalizados e somados no **Score consolidado**."
            )
            st.dataframe(consolidated.round(4), use_container_width=True, hide_index=True)
            st.plotly_chart(
                _bar_rank(consolidated.set_index("Variável")["Score consolidado"],
                          "Score consolidado", "Score", color="Purples"),
                use_container_width=True, key="v2_bar_consolidado",
            )
            st.subheader("🧭 Diagnóstico gerencial")
            st.info(res["diag"])

    # ===================================================================== #
    # ABA 6 — Diagnóstico do Período
    # ===================================================================== #
    with tab_day:
        st.subheader("📅 Diagnóstico de um período específico",
                     help="Mostra o valor real do indicador sazonal no período e as variáveis "
                          "que mais contribuíram para a previsão (SHAP quando disponível).")
        if model_result is None:
            st.info("Treine um modelo (tipo de análise com modelo) para o diagnóstico do período.")
        else:
            periodos = list(X.index)
            labels = [pd.Timestamp(d).strftime("%d/%m/%Y %H:%M") for d in periodos]
            idx = st.selectbox("Período", range(len(periodos)),
                               format_func=lambda i: labels[i], key="v2_day_sel")
            date = periodos[idx]

            real = float(y.loc[date])
            hist_mean = float(y.mean())
            d1, d2 = st.columns(2)
            d1.metric("Indicador sazonal no período", f"{real:.2f}",
                      delta=f"{real - hist_mean:+.2f} vs. média")
            d2.metric("Média histórica", f"{hist_mean:.2f}")

            contrib = analysis.day_contributions(model_result, X, date)
            st.caption(
                "Principais variáveis que mais contribuíram para a previsão deste período "
                "(SHAP quando disponível), comparando o valor do período com a média histórica."
            )
            st.dataframe(contrib, use_container_width=True, hide_index=True)

            if not contrib.empty:
                cc = contrib.set_index("Variável")["Contribuição"].iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=cc.values, y=cc.index, orientation="h",
                    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in cc.values],
                ))
                fig.update_layout(title="Contribuição por variável (período selecionado)",
                                  xaxis_title="Contribuição", height=max(320, 30 * len(cc)),
                                  margin=dict(l=10, r=10, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True, key="v2_day_contrib")

    # ===================================================================== #
    # ABA 7 — Exportar
    # ===================================================================== #
    with tab_export:
        st.subheader("⬇️ Exportar resultados (Excel)")
        sheets = {
            "Base por periodo": period_feat.round(4).reset_index(),
            "Ranking correlacoes": stat,
            "Ranking consolidado": consolidated.round(4),
        }
        if not stat.empty:
            sheets["Ranking MI"] = stat[["Variável", "Mutual Information"]].sort_values(
                "Mutual Information", ascending=False
            )
        if model_result is not None:
            sheets["Importancia modelo"] = (
                model_result.importances.rename("Importância")
                .reset_index().rename(columns={"index": "Variável"})
            )
        if shap_imp is not None:
            sheets["Ranking SHAP"] = (
                shap_imp.rename("SHAP médio").reset_index()
                .rename(columns={"index": "Variável"})
            )
        sheets["Diagnostico"] = res["diag"]

        st.download_button(
            "⬇️ Baixar resultados (Excel)",
            data=analysis.to_excel_multi(sheets),
            file_name="analise_sazonal_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="v2_download",
        )
        st.caption("O arquivo contém: base por período, rankings (correlações, MI, "
                   "importância, SHAP), ranking consolidado e diagnóstico.")
