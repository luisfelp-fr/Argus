"""
Modo "Produção Diária" — Versão 2
---------------------------------
Correlaciona o comportamento dos indicadores de processo (minuto a minuto) com a
produção informada uma vez ao dia, transformando os dados em variáveis diárias,
considerando efeitos de lag (em dias) e apontando os indicadores com maior
EVIDÊNCIA ESTATÍSTICA de influência (prováveis causas — sem causalidade absoluta).

Exposto por ``render_daily_mode()``, chamado pelo seletor de modo em ``app.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis


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
    fig.update_layout(title=title, xaxis_title="Data", yaxis_title=yname, height=360)
    return fig


def _fig_real_vs_pred(y_all: pd.Series, pred_train: pd.Series,
                      pred_test: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_all.index, y=y_all.values, mode="lines+markers",
                             name="Produção real", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=pred_train.index, y=pred_train.values, mode="lines",
                             name="Previsto (treino)", line=dict(color="#2ca02c", dash="dot")))
    fig.add_trace(go.Scatter(x=pred_test.index, y=pred_test.values, mode="lines+markers",
                             name="Previsto (teste)", line=dict(color="#d62728")))
    if len(pred_test):
        fig.add_vline(x=pred_test.index[0], line_dash="dash", line_color="gray",
                      annotation_text="início do teste", annotation_position="top")
    fig.update_layout(title="Produção real × prevista", xaxis_title="Data",
                      yaxis_title="Produção", height=420,
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
    fig.update_layout(title="Matriz de correlação (variáveis selecionadas + Produção)",
                      height=max(400, 32 * n), yaxis=dict(autorange="reversed"),
                      margin=dict(t=50, b=40))
    return fig


# --------------------------------------------------------------------------- #
# Fluxo principal
# --------------------------------------------------------------------------- #
def render_daily_mode() -> None:
    st.header("🏭 Produção Diária — evidência estatística de influência")
    st.caption(
        "Carregue a base de **processo minuto a minuto** e a base de **produção "
        "diária**. O sistema transforma os dados minuto a minuto em variáveis "
        "diárias, considera efeitos de **lag em dias** e aponta os indicadores com "
        "maior evidência estatística de influência sobre a produção."
    )

    tab_cfg, tab_base, tab_stat, tab_model, tab_rank, tab_day, tab_export = st.tabs([
        "📥 Dados & Configuração", "🧱 Base Diária", "📊 Estatística",
        "🤖 Modelo & Explicabilidade", "🏆 Ranking Consolidado",
        "📅 Diagnóstico do Dia", "⬇️ Exportar",
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
                "processo (DataHora + indicadores) e produção (Data + Produção). "
                "Em CSV, carregue o processo aqui e a produção logo abaixo."
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
                prod_sheet = st.selectbox("Aba de produção diária", sheets,
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
            st.info("CSV não possui abas — carregue a base de **produção diária** (CSV) abaixo.")
            prod_file = st.file_uploader("Base de produção diária (CSV)",
                                         type=["csv"], key="v2_prod_file")
            if prod_file is None:
                return
            prod_file.seek(0)
            prod_raw = analysis.read_sheet(prod_file)

        with st.expander("👁️ Prévia das bases", expanded=False):
            st.write("**Processo (minuto a minuto):**")
            st.dataframe(proc_raw.head(10), use_container_width=True)
            st.write("**Produção diária:**")
            st.dataframe(prod_raw.head(10), use_container_width=True)

        # --- Seleção de colunas ----------------------------------------- #
        st.subheader("⚙️ Colunas e parâmetros")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            dt_col = st.selectbox("Coluna de data/hora (processo)",
                                  list(proc_raw.columns), key="v2_dt_col")
        with cc2:
            prod_date_col = st.selectbox("Coluna de data (produção)",
                                         list(prod_raw.columns), key="v2_prod_date")
        with cc3:
            prod_num_cols = [c for c in prod_raw.columns if c != prod_date_col]
            target_col = st.selectbox("Coluna alvo de produção",
                                      prod_num_cols or list(prod_raw.columns),
                                      key="v2_target")

        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            max_lag = st.number_input("Máx. de dias de lag", min_value=0, max_value=30,
                                      value=3, step=1, key="v2_maxlag")
        with cp2:
            tipo = st.selectbox(
                "Tipo de análise",
                ["Completa (com SHAP)", "Estatística + Modelo", "Só estatística"],
                key="v2_tipo",
            )
        with cp3:
            model_options = ["RandomForest"] + (["XGBoost"] if analysis._HAS_XGB else [])
            model_type = st.radio("Modelo preditivo", model_options,
                                  horizontal=True, key="v2_model")
        use_turnos = st.toggle(
            "Criar variáveis por turno (00–08, 08–16, 16–24)", value=False,
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
                with st.spinner("Construindo base diária e analisando..."):
                    daily, step_min = analysis.build_daily_features(
                        proc_raw, dt_col, limits=limits, use_turnos=use_turnos
                    )
                    lagged = analysis.add_lag_features(daily, int(max_lag))
                    merged = analysis.merge_with_production(
                        lagged, prod_raw, prod_date_col, target_col
                    )
                    if merged.empty:
                        st.error(
                            "Nenhum dia em comum entre processo e produção após o "
                            "alinhamento de datas. Verifique as colunas de data."
                        )
                        return
                    clean, report = analysis.clean_modeling_table(merged, target="Producao")
                    X, y = analysis.split_features_target(clean, "Producao")

                    if X.shape[1] == 0:
                        st.error("Nenhuma variável explicativa restou após a limpeza.")
                        return

                    stat = analysis.statistical_ranking(X, y)

                    model_result = None
                    shap_imp = None
                    if run_model:
                        try:
                            model_result = analysis.train_model(
                                X, y, model_type=model_type
                            )
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
                    "daily": daily, "step_min": step_min, "report": report,
                    "X": X, "y": y, "stat": stat, "model": model_result,
                    "shap": shap_imp, "consolidated": consolidated, "diag": diag,
                    "n_days": len(clean), "n_features": X.shape[1],
                }
                st.success(
                    f"Base diária pronta: **{len(clean)} dias** × "
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

    daily = res["daily"]
    X, y = res["X"], res["y"]
    stat = res["stat"]
    model_result = res["model"]
    shap_imp = res["shap"]
    consolidated = res["consolidated"]

    # ===================================================================== #
    # ABA 2 — Base Diária Modelada
    # ===================================================================== #
    with tab_base:
        st.subheader("🧱 Base diária de indicadores")
        st.caption(
            f"Passo temporal detectado: **{res['step_min']:g} min**. "
            f"Cada linha é um dia; as colunas são as variáveis diárias por indicador "
            f"(antes da expansão de lag). Total modelado: **{res['n_days']} dias** × "
            f"**{res['n_features']} variáveis** (com lag)."
        )
        st.dataframe(daily.round(3), use_container_width=True)

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
            st.subheader("📊 Ranking estatístico de associação")
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

            st.subheader("🔎 Dispersão: variável × Produção")
            sel = st.selectbox("Variável", list(X.columns), key="v2_scatter_var")
            st.plotly_chart(_fig_scatter(X[sel], y, sel, "Produção"),
                            use_container_width=True, key="v2_scatter")

            if st.checkbox("Mostrar matriz de correlação (top variáveis)",
                           key="v2_show_corr"):
                top_vars = stat["Variável"].head(12).tolist()
                sub = X[top_vars].join(y.rename("Producao"))
                st.plotly_chart(_fig_corr_matrix(sub), use_container_width=True,
                                key="v2_corr_matrix")

    # ===================================================================== #
    # ABA 4 — Modelo & Explicabilidade
    # ===================================================================== #
    with tab_model:
        st.plotly_chart(_fig_series(y, "Série temporal da produção", "Produção"),
                        use_container_width=True, key="v2_prod_series")
        if model_result is None:
            st.info("Modelo não foi treinado (escolha um tipo de análise com modelo).")
        else:
            m = model_result.metrics
            st.subheader(f"🤖 Modelo {model_result.model_name} — desempenho (teste)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE", f"{m['MAE']:.2f}")
            k2.metric("RMSE", f"{m['RMSE']:.2f}")
            k3.metric("MAPE", "—" if np.isnan(m["MAPE"]) else f"{m['MAPE']:.1f}%")
            k4.metric("R²", "—" if np.isnan(m["R2"]) else f"{m['R2']:.3f}")

            pred_train = pd.Series(model_result.y_pred_train, index=model_result.y_train.index)
            pred_test = pd.Series(model_result.y_pred, index=model_result.y_test.index)
            st.plotly_chart(_fig_real_vs_pred(y, pred_train, pred_test),
                            use_container_width=True, key="v2_real_pred")

            cimp, cshap = st.columns(2)
            cimp.plotly_chart(
                _bar_rank(model_result.importances, "Importância do modelo", "Importância"),
                use_container_width=True, key="v2_bar_imp",
            )
            if shap_imp is not None:
                cshap.plotly_chart(
                    _bar_rank(shap_imp, "Ranking SHAP (|impacto| médio)", "|SHAP| médio",
                              color="Oranges"),
                    use_container_width=True, key="v2_bar_shap",
                )
            else:
                cshap.info("SHAP não disponível neste processamento.")

    # ===================================================================== #
    # ABA 5 — Ranking Consolidado + Diagnóstico
    # ===================================================================== #
    with tab_rank:
        if consolidated.empty:
            st.warning("Sem ranking consolidado disponível.")
        else:
            st.subheader("🏆 Ranking consolidado")
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
    # ABA 6 — Diagnóstico do Dia
    # ===================================================================== #
    with tab_day:
        st.subheader("📅 Diagnóstico de um dia específico")
        if model_result is None:
            st.info("Treine um modelo (tipo de análise com modelo) para o diagnóstico do dia.")
        else:
            dates = list(X.index)
            labels = [pd.Timestamp(d).strftime("%d/%m/%Y") for d in dates]
            idx = st.selectbox("Dia", range(len(dates)),
                               format_func=lambda i: labels[i], key="v2_day_sel")
            date = dates[idx]

            real = float(y.loc[date])
            hist_mean = float(y.mean())
            d1, d2 = st.columns(2)
            d1.metric("Produção real do dia", f"{real:.2f}",
                      delta=f"{real - hist_mean:+.2f} vs. média")
            d2.metric("Média histórica de produção", f"{hist_mean:.2f}")

            contrib = analysis.day_contributions(model_result, X, date)
            st.caption(
                "Principais variáveis que mais contribuíram para a previsão deste dia "
                "(SHAP quando disponível), comparando o valor do dia com a média histórica."
            )
            st.dataframe(contrib, use_container_width=True, hide_index=True)

            if not contrib.empty:
                cc = contrib.set_index("Variável")["Contribuição"].iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=cc.values, y=cc.index, orientation="h",
                    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in cc.values],
                ))
                fig.update_layout(title="Contribuição por variável (dia selecionado)",
                                  xaxis_title="Contribuição", height=max(320, 30 * len(cc)),
                                  margin=dict(l=10, r=10, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True, key="v2_day_contrib")

    # ===================================================================== #
    # ABA 7 — Exportar
    # ===================================================================== #
    with tab_export:
        st.subheader("⬇️ Exportar resultados (Excel)")
        sheets = {
            "Base modelada": daily.round(4).reset_index(),
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
            file_name="producao_diaria_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="v2_download",
        )
        st.caption("O arquivo contém: base modelada, rankings (correlações, MI, "
                   "importância, SHAP), ranking consolidado e diagnóstico.")
