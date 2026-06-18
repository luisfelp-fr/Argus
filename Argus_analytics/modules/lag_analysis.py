"""
Módulo 12 — Análise com Lag (defasagem temporal)
================================================

Verifica se variáveis explicativas afetam uma variável alvo após um atraso.
Calcula a correlação cruzada (cross-correlation) entre a alvo e cada explicativa
em diferentes defasagens e identifica o melhor lag por variável.

A lógica de cross-correlation foi adaptada do núcleo já testado da ferramenta
original (``version 2/analysis.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import (
    numeric_cols, datetime_cols, make_report_item, add_to_report, now_str,
)
from utils.plotting import lag_curve, ranking_bar, real_vs_pred, residuals_plot
from utils.interpretation import interpret_lag, interpret_lag_validation
from utils.glossary import GLOSSARY
from utils.timeseries import validate_lag, _HAS_SM


def cross_correlation(key: pd.Series, other: pd.Series, max_lag: int,
                      min_overlap: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Pearson entre ``key`` e ``other`` para cada lag inteiro.

    lag > 0 -> ``other`` é atrasado -> testa se ``other`` ANTECEDE ``key``.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        pair = pd.concat([key, other.shift(lag)], axis=1).dropna()
        if len(pair) < min_overlap:
            continue
        a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        corrs[i] = float(np.corrcoef(a, b)[0, 1])
    return lags, corrs


def best_lag(lags: np.ndarray, corrs: np.ndarray) -> tuple[int, float]:
    if np.all(np.isnan(corrs)):
        return 0, float("nan")
    idx = int(np.nanargmax(np.abs(corrs)))
    return int(lags[idx]), float(corrs[idx])


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 2):
        return
    if not validation.require_datetime(state):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "A análise com lag verifica se uma variável de processo afeta outra após um "
        "determinado atraso. Útil em processos industriais, onde uma alteração pode "
        "demorar minutos ou horas para gerar impacto."
    )

    num = numeric_cols(state)
    c1, c2 = st.columns(2)
    target = c1.selectbox("Variável alvo (efeito):", num,
                          help="A variável cujo comportamento você quer explicar.")
    dt_col = c2.selectbox("Coluna de data/hora:", datetime_cols(state))
    explan = st.multiselect(
        "Variáveis explicativas (possíveis causas):",
        [c for c in num if c != target],
        default=[c for c in num if c != target][: min(5, len(num) - 1)],
        help=GLOSSARY["cross_correlation"],
    )

    c3, c4 = st.columns(2)
    step_min = c3.number_input("Intervalo do lag (minutos por passo):", min_value=1, value=5,
                               help=GLOSSARY["lag"])
    max_lag_min = c4.number_input("Lag máximo a testar (minutos):", min_value=int(step_min),
                                  value=int(step_min) * 12, step=int(step_min),
                                  help="Maior atraso (em minutos) que será testado entre causa e efeito.")

    if not explan:
        st.warning("Selecione ao menos uma variável explicativa.")
        return

    if st.button("▶ Executar análise de lag", type="primary", key="lag_run"):
        st.session_state["lag_has_run"] = True
        st.session_state.pop("lag_arimax", None)  # invalida validação ARIMAX anterior
    if not st.session_state.get("lag_has_run"):
        st.caption("Configure os parâmetros e clique em **Executar análise de lag**.")
        return

    # Reamostra numa grade regular pelo passo escolhido
    work = df[[dt_col] + [target] + explan].copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    grid = work.resample(f"{int(step_min)}min").mean().interpolate(limit=3, limit_area="inside")

    max_lag_steps = max(1, int(max_lag_min // step_min))
    rows, curves = [], {}
    for col in explan:
        lags, corrs = cross_correlation(grid[target], grid[col], max_lag_steps)
        bl_steps, bl_corr = best_lag(lags, corrs)
        rows.append({
            "Variável": col,
            "Melhor lag (min)": bl_steps * step_min,
            "Correlação no melhor lag": round(bl_corr, 4) if not np.isnan(bl_corr) else np.nan,
            "|Correlação|": round(abs(bl_corr), 4) if not np.isnan(bl_corr) else np.nan,
        })
        curves[col] = (lags * step_min, corrs, bl_steps * step_min)

    ranking = pd.DataFrame(rows).sort_values("|Correlação|", ascending=False).reset_index(drop=True)

    tabs = st.tabs(["🏆 Ranking", "📈 Curva de lag", "🔬 Validação ARIMAX",
                    "🗣️ Interpretação"])

    with tabs[0]:
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        valid = ranking.dropna(subset=["Correlação no melhor lag"])
        fig_rank = None
        if not valid.empty:
            fig_rank = ranking_bar(list(valid["Variável"]),
                                   list(valid["Correlação no melhor lag"]),
                                   title="Variáveis mais relacionadas ao alvo (no melhor lag)",
                                   xlabel="Correlação")
            st.plotly_chart(fig_rank, use_container_width=True, key="lag_rank")

    with tabs[1]:
        sel = st.selectbox("Ver curva de correlação por lag para:", explan, key="lag_sel")
        lags_min, corrs, bl = curves[sel]
        fig_curve = lag_curve(lags_min, corrs, best_lag=bl)
        st.plotly_chart(fig_curve, use_container_width=True, key="lag_curve")
        st.caption("Lag positivo: a variável explicativa antecede o alvo (possível causa→efeito).")

    with tabs[2]:
        st.caption("Confirme estatisticamente o lag detectado: o **ARIMAX** modela a "
                   "dinâmica própria do alvo e inclui a variável explicativa defasada; "
                   "o **Ljung-Box** verifica se os resíduos são ruído branco. Isso "
                   "separa um efeito real de uma correlação espúria por tendência.")
        if not _HAS_SM:
            st.info("A validação ARIMAX requer o pacote **statsmodels** (já listado "
                    "em requirements.txt). A correlação cruzada acima continua "
                    "disponível normalmente.")
        else:
            top_var = str(ranking["Variável"].iloc[0]) if not ranking.empty else explan[0]
            default_var_idx = explan.index(top_var) if top_var in explan else 0
            av_var = st.selectbox("Variável a validar:", explan, index=default_var_idx,
                                  key="lag_av_var", help=GLOSSARY["arimax"])

            # Melhor lag (em passos) da variável escolhida — atualiza ao trocar de variável
            row = ranking[ranking["Variável"] == av_var]
            bl_min = float(row["Melhor lag (min)"].iloc[0]) if not row.empty else 0.0
            default_steps = int(round(bl_min / step_min)) if step_min else 0
            if st.session_state.get("lag_av_var_prev") != av_var:
                st.session_state["lag_av_lag"] = int(default_steps)
                st.session_state["lag_av_var_prev"] = av_var
            av_lag_steps = st.number_input(
                f"Lag a validar (passos · 1 passo = {int(step_min)} min):",
                step=1, key="lag_av_lag", help=GLOSSARY["lag"])

            auto = st.checkbox("Ordem ARIMA automática (por AIC)", value=True,
                               key="lag_av_auto", help=GLOSSARY["arimax"])
            order = None
            if not auto:
                oc = st.columns(3)
                p = oc[0].number_input("p (AR)", min_value=0, max_value=5, value=1, key="lag_av_p")
                d = oc[1].number_input("d (I)", min_value=0, max_value=2, value=0, key="lag_av_d")
                q = oc[2].number_input("q (MA)", min_value=0, max_value=5, value=0, key="lag_av_q")
                order = (int(p), int(d), int(q))

            if st.button("🔬 Validar lag (ARIMAX + Ljung-Box)", key="lag_av_run"):
                with st.spinner("Ajustando modelo ARIMAX…"):
                    av = validate_lag(grid[target], grid[av_var], int(av_lag_steps),
                                      step_min=float(step_min), order=order)
                st.session_state["lag_arimax"] = (av_var, av)

            stored = st.session_state.get("lag_arimax")
            if stored is not None:
                av_var_s, av = stored
                if not av.ok:
                    st.error(f"Não foi possível validar: {av.error}")
                else:
                    cols = st.columns(5)
                    cols[0].metric("Coef. exógena", f"{av.exog_coef:.4g}",
                                   help=GLOSSARY["arimax_coef"])
                    cols[1].metric("p-valor", f"{av.exog_p:.4f}", help=GLOSSARY["p_valor"])
                    cols[2].metric("AIC", f"{av.aic:.1f}",
                                   help="Critério de Akaike: quanto menor, melhor o ajuste relativo.")
                    cols[3].metric("Ljung-Box p", f"{av.lb_p:.4f}", help=GLOSSARY["ljung_box"])
                    cols[4].metric("Veredito", "✅ Validado" if av.validated else "⚠️ Revisar")
                    st.caption(f"ARIMA(p,d,q) = {av.order} · {av.n_obs} pontos · "
                               f"lag testado = {av.lag_min:g} min · variável: {av_var_s}")

                    av_interp = interpret_lag_validation(
                        av_var_s, target, av.lag_min, av.exog_coef, av.exog_p, av.lb_p)
                    (st.success if av.validated else st.warning)(av_interp)

                    g = st.columns(2)
                    with g[0]:
                        st.plotly_chart(
                            real_vs_pred(av.actual, av.fitted, title="Real vs. ajustado (ARIMAX)"),
                            use_container_width=True, key="lag_av_rp")
                    with g[1]:
                        st.plotly_chart(
                            residuals_plot(av.fitted, av.resid, title="Resíduos do ARIMAX"),
                            use_container_width=True, key="lag_av_res")
            else:
                st.caption("Escolha a variável e o lag e clique em **Validar lag**.")

    interp_lines = []
    with tabs[3]:
        for _, r in ranking.head(3).iterrows():
            if not pd.isna(r["Correlação no melhor lag"]):
                txt = interpret_lag(r["Variável"], target,
                                    r["Melhor lag (min)"], r["Correlação no melhor lag"])
                interp_lines.append(txt)
                st.markdown(f"- {txt}")
        if not interp_lines:
            st.warning("Não foi possível estabelecer relações de lag significativas.")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="lag_add"):
        figs = []
        valid = ranking.dropna(subset=["Correlação no melhor lag"])
        if not valid.empty:
            figs.append(ranking_bar(list(valid["Variável"]),
                                    list(valid["Correlação no melhor lag"]),
                                    title="Variáveis mais relacionadas ao alvo", xlabel="Correlação"))

        params = {"passo_min": int(step_min), "lag_max_min": int(max_lag_min)}
        interp = "\n".join(interp_lines) if interp_lines else "Análise de lag executada."
        stored = st.session_state.get("lag_arimax")
        if stored is not None and stored[1].ok:
            av_var_s, av = stored
            params.update({
                "ARIMAX_variável": av_var_s,
                "ARIMAX_ordem": str(av.order),
                "ARIMAX_lag_min": av.lag_min,
                "ARIMAX_coef": round(av.exog_coef, 5),
                "ARIMAX_p_valor": round(av.exog_p, 5),
                "Ljung_Box_p": round(av.lb_p, 5),
                "ARIMAX_validado": bool(av.validated),
            })
            interp += "\n\n**Validação ARIMAX:** " + interpret_lag_validation(
                av_var_s, target, av.lag_min, av.exog_coef, av.exog_p, av.lb_p)

        item = make_report_item(
            name="Análise com Lag",
            variables={"alvo": target, "explicativas": explan, "tempo": dt_col},
            params=params,
            interpretation=interp,
            figures=figs,
            tables={"Ranking de lag": ranking},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
