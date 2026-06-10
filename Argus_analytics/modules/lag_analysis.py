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
from utils.plotting import lag_curve, ranking_bar
from utils.interpretation import interpret_lag
from utils.glossary import GLOSSARY


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

    if not st.button("▶ Executar análise de lag", type="primary", key="lag_run"):
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

    tabs = st.tabs(["🏆 Ranking", "📈 Curva de lag", "🗣️ Interpretação"])

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

    interp_lines = []
    with tabs[2]:
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
        item = make_report_item(
            name="Análise com Lag",
            variables={"alvo": target, "explicativas": explan, "tempo": dt_col},
            params={"passo_min": int(step_min), "lag_max_min": int(max_lag_min)},
            interpretation="\n".join(interp_lines) if interp_lines else "Análise de lag executada.",
            figures=figs,
            tables={"Ranking de lag": ranking},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
