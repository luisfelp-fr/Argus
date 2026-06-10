"""
Módulo 6 — Correlação
=====================

Matriz de correlação (Pearson/Spearman/Kendall), heatmap, dispersão entre duas
variáveis, ranking das correlações mais fortes e interpretação automática.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import corr_heatmap, scatter, ranking_bar
from utils.interpretation import interpret_correlation, correlation_strength

METHODS = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}


def strongest_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Extrai o ranking dos pares de variáveis por |correlação| (triângulo superior)."""
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if not np.isnan(r):
                rows.append({
                    "Variável A": cols[i],
                    "Variável B": cols[j],
                    "Correlação": round(float(r), 4),
                    "|Correlação|": round(abs(float(r)), 4),
                    "Força": correlation_strength(r),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("|Correlação|", ascending=False).reset_index(drop=True)
    return out


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 2):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "Use esta análise para verificar se duas ou mais variáveis se movimentam juntas. "
        "Útil para identificar possíveis relações entre indicadores de processo."
    )
    st.warning(
        "⚠️ **Correlação não significa causalidade.** Ela indica associação entre "
        "variáveis, mas não prova que uma variável causa a outra."
    )

    num = numeric_cols(state)
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        chosen = st.multiselect("Variáveis numéricas:", num,
                                default=num[: min(6, len(num))])
    with col2:
        method_label = st.radio("Método de correlação:", list(METHODS.keys()), horizontal=False)
    method = METHODS[method_label]

    if len(chosen) < 2:
        st.warning("Selecione ao menos duas variáveis.")
        return

    corr = df[chosen].corr(method=method)

    tabs = st.tabs(["🔥 Heatmap", "🏆 Ranking", "🔬 Dispersão", "🗣️ Interpretação"])

    fig_heat = corr_heatmap(corr, title=f"Matriz de Correlação ({method_label})")
    with tabs[0]:
        st.plotly_chart(fig_heat, use_container_width=True, key="corr_heat")
        st.dataframe(corr.round(3), use_container_width=True)

    ranking = strongest_pairs(corr)
    fig_rank = None
    with tabs[1]:
        if ranking.empty:
            st.warning("Não foi possível calcular correlações (variância nula?).")
        else:
            st.dataframe(ranking, use_container_width=True, hide_index=True)
            top = ranking.head(10)
            labels = [f"{a} × {b}" for a, b in zip(top["Variável A"], top["Variável B"])]
            fig_rank = ranking_bar(labels, list(top["Correlação"]),
                                   title="Correlações mais fortes", xlabel="Correlação")
            st.plotly_chart(fig_rank, use_container_width=True, key="corr_rank")

    fig_sc = None
    with tabs[2]:
        cc1, cc2 = st.columns(2)
        vx = cc1.selectbox("Variável X:", chosen, index=0, key="corr_vx")
        vy = cc2.selectbox("Variável Y:", chosen,
                           index=1 if len(chosen) > 1 else 0, key="corr_vy")
        if vx != vy:
            fig_sc = scatter(df[vx], df[vy])
            st.plotly_chart(fig_sc, use_container_width=True, key="corr_scatter")
            r = corr.loc[vx, vy]
            st.metric(f"Correlação {method_label} ({vx} × {vy})", f"{r:.3f}")

    interp = ""
    with tabs[3]:
        if not ranking.empty:
            top = ranking.iloc[0]
            interp = interpret_correlation(top["Variável A"], top["Variável B"],
                                           top["Correlação"], method_label)
            st.markdown(f"**Par mais forte:** {interp}")
            st.caption("Veja o ranking completo na aba correspondente.")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="corr_add"):
        figs = [f for f in (fig_heat, fig_rank, fig_sc) if f is not None]
        item = make_report_item(
            name="Correlação",
            variables={"variáveis": chosen},
            params={"método": method_label},
            interpretation=interp or "Matriz de correlação calculada.",
            figures=figs,
            tables={"Matriz de correlação": corr.round(3),
                    "Ranking de pares": ranking},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
