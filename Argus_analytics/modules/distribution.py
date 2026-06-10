"""
Módulo 4 — Distribuição dos Dados
=================================

Histograma, boxplot, densidade (KDE) e resumo para uma variável numérica, com
nota automática de assimetria e outliers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import histogram, boxplot, density
from utils.interpretation import interpret_skewness


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "O histograma mostra como os valores estão distribuídos. Ele ajuda a identificar "
        "concentração de dados, assimetria, dispersão e possíveis comportamentos não normais."
    )

    col = st.selectbox("Selecione a variável numérica:", numeric_cols(state))
    series = validation.clean_numeric(df[col], min_n=3)
    if series is None:
        return

    nbins = st.slider("Número de classes (bins) do histograma:", 5, 80, 30)

    from scipy.stats import skew, kurtosis
    sk = float(skew(series))
    ku = float(kurtosis(series))

    tabs = st.tabs(["📊 Histograma", "📦 Boxplot", "🌊 Densidade", "🗣️ Resumo"])

    fig_h = histogram(series, nbins=nbins)
    fig_b = boxplot(series)
    fig_d = density(series)

    with tabs[0]:
        st.plotly_chart(fig_h, use_container_width=True, key="dist_hist")
    with tabs[1]:
        st.plotly_chart(fig_b, use_container_width=True, key="dist_box")
        st.caption("Pontos em vermelho são possíveis outliers (regra de 1,5×IQR).")
    with tabs[2]:
        st.plotly_chart(fig_d, use_container_width=True, key="dist_dens")

    skew_txt = interpret_skewness(sk)
    with tabs[3]:
        c = st.columns(4)
        c[0].metric("Média", f"{series.mean():.4g}")
        c[1].metric("Mediana", f"{series.median():.4g}")
        c[2].metric("Assimetria", f"{sk:.2f}")
        c[3].metric("Curtose", f"{ku:.2f}")
        st.markdown(f"- {skew_txt}")
        if ku > 1:
            st.markdown("- Curtose elevada: distribuição com **pico acentuado** e/ou caudas pesadas.")
        elif ku < -1:
            st.markdown("- Curtose baixa: distribuição **achatada** em relação à normal.")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="dist_add"):
        item = make_report_item(
            name="Distribuição dos Dados",
            variables={"variável": col},
            params={"bins": nbins},
            interpretation=f"{skew_txt} (assimetria={sk:.2f}, curtose={ku:.2f}).",
            figures=[fig_h, fig_b, fig_d],
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
