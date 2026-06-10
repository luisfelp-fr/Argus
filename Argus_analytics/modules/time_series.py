"""
Módulo 11 — Análise Temporal
============================

Evolução de uma variável ao longo do tempo: agregação por minuto/hora/dia/
semana/mês, média móvel configurável e gráfico de tendência.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import (
    numeric_cols, datetime_cols, make_report_item, add_to_report, now_str,
)
from utils.plotting import time_trend

FREQ_MAP = {
    "Minuto": "min",
    "Hora": "h",
    "Dia": "D",
    "Semana": "W",
    "Mês": "ME",
}
AGG_MAP = {"Média": "mean", "Soma": "sum", "Mediana": "median",
           "Máximo": "max", "Mínimo": "min"}


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return
    if not validation.require_datetime(state):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "A análise temporal permite observar como uma variável evolui ao longo do tempo, "
        "ajudando a identificar tendências, oscilações e períodos anormais."
    )

    c1, c2, c3 = st.columns(3)
    value_col = c1.selectbox("Variável numérica:", numeric_cols(state))
    dt_col = c2.selectbox("Coluna de data/hora:", datetime_cols(state))
    freq_label = c3.selectbox("Agregação por:", list(FREQ_MAP.keys()), index=2)

    c4, c5 = st.columns(2)
    agg_label = c4.selectbox("Função de agregação:", list(AGG_MAP.keys()))
    window = c5.slider("Janela da média móvel (períodos):", 1, 30, 7)

    work = df[[dt_col, value_col]].dropna().copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    if work.empty:
        st.warning("Não há dados de data/hora válidos.")
        return

    agg = work[value_col].resample(FREQ_MAP[freq_label]).agg(AGG_MAP[agg_label]).dropna()
    if len(agg) < 2:
        st.warning("Período/agregação resultou em poucos pontos. Tente outra granularidade.")
        return

    rolling = agg.rolling(window=window, min_periods=1).mean() if window > 1 else None

    fig = time_trend(agg.index.to_series(), agg, rolling,
                     title=f"{agg_label} de {value_col} por {freq_label.lower()}")
    st.plotly_chart(fig, use_container_width=True, key="ts_trend")

    # Comparação temporal simples (primeiro vs. último terço)
    n = len(agg)
    first = agg.iloc[: n // 3].mean()
    last = agg.iloc[-n // 3:].mean()
    delta = last - first
    m = st.columns(3)
    m[0].metric("Início (1º terço)", f"{first:.4g}")
    m[1].metric("Fim (último terço)", f"{last:.4g}")
    m[2].metric("Variação", f"{delta:+.4g}",
                delta=f"{(100*delta/first if first else 0):+.1f}%")

    if abs(delta) > 0.05 * (abs(first) + 1e-9):
        trend = "alta" if delta > 0 else "queda"
        interp = (f"A variável **{value_col}** apresenta tendência de **{trend}** ao longo "
                  f"do período (variação de {delta:+.4g} entre o início e o fim).")
        st.warning(interp)
    else:
        interp = (f"A variável **{value_col}** permaneceu relativamente **estável** ao longo "
                  "do período analisado.")
        st.success(interp)

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="ts_add"):
        item = make_report_item(
            name="Análise Temporal",
            variables={"variável": value_col, "tempo": dt_col},
            params={"agregação": f"{agg_label}/{freq_label}", "janela_média_móvel": window},
            interpretation=interp,
            figures=[fig],
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
