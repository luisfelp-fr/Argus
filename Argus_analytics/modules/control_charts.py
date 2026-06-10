"""
Módulo 9 — Controle Estatístico de Processo (CEP)
=================================================

Carta de controle de valores individuais (carta I): linha central (média) e
limites de controle a ±3 sigma estimado pela amplitude móvel (MR-bar / 1,128).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import (
    numeric_cols, datetime_cols, make_report_item, add_to_report, now_str,
)
from utils.plotting import control_chart
from utils.interpretation import interpret_control_chart
from utils.glossary import GLOSSARY

D2_N2 = 1.128  # constante d2 para amplitude móvel de tamanho 2


def individuals_limits(values: np.ndarray) -> tuple[float, float, float]:
    """Calcula (centro, LSC, LIC) de uma carta de individuais."""
    center = float(np.mean(values))
    mr = np.abs(np.diff(values))
    mr_bar = float(np.mean(mr)) if len(mr) else 0.0
    sigma = mr_bar / D2_N2 if mr_bar > 0 else float(np.std(values, ddof=1))
    ucl = center + 3 * sigma
    lcl = center - 3 * sigma
    return center, ucl, lcl


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "A carta de controle ajuda a verificar se o processo está estável. Pontos fora "
        "dos limites de controle podem indicar causas especiais de variação."
    )

    c1, c2 = st.columns(2)
    value_col = c1.selectbox("Variável numérica (medida do processo):", numeric_cols(state))

    order_options = ["(Ordem das linhas)"] + datetime_cols(state)
    order_col = c2.selectbox("Eixo X (ordem ou data/hora):", order_options)

    work = df[[value_col]].copy()
    if order_col != "(Ordem das linhas)":
        work[order_col] = df[order_col]
        work = work.dropna().sort_values(order_col)
        x = work[order_col]
    else:
        work = work.dropna().reset_index(drop=True)
        x = pd.Series(range(len(work)), name="Observação")

    series = pd.to_numeric(work[value_col], errors="coerce").dropna()
    if len(series) < 4:
        st.warning("São necessárias ao menos 4 observações para a carta de controle.")
        return

    center, ucl, lcl = individuals_limits(series.to_numpy())
    out_mask = (series.to_numpy() > ucl) | (series.to_numpy() < lcl)
    n_out = int(out_mask.sum())

    fig = control_chart(series, center, ucl, lcl, x=x.reset_index(drop=True))

    m = st.columns(4)
    m[0].metric("Média (LC)", f"{center:.4g}", help=GLOSSARY["cep"])
    m[1].metric("LSC (+3σ)", f"{ucl:.4g}", help=GLOSSARY["limites_controle"])
    m[2].metric("LIC (−3σ)", f"{lcl:.4g}", help=GLOSSARY["limites_controle"])
    m[3].metric("Pontos fora", n_out,
                help="Pontos fora dos limites de controle — possíveis causas especiais.")

    st.plotly_chart(fig, use_container_width=True, key="cep_chart")

    interp = interpret_control_chart(n_out, len(series))
    if n_out == 0:
        st.success(interp)
    else:
        st.warning(interp)
        out_table = pd.DataFrame({
            "Posição": np.where(out_mask)[0],
            "Valor": series.to_numpy()[out_mask],
        })
        with st.expander("Ver pontos fora de controle"):
            st.dataframe(out_table, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="cep_add"):
        item = make_report_item(
            name="Controle Estatístico de Processo",
            variables={"variável": value_col, "eixo_x": order_col},
            params={"LC": round(center, 4), "LSC": round(ucl, 4), "LIC": round(lcl, 4),
                    "pontos_fora": n_out},
            interpretation=interp,
            figures=[fig],
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
