"""
Módulo 10 — Capabilidade Cp / Cpk
=================================

Calcula os índices de capabilidade do processo a partir dos limites de
especificação (LIE/LSE) informados pelo usuário.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import capability_hist
from utils.interpretation import interpret_capability, capability_class


def compute_capability(data: np.ndarray, lie: float | None, lse: float | None) -> dict:
    mean = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    out = {"média": mean, "desvio": sigma, "Cp": None, "Cpk": None, "Cpu": None, "Cpl": None}
    if sigma <= 0:
        return out
    if lie is not None and lse is not None:
        out["Cp"] = (lse - lie) / (6 * sigma)
    if lse is not None:
        out["Cpu"] = (lse - mean) / (3 * sigma)
    if lie is not None:
        out["Cpl"] = (mean - lie) / (3 * sigma)
    cpks = [v for v in (out["Cpu"], out["Cpl"]) if v is not None]
    if cpks:
        out["Cpk"] = min(cpks)
    return out


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "O Cp avalia se a variação do processo cabe dentro dos limites de especificação. "
        "O Cpk também considera se o processo está centralizado."
    )

    col = st.selectbox("Variável numérica:", numeric_cols(state))
    series = validation.clean_numeric(df[col], min_n=5)
    if series is None:
        return
    data = series.to_numpy()

    auto_min, auto_max = float(data.min()), float(data.max())
    c1, c2 = st.columns(2)
    use_lie = c1.checkbox("Definir Limite Inferior (LIE)", value=True)
    use_lse = c2.checkbox("Definir Limite Superior (LSE)", value=True)
    lie = c1.number_input("LIE — Limite Inferior de Especificação",
                          value=round(auto_min, 4)) if use_lie else None
    lse = c2.number_input("LSE — Limite Superior de Especificação",
                          value=round(auto_max, 4)) if use_lse else None

    if lie is None and lse is None:
        st.warning("Informe ao menos um limite de especificação (LIE ou LSE).")
        return
    if lie is not None and lse is not None and lie >= lse:
        st.error("O LIE deve ser menor que o LSE.")
        return

    res = compute_capability(data, lie, lse)

    m = st.columns(4)
    m[0].metric("Média", f"{res['média']:.4g}")
    m[1].metric("Desvio padrão", f"{res['desvio']:.4g}")
    m[2].metric("Cp", f"{res['Cp']:.2f}" if res["Cp"] is not None else "—")
    m[3].metric("Cpk", f"{res['Cpk']:.2f}" if res["Cpk"] is not None else "—")

    fig = capability_hist(series, lie, lse)
    st.plotly_chart(fig, use_container_width=True, key="cap_hist")

    interp = ""
    if res["Cp"] is not None and res["Cpk"] is not None:
        interp = interpret_capability(res["Cp"], res["Cpk"])
        cls = capability_class(res["Cpk"])
        if res["Cpk"] >= 1.33:
            st.success(interp)
        elif res["Cpk"] >= 1.0:
            st.warning(interp)
        else:
            st.error(interp)
    elif res["Cpk"] is not None:
        cls = capability_class(res["Cpk"])
        interp = (f"Com apenas um limite, o índice Cpk = {res['Cpk']:.2f} classifica o "
                  f"processo como **{cls}**.")
        st.warning(interp)

    with st.expander("ℹ️ Como interpretar os índices"):
        st.markdown(
            "- **< 1,00**: processo potencialmente incapaz\n"
            "- **1,00 – 1,33**: processo marginal\n"
            "- **> 1,33**: processo geralmente capaz\n"
            "- **> 1,67**: processo robusto"
        )

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="cap_add"):
        item = make_report_item(
            name="Capabilidade Cp/Cpk",
            variables={"variável": col},
            params={"LIE": lie, "LSE": lse,
                    "Cp": round(res["Cp"], 3) if res["Cp"] is not None else None,
                    "Cpk": round(res["Cpk"], 3) if res["Cpk"] is not None else None},
            interpretation=interp,
            figures=[fig],
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
