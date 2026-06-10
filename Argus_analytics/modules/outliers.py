"""
Módulo 13 — Outliers
====================

Detecta valores atípicos em uma variável numérica por três métodos: IQR,
Z-score e Desvio Absoluto Mediano (MAD). Permite visualizar e, opcionalmente,
remover os outliers da base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import boxplot, outlier_scatter
from utils.interpretation import interpret_outliers


def detect_iqr(s: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def detect_zscore(s: np.ndarray, thr: float = 3.0) -> np.ndarray:
    mu, sigma = np.mean(s), np.std(s, ddof=1)
    if sigma == 0:
        return np.zeros(len(s), dtype=bool)
    return np.abs((s - mu) / sigma) > thr


def detect_mad(s: np.ndarray, thr: float = 3.5) -> np.ndarray:
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if mad == 0:
        return np.zeros(len(s), dtype=bool)
    modified_z = 0.6745 * (s - med) / mad
    return np.abs(modified_z) > thr


METHODS = {
    "IQR (1,5×amplitude interquartílica)": detect_iqr,
    "Z-score (|z| > 3)": detect_zscore,
    "MAD (desvio absoluto mediano)": detect_mad,
}


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "Outliers são valores muito diferentes do comportamento geral dos dados. Eles "
        "podem representar erro de medição, falha de processo ou eventos reais importantes."
    )

    c1, c2 = st.columns(2)
    col = c1.selectbox("Variável numérica:", numeric_cols(state))
    method_label = c2.selectbox("Método de detecção:", list(METHODS.keys()))

    series = validation.clean_numeric(df[col], min_n=4)
    if series is None:
        return
    data = series.to_numpy()

    mask = METHODS[method_label](data)
    n_out = int(mask.sum())
    method_short = method_label.split(" ")[0]

    m = st.columns(3)
    m[0].metric("Total de valores", len(data))
    m[1].metric("Outliers detectados", n_out)
    m[2].metric("% outliers", f"{(100*n_out/len(data)):.1f}%")

    tabs = st.tabs(["📦 Boxplot", "🔵 Dispersão", "📋 Outliers", "🗣️ Interpretação"])

    fig_box = boxplot(series)
    fig_sc = outlier_scatter(series, mask)
    with tabs[0]:
        st.plotly_chart(fig_box, use_container_width=True, key="out_box")
    with tabs[1]:
        st.plotly_chart(fig_sc, use_container_width=True, key="out_scatter")
        st.caption("Pontos em vermelho são os outliers detectados pelo método escolhido.")

    out_table = pd.DataFrame({
        "Posição": series.index[mask],
        "Valor": data[mask],
    })
    with tabs[2]:
        if n_out > 0:
            st.dataframe(out_table, use_container_width=True, hide_index=True)
        else:
            st.success("Nenhum outlier detectado.")

    interp = interpret_outliers(n_out, len(data), method_short)
    with tabs[3]:
        if n_out == 0:
            st.success(interp)
        else:
            st.warning(interp)

    # Opção de remover outliers (cria cópia limpa em session)
    if n_out > 0:
        st.divider()
        if st.checkbox("🧹 Criar cópia da base **sem** estes outliers (não altera o original)"):
            clean_idx = series.index[~mask]
            cleaned = df.loc[df.index.isin(clean_idx) | df[col].isna()].copy()
            st.session_state["df_no_outliers"] = cleaned
            st.success(f"Cópia sem outliers criada ({len(cleaned)} linhas). "
                       "Disponível em `session_state['df_no_outliers']`.")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="out_add"):
        item = make_report_item(
            name="Análise de Outliers",
            variables={"variável": col},
            params={"método": method_short, "outliers": n_out},
            interpretation=interp,
            figures=[fig_box, fig_sc],
            tables={"Outliers detectados": out_table} if n_out > 0 else {},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
