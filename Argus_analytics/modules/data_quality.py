"""
Módulo 2 — Qualidade dos Dados
==============================

Diagnóstico da base: valores ausentes, duplicidades, colunas constantes,
possíveis problemas de formatação e outliers, com sugestões de tratamento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import ARGUS_DANGER, ARGUS_SECONDARY, _apply_theme


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = 100 * miss / len(df) if len(df) else miss * 0
    return pd.DataFrame({
        "Coluna": df.columns,
        "Ausentes": miss.values,
        "% Ausentes": pct.round(1).values,
    }).sort_values("Ausentes", ascending=False).reset_index(drop=True)


def _iqr_outlier_count(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def render(state) -> None:
    if not validation.require_data():
        return

    df: pd.DataFrame = state["df"]
    col_types = state["col_types"]

    st.info(
        "Antes de realizar análises estatísticas, é importante avaliar a qualidade "
        "dos dados. Dados faltantes, duplicados ou mal formatados podem distorcer os "
        "resultados."
    )

    tabs = st.tabs(["❓ Ausentes", "👯 Duplicidades", "⚠️ Problemas", "💡 Sugestões"])

    # --- Ausentes -------------------------------------------------------- #
    with tabs[0]:
        miss = _missing_table(df)
        total_missing = int(miss["Ausentes"].sum())
        st.metric("Total de células ausentes", f"{total_missing:,}".replace(",", "."))
        st.dataframe(miss, use_container_width=True, hide_index=True)

        with_missing = miss[miss["Ausentes"] > 0]
        if not with_missing.empty:
            fig = go.Figure(go.Bar(
                x=with_missing["% Ausentes"], y=with_missing["Coluna"],
                orientation="h", marker_color=ARGUS_DANGER,
                text=with_missing["% Ausentes"].astype(str) + "%", textposition="auto"))
            fig.update_xaxes(title="% de ausentes")
            _apply_theme(fig, "Percentual de dados faltantes por coluna")
            st.plotly_chart(fig, use_container_width=True, key="dq_missing")
        else:
            st.success("✅ Não há valores ausentes na base.")

    # --- Duplicidades ---------------------------------------------------- #
    with tabs[1]:
        n_dup = int(df.duplicated().sum())
        c1, c2 = st.columns(2)
        c1.metric("Linhas duplicadas", n_dup)
        c2.metric("% duplicadas", f"{(100*n_dup/len(df) if len(df) else 0):.1f}%")
        if n_dup > 0:
            st.warning(f"⚠️ Existem **{n_dup} linhas duplicadas** (idênticas). "
                       "Considere removê-las para não enviesar contagens e médias.")
            st.dataframe(df[df.duplicated(keep=False)].head(50), use_container_width=True)
        else:
            st.success("✅ Nenhuma linha totalmente duplicada encontrada.")

    # --- Problemas ------------------------------------------------------- #
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    text_in_numeric = []
    outlier_summary = []
    with tabs[2]:
        st.markdown("**Colunas constantes** (um único valor — não informam nada):")
        if constant_cols:
            st.warning("⚠️ " + ", ".join(f"`{c}`" for c in constant_cols))
        else:
            st.success("✅ Nenhuma coluna constante.")

        st.markdown("**Possíveis outliers** (método IQR, colunas numéricas):")
        for col in numeric_cols(state):
            n_out = _iqr_outlier_count(df[col])
            if n_out > 0:
                outlier_summary.append({"Coluna": col, "Possíveis outliers": n_out})
        if outlier_summary:
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
            st.caption("Veja a tela **Outliers** para detalhar e tratar cada variável.")
        else:
            st.success("✅ Nenhum outlier evidente pelo critério IQR.")

        st.markdown("**Tipos de dados detectados:**")
        st.write({
            "Numéricas": len(col_types.numeric),
            "Categóricas": len(col_types.categorical),
            "Data/Hora": len(col_types.datetime),
            "Texto": len(col_types.text),
        })

    # --- Sugestões ------------------------------------------------------- #
    suggestions = _build_suggestions(df, constant_cols, outlier_summary)
    with tabs[3]:
        for s in suggestions:
            st.markdown(f"- {s}")
        st.divider()
        if st.button("➕ Adicionar diagnóstico ao relatório", key="dq_add"):
            interp = " ".join(suggestions) if suggestions else "Base sem problemas relevantes de qualidade."
            item = make_report_item(
                name="Qualidade dos Dados",
                variables={"colunas": df.shape[1], "linhas": len(df)},
                params={},
                interpretation=interp,
                tables={"Valores ausentes": _missing_table(df)},
                timestamp=now_str(),
            )
            add_to_report(item)
            st.success("Diagnóstico adicionado ao relatório!")


def _build_suggestions(df, constant_cols, outlier_summary) -> list[str]:
    s = []
    total_missing = int(df.isna().sum().sum())
    if total_missing > 0:
        s.append("Trate os **valores ausentes** (remoção de linhas, imputação por média/mediana "
                 "ou interpolação) conforme o contexto.")
    if int(df.duplicated().sum()) > 0:
        s.append("Remova as **linhas duplicadas** para evitar contagens e médias enviesadas.")
    if constant_cols:
        s.append(f"Considere **descartar colunas constantes** ({', '.join(constant_cols)}) — "
                 "não contribuem para as análises.")
    if outlier_summary:
        s.append("Investigue os **possíveis outliers** antes de modelar — podem ser erros ou eventos reais.")
    if not s:
        s.append("✅ A base aparenta boa qualidade. Você pode seguir para as análises com segurança.")
    return s
