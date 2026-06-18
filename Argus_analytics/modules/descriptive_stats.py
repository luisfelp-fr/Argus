"""
Módulo 3 — Estatística Descritiva
=================================

Resume o comportamento de uma ou mais variáveis numéricas: tendência central,
dispersão, quartis e contagens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.interpretation import interpret_descriptive
from utils.glossary import GLOSSARY, band_cards, cv_bands


def describe_series(s: pd.Series) -> dict:
    """Calcula o conjunto completo de estatísticas descritivas de uma série."""
    valid = pd.to_numeric(s, errors="coerce").dropna()
    n = len(valid)
    if n == 0:
        return {}
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if n > 1 else 0.0
    q1, med, q3 = (float(valid.quantile(q)) for q in (0.25, 0.5, 0.75))
    # O CV (dispersão relativa) só é confiável quando a média está claramente
    # afastada de zero. Perto de zero ele explode e perde significado, então o
    # tratamos como indefinido (NaN) — vale tanto para média exatamente 0 quanto
    # numericamente ~0 (ex.: variáveis simétricas em torno de zero).
    cv = std / mean * 100 if abs(mean) > 1e-9 * (std + 1e-12) else float("nan")
    return {
        "Média": mean,
        "Mediana": med,
        "Mínimo": float(valid.min()),
        "Máximo": float(valid.max()),
        "Amplitude": float(valid.max() - valid.min()),
        "Desvio padrão": std,
        "Variância": float(valid.var(ddof=1)) if n > 1 else 0.0,
        "Q1 (25%)": q1,
        "Q3 (75%)": q3,
        "Coef. variação (%)": cv,
        "N válidos": n,
        "N ausentes": int(s.isna().sum()),
    }


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "A estatística descritiva resume o comportamento dos dados. Ela mostra o valor "
        "típico, a dispersão e os limites observados para cada variável."
    )

    cols = st.multiselect(
        "Selecione uma ou mais variáveis numéricas:",
        numeric_cols(state),
        default=numeric_cols(state)[: min(3, len(numeric_cols(state)))],
    )
    if not cols:
        st.warning("Selecione ao menos uma variável.")
        return

    # Tabela consolidada
    table = pd.DataFrame({c: describe_series(df[c]) for c in cols}).T
    table = table.round(4)

    tabs = st.tabs(["📋 Tabela", "🧮 Cards", "📐 Faixas (CV)", "🗣️ Interpretação"])

    with tabs[0]:
        st.dataframe(table, use_container_width=True)

    with tabs[1]:
        for c in cols:
            st.markdown(f"**{c}**")
            stats = describe_series(df[c])
            if not stats:
                st.warning("Sem dados válidos.")
                continue
            m = st.columns(4)
            m[0].metric("Média", f"{stats['Média']:.4g}", help=GLOSSARY["media"])
            m[1].metric("Mediana", f"{stats['Mediana']:.4g}", help=GLOSSARY["mediana"])
            m[2].metric("Desvio padrão", f"{stats['Desvio padrão']:.4g}",
                        help=GLOSSARY["desvio_padrao"])
            _cv = stats["Coef. variação (%)"]
            m[3].metric("CV (%)", f"{_cv:.1f}" if _cv == _cv else "—", help=GLOSSARY["cv"])
            m2 = st.columns(4)
            m2[0].metric("Mínimo", f"{stats['Mínimo']:.4g}", help=GLOSSARY["amplitude"])
            m2[1].metric("Máximo", f"{stats['Máximo']:.4g}", help=GLOSSARY["amplitude"])
            m2[2].metric("N válidos", stats["N válidos"])
            m2[3].metric("N ausentes", stats["N ausentes"])
            st.divider()

    # Faixas de consideração do Coeficiente de Variação
    with tabs[2]:
        st.caption("Em qual faixa de **dispersão relativa (CV)** cada variável se "
                   "encontra? Passe o mouse no ícone ❓ para entender o conceito.")
        for c in cols:
            stats = describe_series(df[c])
            if not stats:
                continue
            st.markdown(f"**{c}**")
            cv = stats["Coef. variação (%)"]
            disp = f"{cv:.1f}%" if cv == cv else "—"
            band_cards("Coeficiente de Variação (CV)", disp, cv_bands(cv),
                       help_text=GLOSSARY["cv"])
            st.divider()

    interp_lines = []
    with tabs[3]:
        for c in cols:
            stats = describe_series(df[c])
            if stats:
                txt = interpret_descriptive(c, stats["Média"], stats["Mediana"],
                                            stats["Coef. variação (%)"])
                interp_lines.append(txt)
                st.markdown(f"- {txt}")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="desc_add"):
        item = make_report_item(
            name="Estatística Descritiva",
            variables={"variáveis": cols},
            params={},
            interpretation="\n".join(interp_lines),
            tables={"Estatísticas descritivas": table},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
