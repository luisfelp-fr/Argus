"""
Módulo 8 — Comparação entre Grupos
==================================

Compara uma variável numérica entre os níveis de uma variável categórica.
Sugere e aplica o teste adequado (t / Mann-Whitney para 2 grupos; ANOVA /
Kruskal-Wallis para 3+), com boxplot e tabela-resumo por grupo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import (
    numeric_cols, categorical_cols, make_report_item, add_to_report, now_str,
)
from utils.plotting import grouped_box
from utils.interpretation import interpret_group_test
from utils.glossary import GLOSSARY


def _groups(df: pd.DataFrame, value_col: str, group_col: str) -> list[np.ndarray]:
    out = []
    for _, sub in df.groupby(group_col, observed=True):
        arr = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy()
        if len(arr) >= 2:
            out.append(arr)
    return out


def _all_normal(groups: list[np.ndarray]) -> bool:
    """Verifica se todos os grupos passam em Shapiro (aprox.) — decide paramétrico."""
    from scipy.stats import shapiro
    for g in groups:
        if len(g) < 3:
            return False
        try:
            if shapiro(g[:5000])[1] <= 0.05:
                return False
        except Exception:
            return False
    return True


def render(state) -> None:
    if not validation.require_data():
        return
    if not validation.require_numeric(state, 1) or not validation.require_categorical(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "Esta análise verifica se diferentes grupos apresentam comportamento "
        "estatisticamente diferente. Ex.: comparar produção entre turnos, máquinas, "
        "operadores ou fornecedores."
    )

    c1, c2 = st.columns(2)
    value_col = c1.selectbox("Variável numérica:", numeric_cols(state))
    group_col = c2.selectbox("Variável de agrupamento (categórica):", categorical_cols(state))

    sub = df[[value_col, group_col]].dropna()
    n_groups = sub[group_col].nunique()
    if n_groups < 2:
        st.warning("A variável de grupo precisa ter ao menos 2 categorias com dados.")
        return

    groups = _groups(sub, value_col, group_col)
    if len(groups) < 2:
        st.warning("Não há grupos suficientes com dados válidos (mínimo 2 por grupo).")
        return

    # Sugestão automática
    parametric_ok = _all_normal(groups)
    if n_groups == 2:
        suggested = "Teste t" if parametric_ok else "Mann-Whitney"
        options = ["Teste t", "Mann-Whitney"]
    else:
        suggested = "ANOVA" if parametric_ok else "Kruskal-Wallis"
        options = ["ANOVA", "Kruskal-Wallis"]

    st.caption(f"💡 Com {n_groups} grupos e "
               f"{'distribuições aproximadamente normais' if parametric_ok else 'normalidade não confirmada'}, "
               f"o teste sugerido é **{suggested}**.")
    _test_help = {
        "Teste t": GLOSSARY["teste_t"], "Mann-Whitney": GLOSSARY["mann_whitney"],
        "ANOVA": GLOSSARY["anova"], "Kruskal-Wallis": GLOSSARY["kruskal"],
    }
    test_name = st.radio(
        "Teste estatístico:", options, index=options.index(suggested), horizontal=True,
        help="Paramétricos (t, ANOVA) assumem normalidade; não paramétricos "
             "(Mann-Whitney, Kruskal-Wallis) não exigem. " + GLOSSARY["p_valor"])

    # Tabela resumo por grupo
    summary = sub.groupby(group_col, observed=True)[value_col].agg(
        N="count", Média="mean", Mediana="median", **{"Desvio padrão": "std"}
    ).round(4).reset_index()

    # Executa o teste
    stat, p = _run_test(test_name, groups)

    tabs = st.tabs(["📦 Boxplot", "📋 Resumo por grupo", "🧪 Teste", "🗣️ Interpretação"])

    fig = grouped_box(sub, value_col, group_col)
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True, key="grp_box")
    with tabs[1]:
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with tabs[2]:
        m = st.columns(3)
        m[0].metric("Teste", test_name, help=_test_help.get(test_name))
        m[1].metric("Estatística", f"{stat:.4f}" if not np.isnan(stat) else "—",
                    help="Valor da estatística do teste; sozinho não conclui — use o p-valor.")
        m[2].metric("p-valor", f"{p:.4f}" if not np.isnan(p) else "—", help=GLOSSARY["p_valor"])

    interp = interpret_group_test(test_name, p)
    with tabs[3]:
        if not np.isnan(p) and p <= 0.05:
            st.success(interp)
        else:
            st.warning(interp)

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="grp_add"):
        item = make_report_item(
            name="Comparação entre Grupos",
            variables={"variável": value_col, "grupo": group_col},
            params={"teste": test_name, "p_valor": round(p, 5) if not np.isnan(p) else None},
            interpretation=interp,
            figures=[fig],
            tables={"Resumo por grupo": summary},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")


def _run_test(test_name: str, groups: list[np.ndarray]) -> tuple[float, float]:
    from scipy import stats
    try:
        if test_name == "Teste t":
            s, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        elif test_name == "Mann-Whitney":
            s, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        elif test_name == "ANOVA":
            s, p = stats.f_oneway(*groups)
        else:  # Kruskal-Wallis
            s, p = stats.kruskal(*groups)
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")
