"""
Módulo 5 — Teste de Normalidade
===============================

Aplica Shapiro-Wilk e Anderson-Darling a uma variável numérica, com conclusão
automática e recomendação paramétrica/não paramétrica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import histogram_with_normal, qq_plot
from utils.interpretation import interpret_normality
from utils.glossary import GLOSSARY


def shapiro_test(data: np.ndarray) -> tuple[float, float]:
    from scipy.stats import shapiro

    # Shapiro é confiável até ~5000 amostras; acima disso, usa amostra aleatória fixa.
    if len(data) > 5000:
        rng = np.random.default_rng(0)
        data = rng.choice(data, 5000, replace=False)
    stat, p = shapiro(data)
    return float(stat), float(p)


def anderson_test(data: np.ndarray):
    from scipy.stats import anderson

    result = anderson(data, dist="norm")
    # Usa o nível de significância de 5% para a conclusão
    idx5 = list(result.significance_level).index(5.0) if 5.0 in result.significance_level else 2
    crit = float(result.critical_values[idx5])
    stat = float(result.statistic)
    normal = stat < crit  # estatística abaixo do crítico => não rejeita normalidade
    return stat, crit, normal


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 1):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "O teste de normalidade verifica se os dados seguem distribuição normal — "
        "premissa de muitos testes paramétricos. Usamos α = 0,05 como referência."
    )

    col = st.selectbox("Selecione a variável numérica:", numeric_cols(state))
    series = validation.clean_numeric(df[col], min_n=3)
    if series is None:
        return
    data = series.to_numpy()

    # Shapiro-Wilk
    sw_stat, sw_p = shapiro_test(data)
    # Anderson-Darling
    try:
        ad_stat, ad_crit, ad_normal = anderson_test(data)
    except Exception:
        ad_stat = ad_crit = float("nan")
        ad_normal = None

    tabs = st.tabs(["📋 Resultados", "📊 Gráficos", "🗣️ Conclusão"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Shapiro-Wilk** ", help=GLOSSARY["shapiro"])
            st.metric("Estatística W", f"{sw_stat:.4f}", help=GLOSSARY["shapiro"])
            st.metric("p-valor", f"{sw_p:.4f}", help=GLOSSARY["p_valor"])
        with c2:
            st.markdown("**Anderson-Darling** ", help=GLOSSARY["anderson"])
            st.metric("Estatística A²", f"{ad_stat:.4f}" if not np.isnan(ad_stat) else "—",
                      help=GLOSSARY["anderson"])
            st.metric("Valor crítico (5%)", f"{ad_crit:.4f}" if not np.isnan(ad_crit) else "—",
                      help=GLOSSARY["anderson"])
        res_table = pd.DataFrame({
            "Teste": ["Shapiro-Wilk", "Anderson-Darling"],
            "Estatística": [round(sw_stat, 4), round(ad_stat, 4)],
            "p-valor / crítico": [round(sw_p, 4), round(ad_crit, 4)],
            "Conclusão (α=0,05)": [
                "Normal" if sw_p > 0.05 else "Não normal",
                ("Normal" if ad_normal else "Não normal") if ad_normal is not None else "—",
            ],
        })

    fig_h = histogram_with_normal(series)
    fig_q = qq_plot(series)
    with tabs[1]:
        st.plotly_chart(fig_h, use_container_width=True, key="norm_hist")
        st.plotly_chart(fig_q, use_container_width=True, key="norm_qq")
        st.caption("No Q-Q, quanto mais os pontos seguem a linha, mais normal é a distribuição.")

    conclusion = interpret_normality(sw_p)
    with tabs[2]:
        if sw_p > 0.05:
            st.success(conclusion)
        else:
            st.warning(conclusion)
        if ad_normal is not None:
            agree = (sw_p > 0.05) == ad_normal
            if agree:
                st.caption("✅ Shapiro-Wilk e Anderson-Darling concordam.")
            else:
                st.caption("⚠️ Os dois testes divergem — avalie com cautela (tamanho amostral pode influenciar).")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="norm_add"):
        item = make_report_item(
            name="Teste de Normalidade",
            variables={"variável": col},
            params={"alpha": 0.05},
            interpretation=conclusion,
            figures=[fig_h, fig_q],
            tables={"Testes de normalidade": res_table},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
