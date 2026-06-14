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
from utils.plotting import capability_hist, histogram
from utils.interpretation import interpret_capability, capability_class
from utils.glossary import GLOSSARY
from utils.transforms import ad_normality, evaluate_transforms, transform_spec


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
    use_lie = c1.checkbox("Definir Limite Inferior (LIE)", value=True, help=GLOSSARY["lie_lse"])
    use_lse = c2.checkbox("Definir Limite Superior (LSE)", value=True, help=GLOSSARY["lie_lse"])
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

    # ------------------------------------------------------------------ #
    # 1) Verificação de normalidade (Anderson-Darling)
    # ------------------------------------------------------------------ #
    st.markdown("#### 1) Verificação de normalidade")
    st.caption("A capabilidade Cp/Cpk assume que os dados são **normais**. Se não "
               "forem, é preciso transformá-los antes de calcular os índices.")
    a2, p_norm = ad_normality(data)
    nm = st.columns(3)
    nm[0].metric("Anderson-Darling A²", f"{a2:.3f}" if a2 == a2 else "—",
                 help=GLOSSARY["anderson"])
    nm[1].metric("p-valor", f"{p_norm:.4f}" if p_norm == p_norm else "—",
                 help=GLOSSARY["p_valor"])
    is_normal = bool(p_norm == p_norm and p_norm > 0.05)
    nm[2].metric("Distribuição", "Normal" if is_normal else "Não normal")

    # Dados/limites de trabalho (podem ser transformados)
    work_data, work_lie, work_lse = data, lie, lse
    transform_name = "Nenhuma (dados originais)"
    transformed = False
    work_series_name = col

    if is_normal:
        st.success("✅ Dados aproximadamente **normais** — a capabilidade pode ser "
                   "calculada diretamente, sem transformação.")
    else:
        st.warning("⚠️ Dados **não normais**. Avalie uma transformação abaixo antes "
                   "de calcular a capabilidade.")
        # -------------------------------------------------------------- #
        # 2) Testar todas as transformações e ranquear por normalidade
        # -------------------------------------------------------------- #
        st.markdown("#### 2) Transformações para normalizar os dados")
        results = evaluate_transforms(data)
        if not results:
            st.error("Não foi possível ajustar nenhuma transformação a estes dados "
                     "(verifique se há valores suficientes e válidos).")
        else:
            rank_df = pd.DataFrame([{
                "Transformação": r.name,
                "p-valor (AD)": round(r.ad_p, 4),
                "Normaliza? (p > 0,05)": "✅ Sim" if r.ad_p > 0.05 else "❌ Não",
                "Parâmetros": r.params,
            } for r in results])
            st.caption("Quanto maior o **p-valor (AD)** após transformar, mais próximos "
                       "da normalidade ficam os dados (p > 0,05 = normal).")
            st.dataframe(rank_df, use_container_width=True, hide_index=True)

            recommended = results[0]  # já vem ordenado pelo melhor p-valor
            if recommended.ad_p > 0.05:
                st.info(f"💡 Mais viável: **{recommended.name}** "
                        f"(p-valor AD = {recommended.ad_p:.4f}). Foi a que melhor "
                        "normalizou os dados.")
            else:
                st.warning("Nenhuma transformação atingiu p > 0,05. A mais próxima foi "
                           f"**{recommended.name}** (p = {recommended.ad_p:.4f}). "
                           "Considere mais dados ou avaliar uma distribuição não normal.")

            # 3) Usuário escolhe qual transformação usar
            st.markdown("#### 3) Escolha a transformação")
            options = ["Nenhuma (usar dados originais)"] + [r.name for r in results]
            default_idx = 1  # a recomendada (results[0])
            choice = st.selectbox("Transformação a aplicar na capabilidade:", options,
                                  index=default_idx,
                                  help="A escolhida é aplicada aos dados e também aos "
                                       "limites de especificação (LIE/LSE).")
            if choice != "Nenhuma (usar dados originais)":
                chosen = next(r for r in results if r.name == choice)
                t_lie, t_lse = transform_spec(chosen, lie, lse)
                if (lie is not None and t_lie is None) or (lse is not None and t_lse is None):
                    st.error("Os limites de especificação ficam fora do domínio desta "
                             "transformação. Escolha outra transformação.")
                else:
                    work_data = chosen.transformed
                    work_lie, work_lse = t_lie, t_lse
                    transform_name = chosen.name
                    transformed = True
                    work_series_name = f"{col} (transformado)"
                    _, p_after = ad_normality(work_data)
                    st.success(f"Aplicada **{chosen.name}**. Normalidade após "
                               f"transformar: p-valor AD = {p_after:.4f} "
                               f"({'normal' if p_after > 0.05 else 'ainda não normal'}).")

    # ------------------------------------------------------------------ #
    # 4) Capabilidade (no espaço escolhido)
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("#### 4) Capabilidade Cp / Cpk"
                + ("  ·  *escala transformada*" if transformed else ""))
    work_series = pd.Series(work_data, name=work_series_name)
    res = compute_capability(work_data, work_lie, work_lse)

    if transformed:
        st.caption(f"Índices calculados sobre os dados transformados por "
                   f"**{transform_name}**, com os limites convertidos para a mesma "
                   "escala — procedimento padrão para capabilidade com dados não normais.")
    elif not is_normal:
        st.caption("⚠️ Calculado sobre os dados **originais não normais** — "
                   "os índices podem ser imprecisos. Prefira uma transformação acima.")

    m = st.columns(4)
    m[0].metric("Média", f"{res['média']:.4g}", help=GLOSSARY["media"])
    m[1].metric("Desvio padrão", f"{res['desvio']:.4g}", help=GLOSSARY["desvio_padrao"])
    m[2].metric("Cp", f"{res['Cp']:.2f}" if res["Cp"] is not None else "—", help=GLOSSARY["cp"])
    m[3].metric("Cpk", f"{res['Cpk']:.2f}" if res["Cpk"] is not None else "—", help=GLOSSARY["cpk"])

    fig = capability_hist(work_series, work_lie, work_lse)
    st.plotly_chart(fig, use_container_width=True, key="cap_hist")
    if transformed:
        with st.expander("Ver histograma na escala original"):
            st.plotly_chart(histogram(series, title=f"Distribuição original — {col}"),
                            use_container_width=True, key="cap_hist_orig")

    interp = ""
    norm_note = (f" Os dados eram não normais (p AD = {p_norm:.4f}) e foram "
                 f"transformados por **{transform_name}** antes do cálculo."
                 if transformed else
                 (f" Atenção: dados não normais (p AD = {p_norm:.4f}) sem transformação."
                  if not is_normal else
                  f" Dados aproximadamente normais (p AD = {p_norm:.4f})."))
    if res["Cp"] is not None and res["Cpk"] is not None:
        interp = interpret_capability(res["Cp"], res["Cpk"]) + norm_note
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
                  f"processo como **{cls}**." + norm_note)
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
            variables={"variável": col, "transformação": transform_name},
            params={"LIE": lie, "LSE": lse,
                    "normalidade_p": round(p_norm, 4) if p_norm == p_norm else None,
                    "Cp": round(res["Cp"], 3) if res["Cp"] is not None else None,
                    "Cpk": round(res["Cpk"], 3) if res["Cpk"] is not None else None},
            interpretation=interp,
            figures=[fig],
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
