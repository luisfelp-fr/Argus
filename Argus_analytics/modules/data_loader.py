"""
Módulo 1 — Visão Geral dos Dados (upload e leitura)
===================================================

Permite importar Excel/CSV, escolher a aba, pré-visualizar a base e ver o
diagnóstico automático dos tipos de coluna e métricas-resumo.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.helpers import (
    list_sheets,
    read_file,
    detect_column_types,
)


def _load(file, sheet_name: str | None) -> None:
    """Lê o arquivo, detecta tipos e guarda tudo no session_state."""
    raw = read_file(file, sheet_name=sheet_name)
    df, col_types = detect_column_types(raw)
    st.session_state["df"] = df
    st.session_state["col_types"] = col_types
    st.session_state["file_name"] = getattr(file, "name", "arquivo")
    # Limpa qualquer tratamento de outliers de uma base anterior
    for k in ("df_original", "col_types_original", "df_no_outliers",
              "outliers_applied", "outliers_removed_log"):
        st.session_state.pop(k, None)


def render(state) -> None:
    st.info(
        "📂 Importe sua base de dados em **Excel** (.xlsx/.xls) ou **CSV**. "
        "O Argus identifica automaticamente os tipos de coluna e mostra um resumo."
    )

    file = st.file_uploader(
        "Selecione o arquivo", type=["xlsx", "xls", "csv"], accept_multiple_files=False
    )

    if file is None:
        if st.session_state.get("df") is not None:
            st.success(f"✅ Base atual: **{st.session_state.get('file_name')}**")
            _show_overview(st.session_state["df"], st.session_state["col_types"])
        return

    # Seleção de aba (Excel com múltiplas abas)
    sheet_name = None
    sheets = list_sheets(file)
    if len(sheets) > 1:
        sheet_name = st.selectbox("Selecione a aba (planilha):", sheets)

    if st.button("📥 Carregar dados", type="primary"):
        try:
            _load(file, sheet_name)
            st.success("✅ Dados carregados com sucesso!")
        except Exception as exc:
            st.error(f"Não foi possível ler o arquivo: {exc}")
            return

    if st.session_state.get("df") is not None:
        _show_overview(st.session_state["df"], st.session_state["col_types"])


def _show_overview(df: pd.DataFrame, col_types) -> None:
    tabs = st.tabs(["👁️ Prévia", "🔎 Tipos de Coluna", "📐 Resumo"])

    # --- Prévia ---------------------------------------------------------- #
    with tabs[0]:
        st.markdown("**Primeiras linhas da base:**")
        st.dataframe(df.head(50), use_container_width=True)
        st.caption(f"Exibindo até 50 de {len(df)} linhas.")

    # --- Tipos ----------------------------------------------------------- #
    with tabs[1]:
        st.markdown("Classificação automática de cada coluna:")
        type_rows = []
        for col in df.columns:
            type_rows.append({
                "Coluna": col,
                "Tipo detectado": col_types.kind_of(col),
                "Valores únicos": int(df[col].nunique(dropna=True)),
                "Ausentes": int(df[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)
        with st.expander("ℹ️ Como os tipos são identificados?"):
            st.markdown(
                "- **Numérica**: valores numéricos (inclusive com vírgula decimal BR).\n"
                "- **Data/Hora**: colunas que o sistema conseguiu converter em datas.\n"
                "- **Categórica**: texto com poucas categorias distintas (≤ 20).\n"
                "- **Texto**: texto livre com muitas categorias diferentes."
            )

    # --- Resumo ---------------------------------------------------------- #
    with tabs[2]:
        n_missing = int(df.isna().sum().sum())
        n_dup = int(df.duplicated().sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("🔢 Linhas", f"{len(df):,}".replace(",", "."))
        c2.metric("📊 Colunas", df.shape[1])
        c3.metric("❓ Dados ausentes", f"{n_missing:,}".replace(",", "."))
        c4, c5, c6 = st.columns(3)
        c4.metric("🔵 Colunas numéricas", len(col_types.numeric))
        c5.metric("🏷️ Colunas categóricas", len(col_types.categorical))
        c6.metric("👯 Linhas duplicadas", n_dup)

        c7, c8 = st.columns(2)
        c7.metric("📅 Colunas data/hora", len(col_types.datetime))
        c8.metric("📝 Colunas de texto", len(col_types.text))

        if n_missing > 0 or n_dup > 0:
            st.warning(
                "⚠️ A base tem dados ausentes e/ou duplicados. Recomendamos abrir "
                "**Qualidade dos Dados** para avaliar antes das análises."
            )
        else:
            st.success("✅ A base não apresenta dados ausentes nem duplicados óbvios.")
