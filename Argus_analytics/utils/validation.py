"""
validation.py — Validações amigáveis antes de cada análise
===========================================================

Funções que verificam pré-requisitos (existência de dados, número mínimo de
linhas, presença de colunas numéricas/datetime) e exibem mensagens claras via
Streamlit. Retornam ``True`` quando é seguro prosseguir.
"""

from __future__ import annotations

import pandas as pd

try:
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False

from .helpers import get_df, numeric_cols, datetime_cols, categorical_cols


def _warn(msg: str) -> None:
    if _HAS_ST:
        st.warning(msg)


def require_data() -> bool:
    """Garante que há um DataFrame carregado."""
    df = get_df()
    if df is None or df.empty:
        _warn("📂 Nenhum dado carregado. Vá em **Visão Geral dos Dados** e "
              "importe um arquivo Excel ou CSV antes de analisar.")
        return False
    return True


def require_min_rows(n: int = 3) -> bool:
    df = get_df()
    if df is None or len(df.dropna(how="all")) < n:
        _warn(f"São necessárias ao menos **{n} observações válidas** para esta análise.")
        return False
    return True


def require_numeric(state, min_cols: int = 1) -> bool:
    """Garante que existam pelo menos ``min_cols`` colunas numéricas."""
    cols = numeric_cols(state)
    if len(cols) < min_cols:
        _warn(f"Esta análise precisa de pelo menos **{min_cols} coluna(s) numérica(s)**. "
              "Verifique a detecção de tipos na tela de dados.")
        return False
    return True


def require_categorical(state, min_cols: int = 1) -> bool:
    cols = categorical_cols(state)
    if len(cols) < min_cols:
        _warn("Esta análise precisa de pelo menos **1 coluna categórica** (de grupo). "
              "Nenhuma foi detectada na sua base.")
        return False
    return True


def require_datetime(state) -> bool:
    cols = datetime_cols(state)
    if len(cols) < 1:
        _warn("Esta análise precisa de uma **coluna de data/hora**. "
              "Nenhuma foi detectada — verifique o formato das datas na sua base.")
        return False
    return True


def clean_numeric(series: pd.Series, min_n: int = 3) -> pd.Series | None:
    """Remove NaN e valida tamanho mínimo. Retorna None (com aviso) se insuficiente."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < min_n:
        _warn(f"A variável **{series.name}** tem apenas {len(s)} valores válidos "
              f"(mínimo {min_n}). Resultado pode não ser confiável.")
        return None
    return s
