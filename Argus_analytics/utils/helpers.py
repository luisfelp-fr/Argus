"""
helpers.py — Utilitários centrais do Argus Analytics
====================================================

Reúne funções de:
  * leitura de arquivos (Excel/CSV) com suporte a decimal brasileiro;
  * detecção automática do tipo de cada coluna (numérica, categórica,
    data/hora, texto);
  * acesso rápido aos subconjuntos de colunas;
  * inicialização e manipulação do ``st.session_state``;
  * adição de análises ao relatório final.

A lógica de leitura/coerção numérica foi adaptada do núcleo já testado da
ferramenta original (``version 2/analysis.py``), mantendo este projeto
independente.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

# Fuso de Brasília para os carimbos de data/hora do relatório. Importante porque
# servidores (ex.: Streamlit Community Cloud) rodam em UTC — sem isso o relatório
# mostraria o horário errado. Usa a base de fusos (DST histórico) quando
# disponível e cai para o offset fixo −03:00 caso o tzdata não exista.
try:
    from zoneinfo import ZoneInfo
    _BR_TZ = ZoneInfo("America/Sao_Paulo")
except Exception:  # pragma: no cover - ambiente sem tzdata
    _BR_TZ = timezone(timedelta(hours=-3))

try:  # streamlit é opcional para permitir testes/compilação isolados
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False


# --------------------------------------------------------------------------- #
# 1. Leitura de arquivos
# --------------------------------------------------------------------------- #
def list_sheets(file) -> list[str]:
    """Retorna os nomes das abas de um arquivo Excel (vazio se for CSV)."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return []
    try:
        xls = pd.ExcelFile(file, engine="openpyxl")
        return list(xls.sheet_names)
    except Exception:
        return []


def coerce_numeric_br(series: pd.Series) -> pd.Series:
    """Tenta converter uma série de texto para número tratando o formato BR.

    Trata separador de milhar ``.`` e vírgula decimal ``,`` (ex.: ``"1.234,56"``
    vira ``1234.56``). Se a conversão falhar para a maioria dos valores, retorna
    a série original (assume que não é numérica).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series

    raw = series.astype(str).str.strip()
    cleaned = (
        raw.str.replace(".", "", regex=False)   # separador de milhar
        .str.replace(",", ".", regex=False)      # vírgula decimal -> ponto
    )
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Só aceita como numérica se converteu a grande maioria dos não-nulos.
    non_null = series.notna().sum()
    if non_null > 0 and converted.notna().sum() >= 0.8 * non_null:
        return converted
    return series


def read_file(file, sheet_name: str | None = None) -> pd.DataFrame:
    """Lê um arquivo .xlsx/.xls/.csv para um DataFrame *cru* (sem forçar índice).

    O CSV usa autodetecção de separador (``sep=None``) para suportar ``;`` (BR)
    e ``,``. Datas e números são tratados depois em :func:`detect_column_types`.
    """
    name = getattr(file, "name", str(file)).lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file, sep=None, engine="python")
        except Exception:
            # fallback robusto para arquivos problemáticos
            file.seek(0)
            df = pd.read_csv(file, sep=";", engine="python", encoding="latin-1")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError(
            "Formato não suportado. Use arquivos .xlsx, .xls ou .csv."
        )

    if isinstance(df, dict):  # múltiplas abas retornadas
        first = next(iter(df.values()))
        df = first

    # Remove colunas/linhas completamente vazias
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Detecção de tipos de coluna
# --------------------------------------------------------------------------- #
@dataclass
class ColumnTypes:
    """Resultado da classificação automática das colunas."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def kind_of(self, col: str) -> str:
        if col in self.numeric:
            return "Numérica"
        if col in self.datetime:
            return "Data/Hora"
        if col in self.categorical:
            return "Categórica"
        return "Texto"


def _looks_datetime(series: pd.Series) -> bool:
    """Heurística: a série parece ser data/hora?"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    return parsed.notna().mean() >= 0.8


def detect_column_types(
    df: pd.DataFrame, max_categorical_unique: int = 20
) -> tuple[pd.DataFrame, ColumnTypes]:
    """Classifica cada coluna e devolve ``(df_convertido, ColumnTypes)``.

    Aplica conversões *in-place* numa cópia: datas viram ``datetime64`` e
    números em texto BR viram ``float``. A classificação categórica usa baixa
    cardinalidade como critério.
    """
    out = df.copy()
    types = ColumnTypes()

    for col in out.columns:
        series = out[col]

        # 1) Já é datetime64? (caso o arquivo já traga o tipo)
        if pd.api.types.is_datetime64_any_dtype(series):
            types.datetime.append(col)
            continue

        # 2) Numérica primeiro (inclui texto BR convertível). Evita que números
        #    com vírgula decimal — ex.: "50,23" — sejam confundidos com datas.
        converted = coerce_numeric_br(series)
        if pd.api.types.is_numeric_dtype(converted):
            out[col] = converted
            types.numeric.append(col)
            continue

        # 3) Data/hora (apenas para colunas não numéricas)
        if _looks_datetime(series):
            out[col] = pd.to_datetime(series, errors="coerce", dayfirst=True)
            types.datetime.append(col)
            continue

        # 4) Categórica vs. texto livre, por cardinalidade
        n_unique = series.nunique(dropna=True)
        if n_unique <= max_categorical_unique:
            types.categorical.append(col)
        else:
            types.text.append(col)

    return out, types


# --------------------------------------------------------------------------- #
# 3. Acesso rápido a subconjuntos de colunas
# --------------------------------------------------------------------------- #
def numeric_cols(state) -> list[str]:
    return list(state.get("col_types").numeric) if state.get("col_types") else []


def categorical_cols(state) -> list[str]:
    return list(state.get("col_types").categorical) if state.get("col_types") else []


def datetime_cols(state) -> list[str]:
    return list(state.get("col_types").datetime) if state.get("col_types") else []


# --------------------------------------------------------------------------- #
# 4. Session state
# --------------------------------------------------------------------------- #
def init_session() -> None:
    """Garante que as chaves usadas pelo app existam no ``session_state``."""
    if not _HAS_ST:
        return
    defaults = {
        "page": "home",
        "df": None,             # DataFrame convertido (tipos detectados)
        "col_types": None,      # ColumnTypes
        "file_name": None,      # nome do arquivo carregado
        "report_items": [],     # lista de dicts (análises adicionadas)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_df():
    """Atalho seguro para o DataFrame atual (ou None)."""
    if not _HAS_ST:
        return None
    return st.session_state.get("df")


def has_data() -> bool:
    df = get_df()
    return df is not None and not df.empty


def add_to_report(item: dict[str, Any]) -> None:
    """Adiciona uma análise ao relatório final (lista em session_state)."""
    if not _HAS_ST:
        return
    st.session_state.setdefault("report_items", [])
    st.session_state["report_items"].append(item)


def make_report_item(
    name: str,
    variables: dict[str, Any],
    params: dict[str, Any],
    interpretation: str,
    figures: list | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Cria o dicionário padronizado de uma análise para o relatório."""
    return {
        "name": name,
        "variables": variables or {},
        "params": params or {},
        "interpretation": interpretation or "",
        "figures": figures or [],
        "tables": tables or {},
        "timestamp": timestamp or "",
    }


def now_str() -> str:
    """Data/hora atual no fuso de **Brasília**, formatada (pt-BR).

    Independe do fuso do servidor onde o app roda (ex.: UTC na nuvem).
    """
    return datetime.now(_BR_TZ).strftime("%d/%m/%Y %H:%M:%S")


# --------------------------------------------------------------------------- #
# 5. Diversos
# --------------------------------------------------------------------------- #
def df_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Serializa um dicionário {nome_aba: DataFrame} em bytes de um .xlsx."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            safe = str(sheet_name)[:31]  # limite de nome de aba do Excel
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=safe, index=True)
            else:
                pd.DataFrame({"info": [str(data)]}).to_excel(
                    writer, sheet_name=safe, index=False
                )
    return buffer.getvalue()


def safe_dropna_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Alinha duas séries removendo pares com NaN. Retorna arrays numpy."""
    pair = pd.concat([x, y], axis=1).dropna()
    return pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
