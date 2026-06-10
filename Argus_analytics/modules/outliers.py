"""
Módulo 13 — Outliers
====================

Detecta valores atípicos em uma variável numérica por três métodos: IQR,
Z-score e Desvio Absoluto Mediano (MAD). Permite visualizar e, opcionalmente,
remover os outliers da base.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import boxplot, outlier_scatter
from utils.interpretation import interpret_outliers
from utils.glossary import GLOSSARY


def _to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serializa um DataFrame em bytes de um .xlsx (sem coluna de índice)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sem outliers", index=False)
    return buffer.getvalue()


def apply_clean_base(cleaned: pd.DataFrame, col: str, n_removed: int) -> None:
    """Passa a usar a base sem outliers nas próximas análises.

    Guarda um backup da base original (apenas uma vez) para permitir restauração
    e registra um histórico dos tratamentos aplicados.
    """
    if "df_original" not in st.session_state:
        st.session_state["df_original"] = st.session_state["df"]
        st.session_state["col_types_original"] = st.session_state["col_types"]
    # As colunas (e portanto os tipos) não mudam ao remover linhas.
    st.session_state["df"] = cleaned.reset_index(drop=True)
    st.session_state["outliers_applied"] = True
    st.session_state.setdefault("outliers_removed_log", []).append(
        {"variável": col, "removidos": int(n_removed)}
    )


def restore_original_base() -> None:
    """Restaura a base original e descarta a versão filtrada."""
    if "df_original" in st.session_state:
        st.session_state["df"] = st.session_state.pop("df_original")
        st.session_state["col_types"] = st.session_state.pop("col_types_original")
    st.session_state["outliers_applied"] = False
    st.session_state.pop("df_no_outliers", None)
    st.session_state.pop("outliers_removed_log", None)


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

    # Banner: indica quando as análises estão usando a base sem outliers
    if state.get("outliers_applied"):
        log = state.get("outliers_removed_log", [])
        resumo = ", ".join(f"{r['variável']} (−{r['removidos']})" for r in log) or "—"
        b1, b2 = st.columns([0.72, 0.28])
        b1.success(
            f"🧹 As análises estão usando a **base sem outliers** "
            f"({len(df)} linhas). Tratamentos: {resumo}."
        )
        if b2.button("↩️ Restaurar base original", use_container_width=True, key="out_restore"):
            restore_original_base()
            st.rerun()

    _method_help = {
        "IQR (1,5×amplitude interquartílica)": GLOSSARY["iqr"],
        "Z-score (|z| > 3)": GLOSSARY["zscore"],
        "MAD (desvio absoluto mediano)": GLOSSARY["mad"],
    }
    c1, c2 = st.columns(2)
    col = c1.selectbox("Variável numérica:", numeric_cols(state))
    method_label = c2.selectbox("Método de detecção:", list(METHODS.keys()),
                                help="IQR, Z-score ou MAD — passe o mouse para entender "
                                     "as diferenças. " + GLOSSARY["outlier"])

    series = validation.clean_numeric(df[col], min_n=4)
    if series is None:
        return
    data = series.to_numpy()

    mask = METHODS[method_label](data)
    n_out = int(mask.sum())
    method_short = method_label.split(" ")[0]

    m = st.columns(3)
    m[0].metric("Total de valores", len(data))
    m[1].metric("Outliers detectados", n_out, help=_method_help.get(method_label))
    m[2].metric("% outliers", f"{(100*n_out/len(data)):.1f}%",
                help="Proporção de valores marcados como outliers pelo método escolhido.")

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

    # Tratamento: gerar base sem os outliers detectados (download + aplicar)
    if n_out > 0:
        st.divider()
        st.markdown("#### 🧹 Tratar os outliers")
        st.caption(
            "Gere a base **sem** os outliers detectados nesta variável. O arquivo "
            "original que você importou nunca é alterado."
        )

        clean_idx = series.index[~mask]
        cleaned = df.loc[df.index.isin(clean_idx) | df[col].isna()].copy()
        st.session_state["df_no_outliers"] = cleaned
        n_removed = len(df) - len(cleaned)
        st.caption(f"Resultado: **{len(cleaned)} linhas** (removidas {n_removed}).")

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "📥 Baixar (Excel)",
            data=_to_excel_bytes(cleaned),
            file_name=f"base_sem_outliers_{col}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="out_dl_xlsx",
        )
        d2.download_button(
            "📥 Baixar (CSV)",
            data=cleaned.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"base_sem_outliers_{col}.csv",
            mime="text/csv",
            use_container_width=True,
            key="out_dl_csv",
        )
        if d3.button("✅ Usar nas próximas análises", use_container_width=True, key="out_apply"):
            apply_clean_base(cleaned, col, n_removed)
            st.rerun()

        st.caption(
            "💡 *Baixar* salva a planilha no seu computador. *Usar nas próximas análises* "
            "faz todas as telas (descritiva, correlação, etc.) passarem a considerar a "
            "base filtrada — você pode reverter a qualquer momento."
        )

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
