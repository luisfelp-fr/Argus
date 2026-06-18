"""
Módulo 7 — Regressão Linear
===========================

Regressão linear simples ou múltipla via statsmodels (OLS): coeficientes,
p-valores, R², R² ajustado, gráficos real×previsto e de resíduos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils import validation
from utils.helpers import numeric_cols, make_report_item, add_to_report, now_str
from utils.plotting import real_vs_pred, residuals_plot
from utils.interpretation import interpret_r2, interpret_coefficients
from utils.glossary import GLOSSARY

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


class _OLSResult:
    """Resultado de OLS compatível (subset) com statsmodels, usando apenas numpy.

    Serve de *fallback* quando statsmodels não está instalado, calculando
    coeficientes, erros padrão, estatísticas t, p-valores, R² e R² ajustado.
    """

    def __init__(self, names, params, bse, tvalues, pvalues, rsquared,
                 rsquared_adj, nobs):
        self.params = pd.Series(params, index=names)
        self.bse = pd.Series(bse, index=names)
        self.tvalues = pd.Series(tvalues, index=names)
        self.pvalues = pd.Series(pvalues, index=names)
        self.rsquared = rsquared
        self.rsquared_adj = rsquared_adj
        self.nobs = nobs

    def predict(self, X):
        return np.asarray(X) @ self.params.values


def _fit_ols_numpy(X_df: pd.DataFrame, y: pd.Series, names: list[str]) -> _OLSResult:
    import numpy as np
    from scipy import stats as _stats

    X = np.column_stack([np.ones(len(X_df)), X_df.to_numpy(dtype=float)])
    yv = y.to_numpy(dtype=float)
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = max(n - k, 1)
    mse = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(mse * xtx_inv))
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * (1 - _stats.t.cdf(np.abs(tvals), dof))
    ss_tot = float(((yv - yv.mean()) ** 2).sum()) or 1.0
    ss_res = float(resid @ resid)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / dof
    return _OLSResult(names, beta, se, tvals, pvals, r2, r2_adj, n)


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    """Ajusta um OLS com constante. Retorna (resultado, X_alinhado, y_alinhado).

    Usa statsmodels quando disponível; caso contrário, recorre a um OLS próprio
    baseado em numpy/scipy (mesmas saídas essenciais).
    """
    data = df[[y_col] + x_cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = data[y_col]
    names = ["const"] + list(x_cols)
    if _HAS_SM:
        X = sm.add_constant(data[x_cols], has_constant="add")
        model = sm.OLS(y, X).fit()
    else:
        model = _fit_ols_numpy(data[x_cols], y, names)
    return model, data[x_cols], y


def render(state) -> None:
    if not validation.require_data() or not validation.require_numeric(state, 2):
        return

    df: pd.DataFrame = state["df"]
    st.info(
        "A regressão ajuda a entender como uma ou mais variáveis podem explicar o "
        "comportamento de uma variável resposta."
    )

    num = numeric_cols(state)
    y_col = st.selectbox("Variável resposta (Y):", num,
                         help="A variável que você quer explicar/prever.")
    x_options = [c for c in num if c != y_col]
    x_cols = st.multiselect("Variáveis explicativas (X):", x_options,
                            default=x_options[: min(2, len(x_options))],
                            help=GLOSSARY["regressao"])

    if not x_cols:
        st.warning("Selecione ao menos uma variável explicativa.")
        return

    if st.button("▶ Executar regressão", type="primary", key="reg_run"):
        st.session_state["reg_has_run"] = True
    if not st.session_state.get("reg_has_run"):
        st.caption("Configure as variáveis e clique em **Executar regressão**.")
        return

    try:
        model, X, y = fit_ols(df, y_col, x_cols)
    except Exception as exc:
        st.error(f"Não foi possível ajustar o modelo: {exc}")
        return

    if len(y) <= len(x_cols) + 1:
        st.warning("Poucas observações para o número de variáveis. Resultado não confiável.")
        return

    # Matriz de projeto com constante na primeira coluna (consistente entre
    # statsmodels e o fallback numpy).
    design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    y_pred = pd.Series(np.asarray(model.predict(design)), index=y.index)
    residuals = y - y_pred

    # Tabela de coeficientes
    coef_table = pd.DataFrame({
        "Variável": model.params.index,
        "Coeficiente": model.params.values.round(5),
        "Erro padrão": model.bse.values.round(5),
        "t": model.tvalues.values.round(3),
        "p-valor": model.pvalues.values.round(5),
    })

    kind = "simples" if len(x_cols) == 1 else "múltipla"
    tabs = st.tabs(["📐 Coeficientes", "🎯 Ajuste", "📉 Resíduos", "🗣️ Interpretação"])

    with tabs[0]:
        st.caption(f"Regressão linear **{kind}** — {y_col} ~ {' + '.join(x_cols)}")
        st.caption("Tabela de coeficientes — cada linha mostra o efeito da variável.")
        st.dataframe(coef_table, use_container_width=True, hide_index=True)
        st.caption("ℹ️ Coeficiente, p-valor e R²: passe o mouse nos ícones ❓ abaixo.")
        m = st.columns(3)
        m[0].metric("R²", f"{model.rsquared:.3f}", help=GLOSSARY["r2"])
        m[1].metric("R² ajustado", f"{model.rsquared_adj:.3f}", help=GLOSSARY["r2_aj"])
        m[2].metric("N observações", int(model.nobs),
                    help="Número de linhas usadas no ajuste (após remover ausentes).")

    fig_rp = real_vs_pred(y.to_numpy(), y_pred.to_numpy())
    fig_res = residuals_plot(y_pred.to_numpy(), residuals.to_numpy())
    with tabs[1]:
        st.plotly_chart(fig_rp, use_container_width=True, key="reg_rp")
        st.caption("Quanto mais próximos da linha tracejada, melhor o ajuste.")
    with tabs[2]:
        st.plotly_chart(fig_res, use_container_width=True, key="reg_res")
        st.caption("Resíduos devem se espalhar aleatoriamente em torno de zero, sem padrão.")

    interp_r2 = interpret_r2(model.rsquared)
    interp_coef = interpret_coefficients(list(model.params.index),
                                         list(model.params.values),
                                         list(model.pvalues.values))
    with tabs[3]:
        st.markdown(f"- {interp_r2}")
        st.markdown(f"- {interp_coef}")

    st.divider()
    if st.button("➕ Adicionar ao relatório", key="reg_add"):
        item = make_report_item(
            name=f"Regressão Linear ({kind})",
            variables={"resposta": y_col, "explicativas": x_cols},
            params={"R2": round(model.rsquared, 4), "R2_aj": round(model.rsquared_adj, 4)},
            interpretation=f"{interp_r2}\n{interp_coef}",
            figures=[fig_rp, fig_res],
            tables={"Coeficientes": coef_table},
            timestamp=now_str(),
        )
        add_to_report(item)
        st.success("Análise adicionada ao relatório!")
