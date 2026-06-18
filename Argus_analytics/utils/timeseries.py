"""
timeseries.py — Validação de lag por ARIMAX + Ljung-Box
=======================================================

A correlação cruzada (CCF) é rápida para *identificar* um lag candidato, mas
pode ser enganosa: duas séries autocorrelacionadas ou com tendência podem
exibir correlação alta sem relação causal real (correlação espúria).

Este módulo confirma o lag de forma mais robusta com um modelo de **função de
transferência / ARIMAX**: modela a dinâmica própria do alvo (parte ARIMA) e
inclui a variável explicativa **defasada** como regressor exógeno. Em seguida,
o teste de **Ljung-Box** verifica se os resíduos são ruído branco — se forem, o
modelo capturou a estrutura temporal e o efeito do regressor não é artefato.

Critério de validação:
  * regressor exógeno **significativo** (p-valor < 0,05) e
  * resíduos **ruído branco** (Ljung-Box p-valor > 0,05).

Tudo é puro NumPy/Pandas/statsmodels (sem Streamlit), para ser testável
isoladamente. O import de statsmodels é protegido: se ausente, ``_HAS_SM`` fica
``False`` e a validação retorna um objeto sinalizando indisponibilidade.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # statsmodels é opcional — mesma convenção de modules/regression.py
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


@dataclass
class ArimaxValidation:
    """Resultado da validação ARIMAX de um lag.

    ``ok`` indica que o modelo foi ajustado com sucesso; quando ``False``, o
    campo ``error`` traz o motivo e os demais ficam NaN.
    """
    lag_min: float
    order: tuple[int, int, int]
    exog_coef: float = float("nan")
    exog_p: float = float("nan")
    aic: float = float("nan")
    lb_p: float = float("nan")
    n_obs: int = 0
    fitted: np.ndarray | None = None
    actual: np.ndarray | None = None
    resid: np.ndarray | None = None
    significant: bool = False
    white_noise: bool = False
    validated: bool = False
    ok: bool = False
    error: str | None = None


_EXOG_NAME = "exog_lag"


def _align(target: pd.Series, exog: pd.Series, lag: int) -> tuple[pd.Series, pd.Series]:
    """Alinha alvo e explicativa defasada por ``lag``, removendo ausentes.

    Reindexa para uma sequência inteira contígua (a grade temporal regular já é
    montada antes), evitando avisos de frequência do statsmodels.
    """
    shifted = exog.shift(lag)
    pair = pd.concat([target, shifted], axis=1).dropna()
    y = pair.iloc[:, 0].reset_index(drop=True).astype(float)
    x = pair.iloc[:, 1].reset_index(drop=True).astype(float)
    return y, x


def _fit_arimax(y: pd.Series, x: pd.Series, order: tuple[int, int, int]):
    """Ajusta ARIMA(order) com regressor exógeno. Pode levantar exceção."""
    exog_df = pd.DataFrame({_EXOG_NAME: x.to_numpy()})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y.to_numpy(), order=order, exog=exog_df,
                      enforce_stationarity=False, enforce_invertibility=False)
        return model.fit()


def select_order_aic(y: pd.Series, x: pd.Series,
                     p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1)
                     ) -> tuple[int, int, int]:
    """Escolhe (p, d, q) por menor AIC numa busca pequena (≤12 ajustes)."""
    best_order, best_aic = (1, 0, 0), float("inf")
    for d in d_range:
        for p in p_range:
            for q in q_range:
                if p == 0 and q == 0 and d == 0:
                    continue  # modelo trivial (só exógena) — pouco informativo
                try:
                    res = _fit_arimax(y, x, (p, d, q))
                    aic = float(res.aic)
                except Exception:
                    continue
                if np.isfinite(aic) and aic < best_aic:
                    best_aic, best_order = aic, (p, d, q)
    return best_order


def validate_lag(target: pd.Series, exog: pd.Series, lag: int, step_min: float = 1.0,
                 order: tuple[int, int, int] | None = None) -> ArimaxValidation:
    """Valida o ``lag`` (em passos) entre ``exog`` e ``target`` via ARIMAX.

    ``step_min`` é apenas o tamanho do passo em minutos, usado para registrar o
    lag em minutos no resultado. Se ``order`` for ``None``, escolhe por AIC.
    """
    lag_min = float(lag) * float(step_min)
    if not _HAS_SM:
        return ArimaxValidation(lag_min=lag_min, order=order or (0, 0, 0),
                                error="statsmodels não está instalado.")

    y, x = _align(target, exog, lag)
    n = int(len(y))
    if n < 20 or y.std() == 0 or x.std() == 0:
        return ArimaxValidation(
            lag_min=lag_min, order=order or (0, 0, 0), n_obs=n,
            error="Dados insuficientes ou sem variação para ajustar o ARIMAX "
                  "(são necessários ≥ 20 pontos válidos).")

    chosen = order or select_order_aic(y, x)
    try:
        res = _fit_arimax(y, x, chosen)
    except Exception as exc:  # pragma: no cover - convergência
        return ArimaxValidation(lag_min=lag_min, order=chosen, n_obs=n,
                                error=f"Falha ao ajustar o modelo: {exc}")

    coef = float(pd.Series(res.params, copy=False).get(_EXOG_NAME, np.nan))
    pval = float(pd.Series(res.pvalues, copy=False).get(_EXOG_NAME, np.nan))
    aic = float(res.aic)

    # Ljung-Box nos resíduos (descartando o início para evitar artefatos de
    # inicialização do filtro de Kalman).
    resid = np.asarray(res.resid, dtype=float)
    fitted = np.asarray(res.fittedvalues, dtype=float)
    actual = y.to_numpy(dtype=float)
    skip = min(max(chosen[0] + chosen[1] + chosen[2], 1), max(n // 5, 1))
    resid_test = resid[skip:]
    lb_lags = max(1, min(10, len(resid_test) // 5))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb = acorr_ljungbox(resid_test, lags=[lb_lags], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[-1])
    except Exception:
        lb_p = float("nan")

    significant = bool(np.isfinite(pval) and pval < 0.05)
    white_noise = bool(np.isfinite(lb_p) and lb_p > 0.05)
    return ArimaxValidation(
        lag_min=lag_min, order=chosen, exog_coef=coef, exog_p=pval, aic=aic,
        lb_p=lb_p, n_obs=n, fitted=fitted, actual=actual, resid=resid,
        significant=significant, white_noise=white_noise,
        validated=significant and white_noise, ok=True,
    )
