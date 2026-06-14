"""
transforms.py — Normalidade e transformações para capabilidade
==============================================================

Para análise de capabilidade (Cp/Cpk) os dados devem ser aproximadamente
normais. Quando não são, o procedimento usual é aplicar uma transformação que
normalize os dados — e calcular os índices no espaço transformado, transformando
também os limites de especificação.

Este módulo fornece:
  * ``ad_normality``     — teste de Anderson-Darling com p-valor;
  * ``evaluate_transforms`` — aplica Box-Cox, Johnson, Log, Raiz, Inverso e
    Yeo-Johnson, e ranqueia pelo p-valor de normalidade após transformar;
  * ``transform_spec``   — transforma os limites de especificação (LIE/LSE)
    para o mesmo espaço, tratando transformações decrescentes.

Tudo é puro NumPy/SciPy (sem Streamlit), para ser testável isoladamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Teste de normalidade Anderson-Darling com p-valor (aproximação de Stephens)
# --------------------------------------------------------------------------- #
def ad_normality(data) -> tuple[float, float]:
    """Anderson-Darling para normalidade. Retorna (estatística A², p-valor).

    O p-valor segue a aproximação clássica de D'Agostino & Stephens.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8 or np.std(x) == 0:
        return float("nan"), float("nan")

    a2 = float(stats.anderson(x, dist="norm").statistic)
    a2s = a2 * (1 + 0.75 / n + 2.25 / n ** 2)  # ajuste para n finito

    if a2s >= 0.6:
        p = np.exp(1.2937 - 5.709 * a2s + 0.0186 * a2s ** 2)
    elif a2s > 0.34:
        p = np.exp(0.9177 - 4.279 * a2s - 1.38 * a2s ** 2)
    elif a2s > 0.2:
        p = 1 - np.exp(-8.318 + 42.796 * a2s - 59.938 * a2s ** 2)
    else:
        p = 1 - np.exp(-13.436 + 101.14 * a2s - 223.73 * a2s ** 2)
    return a2, float(min(max(p, 0.0), 1.0))


# --------------------------------------------------------------------------- #
# Estrutura de resultado de uma transformação
# --------------------------------------------------------------------------- #
@dataclass
class TransformResult:
    name: str
    func: Callable[[np.ndarray], np.ndarray]  # escala original -> transformada
    transformed: np.ndarray
    params: str
    domain: str
    ad_p: float = float("nan")
    increasing: bool = True


def _finite(arr) -> bool:
    return bool(np.all(np.isfinite(arr)))


# --------------------------------------------------------------------------- #
# Transformações clássicas / Box-Cox / Yeo-Johnson
# --------------------------------------------------------------------------- #
def _t_boxcox(x: np.ndarray) -> TransformResult | None:
    if np.any(x <= 0):
        return None
    xt, lam = stats.boxcox(x)

    def f(v, lam=lam):
        v = np.asarray(v, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            if abs(lam) < 1e-8:
                return np.log(v)
            return (np.power(v, lam) - 1.0) / lam

    return TransformResult(f"Box-Cox (λ={lam:.3f})", f, np.asarray(xt),
                           f"λ = {lam:.4f}", "x > 0")


def _t_yeojohnson(x: np.ndarray) -> TransformResult | None:
    try:
        xt, lam = stats.yeojohnson(x)
    except Exception:
        return None

    def f(v, lam=lam):
        v = np.asarray(v, dtype=float)
        with np.errstate(invalid="ignore"):
            return stats.yeojohnson(v, lmbda=lam)

    return TransformResult(f"Yeo-Johnson (λ={lam:.3f})", f, np.asarray(xt),
                           f"λ = {lam:.4f}", "qualquer valor")


def _t_log(x: np.ndarray) -> TransformResult | None:
    if np.any(x <= 0):
        return None
    return TransformResult("Logaritmo (ln)", lambda v: np.log(np.asarray(v, float)),
                           np.log(x), "ln(x)", "x > 0")


def _t_sqrt(x: np.ndarray) -> TransformResult | None:
    if np.any(x < 0):
        return None
    return TransformResult("Raiz quadrada", lambda v: np.sqrt(np.asarray(v, float)),
                           np.sqrt(x), "√x", "x ≥ 0")


def _t_reciprocal(x: np.ndarray) -> TransformResult | None:
    if np.any(x == 0):
        return None
    return TransformResult("Inverso (1/x)", lambda v: 1.0 / np.asarray(v, float),
                           1.0 / x, "1/x", "x ≠ 0")


# --------------------------------------------------------------------------- #
# Transformação de Johnson (método dos quantis de Slifker & Shapiro, 1980)
# --------------------------------------------------------------------------- #
def _johnson_fit(x: np.ndarray, z: float) -> TransformResult | None:
    """Ajusta uma transformação de Johnson (SU/SB/SL) para um dado ``z``."""
    # Quantis dos dados nos z-scores -3z, -z, z, 3z
    probs = stats.norm.cdf([-3 * z, -z, z, 3 * z])
    x_m3, x_m1, x_p1, x_p3 = np.quantile(x, probs)

    m = x_p3 - x_p1
    n = x_m1 - x_m3
    p = x_p1 - x_m1
    if m <= 0 or n <= 0 or p <= 0:
        return None

    d = (m * n) / (p * p)
    mp, npr = m / p, n / p

    try:
        if d > 1.01:  # ---- SU (não limitada) -----------------------------
            eta = 2 * z / np.arccosh(0.5 * (mp + npr))
            gamma = eta * np.arcsinh((npr - mp) / (2 * np.sqrt(mp * npr - 1)))
            lam = (2 * p * np.sqrt(mp * npr - 1)
                   / ((mp + npr - 2) * np.sqrt(mp + npr + 2)))
            xi = 0.5 * (x_p1 + x_m1) + p * (npr - mp) / (2 * (mp + npr - 2))
            fam = "SU"

            def f(v, g=gamma, e=eta, l=lam, xi=xi):
                v = np.asarray(v, dtype=float)
                return g + e * np.arcsinh((v - xi) / l)

        elif d < 0.99:  # ---- SB (limitada) -------------------------------
            a = (1 + p / m) * (1 + p / n)
            eta = z / np.arccosh(0.5 * np.sqrt(a))
            pmn = p * p / (m * n)
            gamma = eta * np.arcsinh((p / n - p / m) * np.sqrt(a - 4)
                                     / (2 * (pmn - 1)))
            lam = p * np.sqrt(((a - 2) ** 2 - 4)) / (pmn - 1)
            xi = (0.5 * (x_p1 + x_m1) - 0.5 * lam
                  + p * (p / n - p / m) / (2 * (pmn - 1)))
            fam = "SB"

            def f(v, g=gamma, e=eta, l=lam, xi=xi):
                v = np.asarray(v, dtype=float)
                y = (v - xi) / l
                eps = 1e-9
                y = np.clip(y, eps, 1 - eps)
                return g + e * np.log(y / (1 - y))

        else:  # ---- SL (lognormal) ---------------------------------------
            eta = 2 * z / np.log(mp)
            gamma = eta * np.log((mp - 1) / (p * np.sqrt(mp)))
            xi = 0.5 * (x_p1 + x_m1) - 0.5 * p * (mp + 1) / (mp - 1)
            lam = 1.0
            fam = "SL"

            def f(v, g=gamma, e=eta, xi=xi):
                v = np.asarray(v, dtype=float)
                with np.errstate(invalid="ignore"):
                    return g + e * np.log(v - xi)

    except Exception:
        return None

    if not all(np.isfinite([eta, gamma, lam, xi])) or eta <= 0:
        return None

    transformed = f(x)
    if not _finite(transformed):
        return None

    params = f"família {fam}, z={z:.2f}, η={eta:.3g}, γ={gamma:.3g}, ξ={xi:.3g}"
    return TransformResult(f"Johnson ({fam})", f, np.asarray(transformed),
                           params, "qualquer valor")


def _t_johnson(x: np.ndarray) -> TransformResult | None:
    """Busca o melhor ``z`` maximizando o p-valor de normalidade."""
    if x.size < 20:
        return None
    best, best_p = None, -1.0
    for z in np.arange(0.25, 1.26, 0.05):
        fit = _johnson_fit(x, float(z))
        if fit is None:
            continue
        _, p = ad_normality(fit.transformed)
        if np.isfinite(p) and p > best_p:
            best, best_p = fit, p
    return best


# --------------------------------------------------------------------------- #
# Avaliação conjunta + transformação dos limites
# --------------------------------------------------------------------------- #
_BUILDERS = (_t_boxcox, _t_johnson, _t_log, _t_sqrt, _t_reciprocal, _t_yeojohnson)


def evaluate_transforms(data) -> list[TransformResult]:
    """Aplica todas as transformações aplicáveis e ranqueia por normalidade (p AD)."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    results: list[TransformResult] = []
    for build in _BUILDERS:
        try:
            r = build(x)
        except Exception:
            r = None
        if r is None or not _finite(r.transformed) or np.std(r.transformed) == 0:
            continue
        _, p = ad_normality(r.transformed)
        if not np.isfinite(p):
            continue
        r.ad_p = p
        # direção (crescente/decrescente) para transformar limites unilaterais
        lo, hi = float(np.min(x)), float(np.max(x))
        try:
            f_lo = float(np.ravel(r.func(np.array([lo])))[0])
            f_hi = float(np.ravel(r.func(np.array([hi])))[0])
            r.increasing = f_hi >= f_lo
        except Exception:
            r.increasing = True
        results.append(r)

    results.sort(key=lambda t: t.ad_p, reverse=True)
    return results


def transform_spec(result: TransformResult, lie, lse):
    """Transforma (LIE, LSE) para o espaço da transformação.

    Retorna ``(novo_lie, novo_lse)``. Trata transformações decrescentes (o limite
    inferior pode virar superior) e valores fora do domínio (viram ``None``).
    """
    def tf(val):
        if val is None:
            return None
        try:
            out = float(np.ravel(result.func(np.array([val], dtype=float)))[0])
        except Exception:
            return None
        return out if np.isfinite(out) else None

    a, b = tf(lie), tf(lse)
    if a is not None and b is not None:
        return min(a, b), max(a, b)
    # unilateral: se a transformação é decrescente, a borda troca de lado
    if result.increasing:
        return a, b
    return b, a
