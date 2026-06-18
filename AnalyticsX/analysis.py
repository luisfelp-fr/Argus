"""
Análise de correlação multivariada com detecção de lag (atraso causa→efeito)
----------------------------------------------------------------------------
Núcleo de cálculo, sem dependência de interface gráfica (testável isoladamente).

Fluxo:
  1. load_table          -> lê a planilha (data/hora + colunas de indicadores)
  2. detect_and_resample -> detecta o passo temporal e reamostra numa grade regular
  3. analyze_all         -> para cada indicador, detecta o lag ótimo e calcula
                            correlação de Pearson + Informação Mútua

Convenção de lag: lag POSITIVO significa que o indicador ANTECEDE o
indicador-chave (a causa ocorre antes do efeito). Em outras palavras, deslocar
o indicador para frente no tempo (atrasá-lo) alinha-o com o indicador-chave.
"""

from __future__ import annotations

import io
import itertools
import re
import unicodedata
from dataclasses import dataclass
from math import factorial

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.linear_model import LinearRegression
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - dependência opcional
    _HAS_SKLEARN = False

# --- Dependências adicionais da VERSÃO 2 (produção diária) ----------------- #
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - dependência opcional
    _HAS_SCIPY = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    _HAS_RF = True
except ImportError:  # pragma: no cover - dependência opcional
    _HAS_RF = False

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover - dependência opcional
    _HAS_XGB = False

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:  # pragma: no cover - dependência opcional
    _HAS_SHAP = False


# --------------------------------------------------------------------------- #
# 1. Leitura da planilha
# --------------------------------------------------------------------------- #
def load_table(file, datetime_col: int | str = 0) -> pd.DataFrame:
    """Lê uma planilha .xlsx ou .csv.

    Espera a coluna de data/hora na primeira posição (ou indicada por
    ``datetime_col``) e as demais colunas como indicadores numéricos.

    Retorna um DataFrame indexado por DatetimeIndex, ordenado, com colunas
    numéricas (vírgula decimal BR tratada).
    """
    name = getattr(file, "name", str(file)).lower()

    if name.endswith(".csv"):
        # Tenta separador ';' (padrão BR) e cai para ',' se gerar 1 coluna só.
        df = pd.read_csv(file, sep=None, engine="python")
    else:
        df = pd.read_excel(file, engine="openpyxl")

    if df.shape[1] < 2:
        raise ValueError(
            "A planilha precisa ter ao menos 2 colunas: data/hora + 1 indicador."
        )

    # Identifica a coluna de data/hora
    if isinstance(datetime_col, int):
        dt_name = df.columns[datetime_col]
    else:
        dt_name = datetime_col

    df[dt_name] = pd.to_datetime(df[dt_name], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[dt_name])
    df = df.set_index(dt_name).sort_index()
    df.index.name = "datetime"

    # Coage indicadores para numérico (trata vírgula decimal e separador de milhar BR)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            s = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)   # separador de milhar
                .str.replace(",", ".", regex=False)   # vírgula decimal -> ponto
            )
            df[col] = pd.to_numeric(s, errors="coerce")

    # Descarta colunas totalmente vazias (não numéricas / sem dados)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 1:
        raise ValueError("Nenhuma coluna de indicador numérica foi encontrada.")

    return df


# --------------------------------------------------------------------------- #
# 2. Detecção do passo temporal + reamostragem
# --------------------------------------------------------------------------- #
@dataclass
class ResampleInfo:
    step_minutes: float          # passo da grade regular (em minutos)
    n_rows: int                  # nº de linhas após reamostragem
    n_gaps_filled: int           # nº de células interpoladas
    nan_fraction: float          # fração de NaN restante (após interpolação)
    irregular: bool              # True se os timestamps originais eram irregulares


def _detect_step_minutes(index: pd.DatetimeIndex) -> tuple[float, bool]:
    """Estima o passo base (mediana das diferenças) em minutos e se é irregular."""
    if len(index) < 2:
        raise ValueError("Série temporal muito curta (menos de 2 amostras).")

    diffs = np.diff(index.values).astype("timedelta64[s]").astype(float)  # segundos
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Timestamps duplicados/sem variação; não há passo válido.")

    median_s = float(np.median(diffs))
    # Irregular se houver dispersão relevante em torno da mediana
    irregular = bool(np.std(diffs) > 0.1 * median_s)
    return median_s / 60.0, irregular


def detect_and_resample(
    df: pd.DataFrame,
    step_minutes: float | None = None,
    interp_limit: int = 3,
) -> tuple[pd.DataFrame, ResampleInfo]:
    """Reamostra o DataFrame numa grade temporal regular.

    Se ``step_minutes`` não for informado, ele é detectado automaticamente a
    partir da mediana dos intervalos entre timestamps consecutivos.

    Lacunas curtas (até ``interp_limit`` amostras consecutivas) são interpoladas
    linearmente; lacunas maiores permanecem NaN.
    """
    detected_step, irregular = _detect_step_minutes(df.index)
    step = step_minutes if step_minutes else detected_step
    step = max(step, 1e-6)

    freq = pd.Timedelta(minutes=step)
    regular = df.resample(freq).mean()

    nan_before = int(regular.isna().sum().sum())
    regular = regular.interpolate(method="linear", limit=interp_limit, limit_area="inside")
    nan_after = int(regular.isna().sum().sum())

    total_cells = regular.size if regular.size else 1
    info = ResampleInfo(
        step_minutes=round(step, 6),
        n_rows=len(regular),
        n_gaps_filled=nan_before - nan_after,
        nan_fraction=nan_after / total_cells,
        irregular=irregular,
    )
    return regular, info


def composite_average(
    df: pd.DataFrame, lab_indicators: list[str], period_hours: float
) -> tuple[pd.DataFrame, ResampleInfo]:
    """Trata variáveis de laboratório (amostras compostas de N horas).

    Uma amostra de laboratório registrada no instante T representa a **média do
    período** ``(T − N horas, T]`` (coleta composta e homogeneizada) — logo **não
    pode ser interpolada**. Para tornar todos os indicadores comparáveis nessa
    mesma base, para cada instante T de coleta de laboratório calcula-se a **média
    de TODOS os indicadores** na janela ``(T − N horas, T]``. As variáveis de
    laboratório, tendo um único valor na janela, resultam no próprio valor reportado.

    As âncoras são os instantes em que há valor de alguma variável de laboratório.
    Retorna ``(df_composto, info)`` indexado por essas âncoras (sem interpolação).
    """
    missing = [c for c in lab_indicators if c not in df.columns]
    if missing:
        raise ValueError(f"Indicadores de laboratório inexistentes: {missing}")
    if not lab_indicators:
        raise ValueError("Nenhuma variável de laboratório informada.")
    if period_hours <= 0:
        raise ValueError("O período da análise composta deve ser positivo.")

    period = pd.Timedelta(hours=period_hours)
    lab_present = df[lab_indicators].notna().any(axis=1)
    anchors = df.index[lab_present]
    if len(anchors) < 2:
        raise ValueError(
            "São necessárias ao menos 2 coletas de laboratório para a análise."
        )

    rows = {
        T: df.loc[(df.index > T - period) & (df.index <= T)].mean(numeric_only=True)
        for T in anchors
    }
    comp = pd.DataFrame(rows).T.sort_index()
    comp.index.name = df.index.name or "datetime"

    # Irregularidade do agendamento de laboratório (espaçamento entre coletas)
    diffs = np.diff(comp.index.values).astype("timedelta64[s]").astype(float)
    irregular = bool(len(diffs) and np.std(diffs) > 0.1 * np.median(diffs))

    total_cells = comp.size if comp.size else 1
    info = ResampleInfo(
        step_minutes=round(period_hours * 60.0, 6),
        n_rows=len(comp),
        n_gaps_filled=0,
        nan_fraction=float(comp.isna().sum().sum()) / total_cells,
        irregular=irregular,
    )
    return comp, info


# --------------------------------------------------------------------------- #
# 3. Correlação cruzada com defasagem (TLCC)
# --------------------------------------------------------------------------- #
def cross_correlation(
    key: pd.Series,
    other: pd.Series,
    max_lag: int,
    min_overlap: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlação de Pearson entre ``key`` e ``other`` para cada lag inteiro.

    lag > 0  -> ``other`` é deslocado para frente (atrasado) -> testa se ``other``
                ANTECEDE ``key`` (other é a causa, key é o efeito).

    Retorna (lags, corrs) onde corr[i] pode ser NaN se o overlap for insuficiente.
    """
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.full(len(lags), np.nan)

    for i, lag in enumerate(lags):
        shifted = other.shift(lag)
        pair = pd.concat([key, shifted], axis=1).dropna()
        if len(pair) < min_overlap:
            continue
        a = pair.iloc[:, 0].to_numpy()
        b = pair.iloc[:, 1].to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        corrs[i] = float(np.corrcoef(a, b)[0, 1])

    return lags, corrs


def _mutual_information_at_lag(key: pd.Series, other: pd.Series, lag: int) -> float:
    """Informação mútua entre key e other deslocado por ``lag``."""
    if not _HAS_SKLEARN:
        return float("nan")
    pair = pd.concat([key, other.shift(lag)], axis=1).dropna()
    if len(pair) < 10:
        return float("nan")
    x = pair.iloc[:, 1].to_numpy().reshape(-1, 1)
    y = pair.iloc[:, 0].to_numpy()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    mi = mutual_info_regression(x, y, random_state=0)
    return float(mi[0])


# --------------------------------------------------------------------------- #
# 4. Análise de um indicador e ranking completo
# --------------------------------------------------------------------------- #
@dataclass
class LagResult:
    indicator: str
    best_lag_samples: int
    best_lag_minutes: float
    corr_at_lag: float          # Pearson no lag ótimo
    corr_at_zero: float         # Pearson em lag 0 (sem correção de atraso)
    mutual_info: float          # Informação Mútua no lag ótimo
    n_points: int               # nº de pontos usados no lag ótimo
    lags: np.ndarray            # vetor de lags (amostras) para o gráfico
    corrs: np.ndarray           # curva de correlação correspondente


def analyze_indicator(
    df: pd.DataFrame, key: str, other: str, max_lag: int, step_minutes: float
) -> LagResult:
    key_s, other_s = df[key], df[other]
    lags, corrs = cross_correlation(key_s, other_s, max_lag)

    if np.all(np.isnan(corrs)):
        best_lag = 0
        corr_at_lag = float("nan")
    else:
        best_idx = int(np.nanargmax(np.abs(corrs)))
        best_lag = int(lags[best_idx])
        corr_at_lag = float(corrs[best_idx])

    zero_idx = int(np.where(lags == 0)[0][0])
    corr_at_zero = float(corrs[zero_idx])

    pair = pd.concat([key_s, other_s.shift(best_lag)], axis=1).dropna()

    return LagResult(
        indicator=other,
        best_lag_samples=best_lag,
        best_lag_minutes=round(best_lag * step_minutes, 3),
        corr_at_lag=corr_at_lag,
        corr_at_zero=corr_at_zero,
        mutual_info=_mutual_information_at_lag(key_s, other_s, best_lag),
        n_points=len(pair),
        lags=lags,
        corrs=corrs,
    )


def analyze_all(
    df: pd.DataFrame, key: str, max_lag: int, step_minutes: float
) -> tuple[pd.DataFrame, dict[str, LagResult]]:
    """Analisa todos os indicadores ≠ chave.

    Retorna:
      - DataFrame de ranking ordenado por |corr no lag ótimo| (descendente)
      - dict {indicador: LagResult} com as curvas para os gráficos
    """
    results: dict[str, LagResult] = {}
    for col in df.columns:
        if col == key:
            continue
        results[col] = analyze_indicator(df, key, col, max_lag, step_minutes)

    rows = [
        {
            "Indicador": r.indicator,
            "Lag ótimo (min)": r.best_lag_minutes,
            "Lag (amostras)": r.best_lag_samples,
            "Correlação (lag ótimo)": round(r.corr_at_lag, 4),
            "|Correlação|": round(abs(r.corr_at_lag), 4) if not np.isnan(r.corr_at_lag) else np.nan,
            "Correlação (lag 0)": round(r.corr_at_zero, 4),
            "Informação Mútua": round(r.mutual_info, 4),
            "Nº pontos": r.n_points,
        }
        for r in results.values()
    ]
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            "|Correlação|", ascending=False, na_position="last"
        ).reset_index(drop=True)

    return ranking, results


# --------------------------------------------------------------------------- #
# 5. Detecção do maior distúrbio (para o gráfico de impacto causa→efeito)
# --------------------------------------------------------------------------- #
@dataclass
class MainEvent:
    peak_time: pd.Timestamp          # instante do maior distúrbio na causa
    peak_z: float                    # z-score (assinado) no pico
    cause_start: pd.Timestamp        # início da janela do distúrbio (causa)
    cause_end: pd.Timestamp          # fim da janela do distúrbio (causa)
    effect_start: pd.Timestamp       # início da janela de efeito na chave (causa + lag)
    effect_end: pd.Timestamp         # fim da janela de efeito na chave
    lag_minutes: float               # atraso aplicado (causa→efeito)


def detect_main_event(
    cause: pd.Series,
    lag_samples: int,
    step_minutes: float,
    threshold_frac: float = 0.5,
    min_z: float = 1.5,
    pad: int = 2,
) -> MainEvent | None:
    """Localiza o maior distúrbio na série-causa e a janela de efeito na chave.

    - Padroniza a série (z-score) e acha o instante de maior |z| (evento mais forte).
    - A **janela da causa** é o trecho contíguo em torno do pico onde |z| permanece
      acima de ``max(threshold_frac * |z_pico|, min_z)`` (com folga de ``pad`` amostras).
    - A **janela do efeito** é a janela da causa deslocada para frente por
      ``lag_samples`` (o impacto aparece ``lag`` minutos depois na variável-chave).

    Retorna ``None`` se a série não tiver variação suficiente para um evento.
    """
    s = cause.dropna()
    if len(s) < 3 or s.std() == 0:
        return None

    z = (s - s.mean()) / s.std()
    absz = z.abs()
    peak_pos = int(np.argmax(absz.to_numpy()))   # posição inteira no array sem NaN
    peak_val = float(absz.iloc[peak_pos])

    thr = max(threshold_frac * peak_val, min_z)

    # Expande à esquerda e à direita enquanto |z| >= limiar
    left = peak_pos
    while left - 1 >= 0 and absz.iloc[left - 1] >= thr:
        left -= 1
    right = peak_pos
    n = len(absz)
    while right + 1 < n and absz.iloc[right + 1] >= thr:
        right += 1

    left = max(0, left - pad)
    right = min(n - 1, right + pad)

    times = s.index
    cause_start, cause_end = times[left], times[right]
    delta = pd.Timedelta(minutes=lag_samples * step_minutes)

    return MainEvent(
        peak_time=times[peak_pos],
        peak_z=float(z.iloc[peak_pos]),
        cause_start=cause_start,
        cause_end=cause_end,
        effect_start=cause_start + delta,
        effect_end=cause_end + delta,
        lag_minutes=round(lag_samples * step_minutes, 3),
    )


# --------------------------------------------------------------------------- #
# 6. Impacto (% de variância explicada) e status atual — para o painel
# --------------------------------------------------------------------------- #
def impact_percentage(corr: float) -> float:
    """Percentual de impacto = R² = corr² × 100.

    Interpretação: fração da variação do indicador-chave explicada por este
    indicador (no lag ótimo). É a métrica padrão de variância explicada.
    """
    if corr is None or np.isnan(corr):
        return float("nan")
    return float(corr ** 2 * 100.0)


def add_impact_columns(ranking: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ao ranking as colunas de painel:

    - **Impacto (%)**: R² individual (variância explicada) de cada indicador.
    - **Contribuição relativa (%)**: R² normalizado para somar 100% entre os
      indicadores (importância relativa de cada driver).
    - **Sentido**: 'Direto ↑' (corr > 0) ou 'Inverso ↓' (corr < 0).
    """
    out = ranking.copy()
    if out.empty:
        return out

    corr = out["Correlação (lag ótimo)"]
    out["Impacto (%)"] = (corr ** 2 * 100).round(1)
    total_r2 = float((corr ** 2).sum())
    out["Contribuição relativa (%)"] = (
        (corr ** 2 / total_r2 * 100).round(1) if total_r2 > 0 else 0.0
    )
    out["Sentido"] = np.where(corr >= 0, "Direto ↑", "Inverso ↓")
    return out


@dataclass
class CurrentStatus:
    z: float                 # z-score do nível recente vs. média histórica
    disturbed: bool          # True se |z| acima do limiar
    direction: int           # +1 subindo, -1 caindo, 0 estável


def recent_status(
    series: pd.Series, window: int = 3, threshold: float = 1.5
) -> CurrentStatus:
    """Avalia se a série está num distúrbio AGORA (nível recente atípico).

    Compara a média das últimas ``window`` amostras com a média histórica,
    em unidades de desvio padrão. |z| ≥ ``threshold`` indica distúrbio ativo.
    """
    s = series.dropna()
    if len(s) < window + 2 or s.std() == 0:
        return CurrentStatus(z=0.0, disturbed=False, direction=0)
    z = float((s.iloc[-window:].mean() - s.mean()) / s.std())
    return CurrentStatus(
        z=z, disturbed=abs(z) >= threshold, direction=int(np.sign(z))
    )


@dataclass
class StabilityResult:
    mean: float
    std: float
    cv: float             # coeficiente de variação σ/|μ| (fração; nan se μ≈0)
    range_rel: float      # amplitude relativa (max−min)/|μ|
    trend: int            # sinal da tendência líquida no período (+1/-1/0)
    classification: str   # 'Estável' | 'Moderada' | 'Instável' | '—'
    strong: bool          # variação forte (cv ≥ strong_cv)


def overall_stability(
    series: pd.Series, moderate_cv: float = 0.05, strong_cv: float = 0.15
) -> StabilityResult:
    """Avalia a estabilidade de um indicador no PERÍODO TODO analisado.

    Usa o **coeficiente de variação** ``CV = σ / |μ|`` (desvio padrão sobre o
    módulo da média) — métrica padrão de estabilidade de processo: mede o quanto o
    indicador oscilou em relação ao seu nível típico ao longo de toda a série.

    Classificação (padrão): ``CV < 5%`` Estável; ``5%–15%`` Moderada; ``≥ 15%``
    Instável. ``strong`` indica variação forte (``CV ≥ strong_cv``), usada nos
    gatilhos. ``trend`` é o sinal da diferença entre a média do último e do
    primeiro terço do período (tendência líquida).

    Observação: o CV não é uma medida válida quando a série cruza o zero (média
    próxima de zero / sinal que muda); nesses casos a classificação retorna '—'.
    """
    s = series.dropna()
    if len(s) < 3:
        return StabilityResult(float("nan"), 0.0, float("nan"),
                               float("nan"), 0, "—", False)

    mean, std = float(s.mean()), float(s.std())
    # CV só é interpretável para séries que não cruzam o zero (nível bem definido)
    cv_valid = float(s.min()) > 0 or float(s.max()) < 0
    cv = std / abs(mean) if cv_valid else float("nan")
    range_rel = (float(s.max()) - float(s.min())) / abs(mean) if cv_valid else float("nan")

    k = max(1, len(s) // 3)
    trend = int(np.sign(s.iloc[-k:].mean() - s.iloc[:k].mean()))

    if np.isnan(cv):
        classification, strong = "—", False
    elif cv < moderate_cv:
        classification, strong = "Estável", False
    elif cv < strong_cv:
        classification, strong = "Moderada", False
    else:
        classification, strong = "Instável", True

    return StabilityResult(mean, std, cv, range_rel, trend, classification, strong)


# --------------------------------------------------------------------------- #
# 7. Atribuição rigorosa de variância (Shapley/LMG) + multicolinearidade
# --------------------------------------------------------------------------- #
def build_design_matrix(
    df: pd.DataFrame, key: str, results: dict[str, LagResult]
) -> tuple[pd.DataFrame, pd.Series]:
    """Monta a matriz de preditores alinhados pelo lag ótimo de cada um.

    Cada indicador é deslocado pela sua própria defasagem ótima
    (``res.best_lag_samples``), de modo que cada coluna fica temporalmente
    alinhada com o efeito no indicador-chave. Linhas com NaN são descartadas.

    Retorna ``(X, y)`` onde ``X`` são os indicadores alinhados e ``y`` é a chave.
    """
    cols = {ind: df[ind].shift(res.best_lag_samples) for ind, res in results.items()}
    aligned = pd.DataFrame(cols, index=df.index)
    data = pd.concat([df[key].rename("__key__"), aligned], axis=1).dropna()
    y = data["__key__"]
    X = data.drop(columns="__key__")
    return X, y


def _r2(X: np.ndarray, y: np.ndarray) -> float:
    """R² de uma regressão linear de y sobre X (≥ 0; 0 se X vazio)."""
    if X.shape[1] == 0:
        return 0.0
    model = LinearRegression().fit(X, y)
    return max(float(model.score(X, y)), 0.0)


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Fator de Inflação de Variância (VIF) de cada coluna.

    ``VIF_i = 1 / (1 - R²_i)``, com ``R²_i`` da regressão da coluna *i* contra as
    demais. VIF alto (> 5–10) indica multicolinearidade forte.
    """
    Xv = X.to_numpy(dtype=float)
    vifs = {}
    for i, col in enumerate(X.columns):
        others = np.delete(Xv, i, axis=1)
        r2 = _r2(others, Xv[:, i])
        vifs[col] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs)


def reduce_multicollinearity(
    X: pd.DataFrame,
    y: pd.Series,
    corr_threshold: float = 0.9,
    vif_threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove indicadores redundantes (multicolinearidade), mantendo representantes.

    1. **Cluster por correlação:** colunas com ``|corr| ≥ corr_threshold`` entre si
       formam um grupo; mantém-se a mais correlacionada com ``y`` e removem-se as outras.
    2. **VIF iterativo:** enquanto ``max(VIF) > vif_threshold``, remove-se a de maior VIF.

    Retorna ``(X_reduzido, removidos)`` onde ``removidos`` tem colunas
    ``Removido``, ``Mantido``, ``Motivo``.
    """
    removed: list[dict] = []
    if X.shape[1] <= 1:
        return X, pd.DataFrame(columns=["Removido", "Mantido", "Motivo"])

    # Relevância de cada coluna com o alvo (|corr| com y), para escolher representantes
    rel = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() else 0.0)
    order = rel.sort_values(ascending=False).index.tolist()

    # 1) Cluster por correlação entre preditores
    corr = X.corr().abs()
    keep, assigned = [], set()
    for f in order:
        if f in assigned:
            continue
        keep.append(f)
        assigned.add(f)
        for g in order:
            if g != f and g not in assigned and corr.loc[f, g] >= corr_threshold:
                assigned.add(g)
                removed.append({
                    "Removido": g, "Mantido": f,
                    "Motivo": f"|corr| {corr.loc[f, g]:.2f} ≥ {corr_threshold:g} com {f}",
                })
    Xk = X[keep]

    # 2) VIF iterativo no conjunto mantido
    while Xk.shape[1] > 1:
        vif = compute_vif(Xk)
        worst, worst_vif = vif.idxmax(), vif.max()
        if worst_vif <= vif_threshold:
            break
        removed.append({
            "Removido": worst, "Mantido": "—",
            "Motivo": f"VIF {worst_vif:.1f} > {vif_threshold:g}",
        })
        Xk = Xk.drop(columns=worst)

    return Xk, pd.DataFrame(removed, columns=["Removido", "Mantido", "Motivo"])


def relative_importance(
    X: pd.DataFrame, y: pd.Series, max_exact: int = 10, n_perm: int = 400, seed: int = 0
) -> tuple[dict[str, float], float]:
    """Decomposição de Shapley/LMG do R² entre os preditores.

    A fatia de cada preditor é a média da sua contribuição marginal
    ``R²(S∪{i}) − R²(S)`` sobre todos os subconjuntos ``S`` que não o contêm. As
    fatias são **não-negativas, não-sobrepostas e somam o R² do modelo completo**.

    - ``p ≤ max_exact``: cálculo exato (todos os 2^p subconjuntos, com cache).
    - ``p > max_exact``: aproximação por ``n_perm`` permutações aleatórias.

    Retorna ``(shares, model_r2)`` com ``shares`` em unidades de R² (fração da
    variância de ``y``).
    """
    feats = list(X.columns)
    p = len(feats)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)

    cache: dict[tuple[int, ...], float] = {}

    def r2(subset: tuple[int, ...]) -> float:
        if subset not in cache:
            cache[subset] = _r2(Xv[:, list(subset)], yv) if subset else 0.0
        return cache[subset]

    shares = {f: 0.0 for f in feats}
    if p == 0:
        return shares, 0.0

    if p <= max_exact:
        idx = range(p)
        for i in idx:
            others = [j for j in idx if j != i]
            contrib = 0.0
            for k in range(len(others) + 1):
                w = factorial(k) * factorial(p - k - 1) / factorial(p)
                for S in itertools.combinations(others, k):
                    contrib += w * (r2(tuple(sorted(S + (i,)))) - r2(tuple(sorted(S))))
            shares[feats[i]] = contrib
    else:
        rng = np.random.default_rng(seed)
        base = list(range(p))
        for _ in range(n_perm):
            cur: tuple[int, ...] = ()
            prev = 0.0
            for j in rng.permutation(base):
                nxt = tuple(sorted(cur + (int(j),)))
                val = r2(nxt)
                shares[feats[int(j)]] += val - prev
                cur, prev = nxt, val
        for f in shares:
            shares[f] /= n_perm

    return shares, r2(tuple(range(p)))


@dataclass
class AttributionResult:
    table: pd.DataFrame       # ranking por fatia de variância (ver colunas abaixo)
    model_r2: float           # R² do modelo completo (variância total explicada)
    n_samples: int            # nº de linhas usadas (após alinhar lags e dropna)
    removed: pd.DataFrame      # indicadores removidos por redundância


def robust_attribution(
    df: pd.DataFrame,
    key: str,
    results: dict[str, LagResult],
    remove_collinear: bool = True,
    corr_threshold: float = 0.9,
    vif_threshold: float = 10.0,
    max_exact: int = 10,
    n_perm: int = 400,
) -> AttributionResult:
    """Atribuição rigorosa: fatia real (não-sobreposta) da variância por driver.

    Encadeia: matriz alinhada por lag → (opcional) remoção de multicolinearidade →
    decomposição de Shapley/LMG. A tabela traz, por indicador mantido:
    ``Indicador``, ``Lag (min)``, ``Sentido``, ``Fatia da variância (%)``
    (= share×100), ``Contribuição no modelo (%)`` (= share/model_r2×100) e ``VIF``.

    ``remove_collinear``: se ``False``, mantém todos os indicadores (sem eliminar
    redundâncias) — o VIF continua sendo reportado como alerta.
    """
    X, y = build_design_matrix(df, key, results)
    cols = ["Indicador", "Lag (min)", "Sentido",
            "Fatia da variância (%)", "Contribuição no modelo (%)", "VIF"]

    if X.shape[1] == 0 or len(y) < 5:
        empty = pd.DataFrame(columns=cols)
        return AttributionResult(empty, 0.0, len(y), pd.DataFrame())

    if remove_collinear:
        Xk, removed = reduce_multicollinearity(X, y, corr_threshold, vif_threshold)
    else:
        Xk = X
        removed = pd.DataFrame(columns=["Removido", "Mantido", "Motivo"])
    shares, model_r2 = relative_importance(Xk, y, max_exact=max_exact, n_perm=n_perm)
    vif = compute_vif(Xk) if Xk.shape[1] > 1 else pd.Series({c: 1.0 for c in Xk.columns})

    rows = []
    for ind in Xk.columns:
        res = results[ind]
        corr = res.corr_at_lag
        rows.append({
            "Indicador": ind,
            "Lag (min)": res.best_lag_minutes,
            "Sentido": "Direto ↑" if (np.isnan(corr) or corr >= 0) else "Inverso ↓",
            "Fatia da variância (%)": round(shares[ind] * 100, 1),
            "Contribuição no modelo (%)": round(shares[ind] / model_r2 * 100, 1)
            if model_r2 > 0 else 0.0,
            "VIF": round(float(vif.get(ind, 1.0)), 2),
        })
    table = (
        pd.DataFrame(rows, columns=cols)
        .sort_values("Fatia da variância (%)", ascending=False)
        .reset_index(drop=True)
    )
    return AttributionResult(table, float(model_r2), len(y), removed)


# =========================================================================== #
#                                                                             #
#   VERSÃO 2 — PROCESSO (minuto a minuto) × PRODUÇÃO DIÁRIA                    #
#   --------------------------------------------------------------            #
#   Transforma os dados de processo minuto a minuto em variáveis              #
#   explicativas DIÁRIAS, considera efeitos de lag em DIAS e correlaciona/    #
#   modela contra um indicador de produção informado uma vez ao dia.          #
#   Aponta os indicadores com maior EVIDÊNCIA ESTATÍSTICA de influência       #
#   (prováveis causas — sem afirmar causalidade absoluta).                    #
#                                                                             #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# A.0  Helpers de texto/numérico
# --------------------------------------------------------------------------- #
def _clean_token(name) -> str:
    """Padroniza um nome de coluna: sem acento, sem espaço, só [A-Za-z0-9_]."""
    s = str(name).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    return s or "col"


def _standardize_names(cols) -> list[str]:
    """Padroniza uma lista de nomes, evitando duplicatas (sufixo _2, _3, ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        base = _clean_token(c)
        if base in seen:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 1
            out.append(base)
    return out


def _coerce_numeric_brl(series: pd.Series) -> pd.Series:
    """Converte uma série de texto para numérico tratando decimal BR (vírgula)."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    s = (
        series.astype(str).str.strip()
        .str.replace(".", "", regex=False)   # separador de milhar
        .str.replace(",", ".", regex=False)   # vírgula decimal -> ponto
    )
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------- #
# A.1  Leitura de arquivo com múltiplas abas
# --------------------------------------------------------------------------- #
def list_sheets(file) -> list[str]:
    """Lista as abas de um Excel. Retorna [] para CSV (que não tem abas)."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return []
    try:
        return list(pd.ExcelFile(file).sheet_names)
    except Exception:  # noqa: BLE001
        return []


def read_sheet(file, sheet_name: str | None = None) -> pd.DataFrame:
    """Lê uma aba específica (Excel) ou o CSV inteiro, SEM forçar índice de tempo.

    A coluna de data/hora e as colunas de indicadores são escolhidas depois, na
    interface. A coerção numérica (decimal BR) é feita nas etapas seguintes,
    preservando aqui as colunas de data como texto/datetime.
    """
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")


# --------------------------------------------------------------------------- #
# A.2  Engenharia de variáveis diárias (estatística, instabilidade, faixa, turno)
# --------------------------------------------------------------------------- #
# Sufixos das estatísticas geradas por indicador (para preencher NaN em dias vazios)
_DAILY_SUFFIXES = [
    "mean", "median", "min", "max", "range", "std", "cv",
    "p10", "p25", "p75", "p90", "sum",
    "mean_abs_diff", "max_rate", "n_oscilacoes",
]


def estimate_step_minutes(dt: pd.Series) -> float:
    """Estima o passo temporal (mediana das diferenças) em minutos."""
    vals = np.sort(pd.to_datetime(dt).dropna().values)
    if len(vals) < 2:
        return 1.0
    diffs = np.diff(vals).astype("timedelta64[s]").astype(float)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) / 60.0 if len(diffs) else 1.0


def _window_feature_record(
    gdf: pd.DataFrame, ind_cols: list, limits: dict, step_min: float,
    n_expected: int, osc_k: float = 3.0, use_turnos: bool = False,
) -> dict:
    """Calcula o dicionário de atributos de UMA janela (um dia ou um período).

    ``gdf`` é o recorte da base de processo na janela (deve conter ``__dt__`` para
    os turnos). Reutilizado por ``build_daily_features`` (dia) e
    ``build_period_features`` (período sazonal)."""
    rec: dict = {}
    for c in ind_cols:
        base = _clean_token(c)
        arr = gdf[c].dropna().to_numpy(dtype=float)
        n_valid = arr.size
        rec[f"{base}_n_valid"] = n_valid
        rec[f"{base}_pct_missing"] = max(0.0, 1 - n_valid / n_expected) * 100.0
        if n_valid == 0:
            for suf in _DAILY_SUFFIXES:
                rec[f"{base}_{suf}"] = np.nan
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n_valid > 1 else 0.0
            mn, mx = float(np.min(arr)), float(np.max(arr))
            rec[f"{base}_mean"] = mean
            rec[f"{base}_median"] = float(np.median(arr))
            rec[f"{base}_min"] = mn
            rec[f"{base}_max"] = mx
            rec[f"{base}_range"] = mx - mn
            rec[f"{base}_std"] = std
            rec[f"{base}_cv"] = std / abs(mean) if mean != 0 else np.nan
            rec[f"{base}_p10"] = float(np.percentile(arr, 10))
            rec[f"{base}_p25"] = float(np.percentile(arr, 25))
            rec[f"{base}_p75"] = float(np.percentile(arr, 75))
            rec[f"{base}_p90"] = float(np.percentile(arr, 90))
            rec[f"{base}_sum"] = float(np.sum(arr))
            diff = np.diff(arr)
            if diff.size:
                rec[f"{base}_mean_abs_diff"] = float(np.mean(np.abs(diff)))
                rec[f"{base}_max_rate"] = float(np.max(np.abs(diff)))
                sdiff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
                thr = osc_k * sdiff
                rec[f"{base}_n_oscilacoes"] = (
                    int(np.sum(np.abs(diff) > thr)) if thr > 0 else 0
                )
            else:
                rec[f"{base}_mean_abs_diff"] = 0.0
                rec[f"{base}_max_rate"] = 0.0
                rec[f"{base}_n_oscilacoes"] = 0

        # Tempo fora de faixa (apenas indicadores com limite configurado)
        if c in limits and n_valid > 0:
            lo, hi = limits[c]
            has_lo = lo is not None and not (isinstance(lo, float) and np.isnan(lo))
            has_hi = hi is not None and not (isinstance(hi, float) and np.isnan(hi))
            out_mask = np.zeros(arr.shape, dtype=bool)
            if has_lo:
                below = arr < lo
                out_mask |= below
                rec[f"{base}_min_abaixo"] = float(np.sum(below) * step_min)
                rec[f"{base}_area_abaixo"] = float(np.sum((lo - arr)[below]))
            if has_hi:
                above = arr > hi
                out_mask |= above
                rec[f"{base}_min_acima"] = float(np.sum(above) * step_min)
                rec[f"{base}_area_acima"] = float(np.sum((arr - hi)[above]))
            rec[f"{base}_pct_fora"] = float(np.sum(out_mask) / n_valid * 100.0)

    # Turnos (00–08, 08–16, 16–24) — relevante sobretudo para período diário
    if use_turnos:
        hours = gdf["__dt__"].dt.hour
        for tname, lh, hh in [("t1", 0, 8), ("t2", 8, 16), ("t3", 16, 24)]:
            sub = gdf.loc[(hours >= lh) & (hours < hh)]
            for c in ind_cols:
                base = _clean_token(c)
                xs = sub[c].dropna().to_numpy(dtype=float)
                rec[f"{base}_mean_{tname}"] = float(np.mean(xs)) if xs.size else np.nan
                rec[f"{base}_std_{tname}"] = (
                    float(np.std(xs, ddof=1)) if xs.size > 1
                    else (0.0 if xs.size == 1 else np.nan)
                )
    return rec


def build_daily_features(
    proc_df: pd.DataFrame,
    dt_col: str,
    *,
    limits: dict | None = None,
    use_turnos: bool = False,
    osc_k: float = 3.0,
) -> tuple[pd.DataFrame, float]:
    """Transforma a base de processo minuto a minuto numa base DIÁRIA.

    Para cada indicador numérico, agrega por dia gerando estatísticas (média,
    mediana, mín, máx, amplitude, desvio padrão, CV, percentis, soma, nº de
    registros válidos, % de ausentes), variáveis de instabilidade (Δ médio entre
    leituras, taxa máxima de variação, nº de oscilações bruscas) e, para os
    indicadores com limite configurado em ``limits={ind: (lo, hi)}``, o tempo e a
    área fora de faixa. Com ``use_turnos`` acrescenta média/desvio por turno
    (00–08, 08–16, 16–24).

    Retorna ``(daily_df, step_minutes)``; ``daily_df`` é indexado por dia
    (Timestamp à meia-noite, nome do índice = ``data``).
    """
    limits = limits or {}
    df = proc_df.copy()
    dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df["__dt__"] = dt[dt.notna()].values
    df = df.sort_values("__dt__")

    # Indicadores = todas as colunas menos a de data/hora; coage para numérico (BR)
    ind_cols = [c for c in proc_df.columns if c != dt_col]
    for c in ind_cols:
        df[c] = _coerce_numeric_brl(df[c])
    ind_cols = [c for c in ind_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not ind_cols:
        raise ValueError("Nenhum indicador numérico encontrado na base de processo.")

    step_min = estimate_step_minutes(df["__dt__"])
    n_expected = max(1, int(round((24 * 60) / step_min)))  # leituras esperadas/dia
    df["__date__"] = df["__dt__"].dt.normalize()

    records: list[dict] = []
    for date, gdf in df.groupby("__date__"):
        rec: dict = {"data": date}
        rec.update(_window_feature_record(
            gdf, ind_cols, limits, step_min, n_expected, osc_k, use_turnos))
        records.append(rec)

    daily = pd.DataFrame(records).set_index("data").sort_index()
    daily.index.name = "data"
    return daily, step_min


# --------------------------------------------------------------------------- #
# A.3  Variáveis com lag em dias (D0, D-1, ..., D-N)
# --------------------------------------------------------------------------- #
def add_lag_features(daily_df: pd.DataFrame, max_lag_days: int) -> pd.DataFrame:
    """Cria, para cada variável diária, as defasagens D0..D-N.

    ``{col}_lag_0d`` = mesmo dia (D0); ``{col}_lag_1d`` = dia anterior (D-1); etc.
    """
    df = daily_df.sort_index()
    frames = []
    for k in range(0, int(max_lag_days) + 1):
        shifted = df.shift(k)
        shifted.columns = [f"{c}_lag_{k}d" for c in df.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1)


# --------------------------------------------------------------------------- #
# A.3b  ANÁLISE SAZONAL — período definido pelos horários do indicador sazonal
#       (cadência diária OU sub-diária) e lag configurado em MINUTOS.
# --------------------------------------------------------------------------- #
def parse_target_series(
    prod_df: pd.DataFrame, date_col: str, target_col: str
) -> tuple[pd.Series, float]:
    """Lê o indicador sazonal como série temporal e detecta o período (em minutos).

    Retorna ``(serie, period_min)`` onde ``serie`` é indexada pelos instantes do
    indicador (âncoras de período) e ``period_min`` é a mediana do espaçamento
    entre leituras consecutivas (24 h → 1440 min; a cada 4 h → 240 min; etc.).
    """
    p = prod_df.copy()
    tdt = pd.to_datetime(p[date_col], dayfirst=True, errors="coerce")
    val = _coerce_numeric_brl(p[target_col])
    s = pd.Series(np.asarray(val, dtype=float), index=tdt)
    s = s[s.index.notna()].dropna()
    s = s.groupby(level=0).mean().sort_index()
    if len(s) < 2:
        raise ValueError("São necessárias ao menos 2 leituras do indicador sazonal.")
    diffs = np.diff(s.index.values).astype("timedelta64[s]").astype(float)
    diffs = diffs[diffs > 0]
    period_min = float(np.median(diffs)) / 60.0 if len(diffs) else 1440.0
    s.name = "Sazonal"
    s.index.name = "periodo"
    return s, period_min


def build_period_features(
    proc_df: pd.DataFrame,
    dt_col: str,
    anchors,
    period_min: float,
    *,
    limits: dict | None = None,
    use_turnos: bool = False,
    osc_k: float = 3.0,
) -> tuple[pd.DataFrame, float]:
    """Agrega o processo minuto a minuto na janela de cada período sazonal.

    Para cada âncora ``T_i`` (instante do indicador sazonal), a janela é
    ``(T_{i-1}, T_i]`` (a primeira é ``(T_1 − período, T_1]``). Gera os mesmos
    atributos por indicador de ``build_daily_features``. Indexado pelas âncoras.

    Retorna ``(period_df, step_minutes)``.
    """
    limits = limits or {}
    df = proc_df.copy()
    dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df["__dt__"] = dt[dt.notna()].values
    df = df.sort_values("__dt__").reset_index(drop=True)

    ind_cols = [c for c in proc_df.columns if c != dt_col]
    for c in ind_cols:
        df[c] = _coerce_numeric_brl(df[c])
    ind_cols = [c for c in ind_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not ind_cols:
        raise ValueError("Nenhum indicador numérico encontrado na base de processo.")

    step_min = estimate_step_minutes(df["__dt__"])
    n_expected = max(1, int(round(period_min / step_min)))
    period = pd.Timedelta(minutes=period_min)

    anchors = pd.DatetimeIndex(pd.to_datetime(anchors)).sort_values()
    times = df["__dt__"].to_numpy(dtype="datetime64[ns]")

    records, index = [], []
    for i, T in enumerate(anchors):
        start = anchors[i - 1] if i > 0 else (T - period)
        lo = int(np.searchsorted(times, np.datetime64(pd.Timestamp(start)), side="right"))
        hi = int(np.searchsorted(times, np.datetime64(pd.Timestamp(T)), side="right"))
        gdf = df.iloc[lo:hi]
        records.append(_window_feature_record(
            gdf, ind_cols, limits, step_min, n_expected, osc_k, use_turnos))
        index.append(T)

    out = pd.DataFrame(records, index=pd.DatetimeIndex(index, name="periodo"))
    return out, step_min


def add_lag_features_min(
    period_df: pd.DataFrame, max_lag_min: float, period_min: float
) -> pd.DataFrame:
    """Cria blocos de defasagem de PERÍODOS anteriores, com o máximo em minutos.

    Nº de blocos = ``floor(max_lag_min / period_min)``. O bloco do período ATUAL
    mantém o nome original (sem sufixo); os blocos defasados recebem
    ``{col}_lag_{m}min`` (``m = k·period_min``, m > 0).
    """
    df = period_df.sort_index()
    period_min = period_min if period_min and period_min > 0 else 1.0
    n_lags = int(np.floor(max_lag_min / period_min)) if max_lag_min > 0 else 0
    frames = []
    for k in range(0, n_lags + 1):
        m = int(round(k * period_min))
        shifted = df.shift(k)
        shifted.columns = (list(df.columns) if m == 0
                           else [f"{c}_lag_{m}min" for c in df.columns])
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def merge_with_target(
    period_feat: pd.DataFrame, target_series: pd.Series, target: str = "Sazonal"
) -> pd.DataFrame:
    """Junta as variáveis por período com o indicador sazonal (alinhado pelas âncoras)."""
    ts = target_series.rename(target)
    return period_feat.join(ts, how="inner")


# --------------------------------------------------------------------------- #
# A.4  Junção com a produção diária + limpeza da base de modelagem
# --------------------------------------------------------------------------- #
def merge_with_production(
    daily_feat: pd.DataFrame,
    prod_df: pd.DataFrame,
    prod_date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Junta a base diária de processo com a produção diária (alvo = ``Producao``)."""
    p = prod_df.copy()
    pdate = pd.to_datetime(p[prod_date_col], dayfirst=True, errors="coerce")
    target = _coerce_numeric_brl(p[target_col])
    prod = pd.DataFrame(
        {"data": pdate.dt.normalize().values, "Producao": target.values}
    ).dropna(subset=["data"])
    prod_daily = prod.groupby("data")["Producao"].mean()
    out = daily_feat.join(prod_daily, how="inner")
    return out


def clean_modeling_table(
    df: pd.DataFrame, *, target: str = "Sazonal", max_null_frac: float = 0.4
) -> tuple[pd.DataFrame, dict]:
    """Trata a base de modelagem (req. 12).

    - coerção numérica; remoção de colunas constantes; remoção de colunas com
      excesso de nulos (> ``max_null_frac``); imputação dos nulos restantes pela
      mediana; descarte de linhas sem o alvo; padronização dos nomes.

    Retorna ``(df_limpo, relatorio)`` com as listas de colunas afetadas.
    """
    work = df.copy()
    report = {"nao_numericas": [], "muitos_nulos": [], "constantes": []}

    for c in work.columns:
        if not pd.api.types.is_numeric_dtype(work[c]):
            report["nao_numericas"].append(c)
            work[c] = _coerce_numeric_brl(work[c])

    if target in work.columns:
        work = work.dropna(subset=[target])

    feat_cols = [c for c in work.columns if c != target]
    if feat_cols:
        null_frac = work[feat_cols].isna().mean()
        drop_null = null_frac[null_frac > max_null_frac].index.tolist()
        report["muitos_nulos"] = drop_null
        work = work.drop(columns=drop_null)

    feat_cols = [c for c in work.columns if c != target]
    if feat_cols:
        work[feat_cols] = work[feat_cols].fillna(work[feat_cols].median())
        nunique = work[feat_cols].nunique()
        const = nunique[nunique <= 1].index.tolist()
        report["constantes"] = const
        work = work.drop(columns=const)

    # Padroniza nomes (mantém 'Producao' inalterado, pois já é ascii/sem espaço)
    work.columns = _standardize_names(work.columns)
    return work, report


def split_features_target(
    df: pd.DataFrame, target: str = "Sazonal"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a base limpa em ``(X, y)`` mantendo o índice (dia ou período)."""
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    return X, y


# --------------------------------------------------------------------------- #
# A.5  Análise estatística (Pearson, Spearman, p-value, Mutual Information)
# --------------------------------------------------------------------------- #
def _mi_value(x: np.ndarray, y: np.ndarray) -> float:
    if not _HAS_SKLEARN or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])


def statistical_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Ranking de associação por variável: Pearson/Spearman (+p-value) e MI."""
    yv = y.to_numpy(dtype=float)
    rows = []
    for c in X.columns:
        xv = X[c].to_numpy(dtype=float)
        mask = np.isfinite(xv) & np.isfinite(yv)
        if mask.sum() < 3 or np.std(xv[mask]) == 0:
            continue
        xa, ya = xv[mask], yv[mask]
        if _HAS_SCIPY:
            pr, pp = _scipy_stats.pearsonr(xa, ya)
            sr, sp = _scipy_stats.spearmanr(xa, ya)
        else:
            pr = float(np.corrcoef(xa, ya)[0, 1])
            sr = float(pd.Series(xa).corr(pd.Series(ya), method="spearman"))
            pp = sp = float("nan")
        rows.append({
            "Variável": c,
            "Pearson r": round(float(pr), 4),
            "Pearson p": round(float(pp), 4),
            "Spearman r": round(float(sr), 4),
            "Spearman p": round(float(sp), 4),
            "Mutual Information": round(_mi_value(xa, ya), 4),
            "Força |Spearman|": round(abs(float(sr)), 4),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Força |Spearman|", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# A.6  Modelo preditivo (Random Forest / XGBoost) com split temporal
# --------------------------------------------------------------------------- #
@dataclass
class ModelResult:
    model: object
    model_name: str
    feature_names: list
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_pred_train: np.ndarray
    metrics: dict
    importances: pd.Series


def temporal_split(
    X: pd.DataFrame, y: pd.Series, test_frac: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa treino/teste RESPEITANDO A ORDEM TEMPORAL (sem embaralhar)."""
    n = len(X)
    n_test = max(1, int(round(n * test_frac)))
    n_train = max(1, n - n_test)
    return X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nz = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)
        if nz.any() else float("nan")
    )
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "RandomForest",
    test_frac: float = 0.25,
    random_state: int = 0,
) -> ModelResult:
    """Treina Random Forest (padrão) ou XGBoost (se instalado) com split temporal."""
    if not _HAS_RF:
        raise RuntimeError("scikit-learn não está instalado (RandomForest indisponível).")
    if len(X) < 6:
        raise ValueError("Amostras diárias insuficientes para treinar o modelo (mín. 6).")

    Xtr, Xte, ytr, yte = temporal_split(X, y, test_frac)
    if model_type == "XGBoost" and _HAS_XGB:
        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=random_state, n_jobs=-1,
        )
        name = "XGBoost"
    else:
        model = RandomForestRegressor(
            n_estimators=400, random_state=random_state, n_jobs=-1
        )
        name = "RandomForest"

    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)
    y_pred_train = model.predict(Xtr)
    metrics = _regression_metrics(yte.to_numpy(dtype=float), np.asarray(y_pred, dtype=float))
    imp = pd.Series(
        getattr(model, "feature_importances_", np.zeros(X.shape[1])),
        index=X.columns,
    ).sort_values(ascending=False)
    return ModelResult(
        model=model, model_name=name, feature_names=list(X.columns),
        y_train=ytr, y_test=yte, y_pred=np.asarray(y_pred, dtype=float),
        y_pred_train=np.asarray(y_pred_train, dtype=float),
        metrics=metrics, importances=imp,
    )


# --------------------------------------------------------------------------- #
# A.7  Explicabilidade (SHAP)
# --------------------------------------------------------------------------- #
def shap_importance(model_result: ModelResult, X: pd.DataFrame) -> pd.Series | None:
    """Ranking por |SHAP| médio (impacto médio absoluto). ``None`` se SHAP ausente."""
    if not _HAS_SHAP:
        return None
    try:
        explainer = _shap.TreeExplainer(model_result.model)
        sv = explainer.shap_values(X)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
    return pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# A.8  Ranking consolidado + diagnóstico gerencial
# --------------------------------------------------------------------------- #
def _norm_abs(s: pd.Series) -> pd.Series:
    v = s.abs()
    mx = v.max()
    return v / mx if mx and not np.isnan(mx) else v * 0.0


def consolidated_ranking(
    stat_df: pd.DataFrame,
    importances: pd.Series | None = None,
    shap_imp: pd.Series | None = None,
) -> pd.DataFrame:
    """Combina Spearman, Mutual Information, importância do modelo e SHAP.

    Cada fonte é normalizada (0–1 por seu máximo absoluto) e o ``Score
    consolidado`` é a média das fontes disponíveis. Ordenado do maior para o menor.
    """
    if stat_df is not None and not stat_df.empty:
        base = stat_df.set_index("Variável")[["Spearman r", "Mutual Information"]].copy()
        base.columns = ["Spearman", "Mutual Information"]
    else:
        base = pd.DataFrame()

    if importances is not None:
        base = base.join(importances.rename("Importância modelo"), how="outer")
    if shap_imp is not None:
        base = base.join(shap_imp.rename("SHAP"), how="outer")

    if base.empty:
        return pd.DataFrame(columns=["Variável", "Score consolidado"])

    norm_cols = []
    for col in ["Spearman", "Mutual Information", "Importância modelo", "SHAP"]:
        if col in base.columns:
            base[f"__n_{col}"] = _norm_abs(base[col])
            norm_cols.append(f"__n_{col}")
    base["Score consolidado"] = base[norm_cols].mean(axis=1, skipna=True).round(4)
    base = base.drop(columns=norm_cols)
    out = (
        base.sort_values("Score consolidado", ascending=False)
        .reset_index()
        .rename(columns={"index": "Variável"})
    )
    return out


def _theme_of(name: str) -> str | None:
    n = name.lower()
    if any(t in n for t in ("_std", "_cv", "n_oscilacoes", "max_rate", "mean_abs_diff", "_range")):
        return "instabilidade operacional"
    if any(t in n for t in ("_fora", "area_", "_abaixo", "_acima")):
        return "tempo fora da faixa ideal"
    return None


def generate_diagnosis(consolidated: pd.DataFrame, model_result: ModelResult | None,
                       sheet_names: list[str] | None = None) -> str:
    """Gera um diagnóstico em linguagem gerencial (req. 17)."""
    if consolidated is None or consolidated.empty:
        return "Não há evidência estatística suficiente para um diagnóstico."

    top = consolidated.head(3)["Variável"].tolist()
    themes = {t for t in (_theme_of(v) for v in top) if t}
    # lag defasado: novo formato em minutos (_lag_<m>min, m>0) ou legado em dias
    lag_terms = [v for v in top
                 if re.search(r"_lag_([1-9]\d*)min", v) or re.search(r"_lag_([1-9]\d*)d", v)]

    nomes = ", ".join(f"**{humanize_variable(v, sheet_names)}**" for v in top)
    txt = f"O indicador sazonal apresentou maior associação com {nomes}. "

    if themes:
        txt += (
            "Os resultados indicam que a variação está mais relacionada à "
            + " e ao ".join(sorted(themes)) + ". "
        )
    else:
        txt += (
            "Os resultados indicam associação principalmente com o nível médio "
            "desses indicadores. "
        )

    if lag_terms:
        txt += (
            "Há indícios de **efeito defasado** (comportamento de períodos anteriores "
            "influenciando o valor atual). "
        )

    if model_result is not None and not np.isnan(model_result.metrics.get("R2", np.nan)):
        txt += (
            f"O modelo {model_result.model_name} explicou cerca de "
            f"**{max(0.0, model_result.metrics['R2']) * 100:.0f}%** da variação do "
            "indicador sazonal no período de teste. "
        )

    txt += (
        "Recomenda-se investigar estes indicadores no(s) período(s) anterior(es) ao "
        "fechamento do indicador sazonal. Estes são **indicadores com maior evidência "
        "estatística de influência** — prováveis causas, não causalidade comprovada."
    )
    return txt


# --------------------------------------------------------------------------- #
# A.9  Diagnóstico de um dia específico
# --------------------------------------------------------------------------- #
def day_contributions(
    model_result: ModelResult, X: pd.DataFrame, date, top_n: int = 10
) -> pd.DataFrame:
    """Principais variáveis que contribuíram para a previsão de um dia.

    Usa SHAP da linha do dia quando disponível; caso contrário, usa o desvio
    padronizado vs. a média histórica ponderado pela importância do modelo.
    Acompanha valor do dia × média histórica de cada variável.
    """
    if date not in X.index:
        return pd.DataFrame(
            columns=["Variável", "Valor no dia", "Média histórica", "Desvio (σ)", "Contribuição"]
        )

    row = X.loc[[date]]
    contrib = None
    if _HAS_SHAP:
        try:
            explainer = _shap.TreeExplainer(model_result.model)
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[0]
            contrib = pd.Series(np.asarray(sv)[0], index=X.columns)
        except Exception:  # noqa: BLE001
            contrib = None

    mean = X.mean()
    std = X.std().replace(0, np.nan)
    z = ((row.iloc[0] - mean) / std).fillna(0.0)
    if contrib is None:
        imp = model_result.importances.reindex(X.columns).fillna(0.0)
        contrib = z * imp

    out = pd.DataFrame({
        "Variável": list(X.columns),
        "Valor no dia": row.iloc[0].to_numpy(dtype=float),
        "Média histórica": mean.to_numpy(dtype=float),
        "Desvio (σ)": z.to_numpy(dtype=float),
        "Contribuição": np.asarray(contrib, dtype=float),
    })
    out["__abs__"] = out["Contribuição"].abs()
    out = (
        out.sort_values("__abs__", ascending=False)
        .drop(columns="__abs__")
        .head(top_n)
        .reset_index(drop=True)
    )
    for col in ("Valor no dia", "Média histórica", "Desvio (σ)", "Contribuição"):
        out[col] = out[col].round(4)
    return out


# --------------------------------------------------------------------------- #
# A.10  Exportação Excel multi-aba
# --------------------------------------------------------------------------- #
def to_excel_multi(sheets: dict) -> bytes:
    """Serializa um dicionário ``{nome_aba: DataFrame|texto}`` em um Excel multi-aba."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, data in sheets.items():
            safe = str(name)[:31]
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=safe, index=False)
            else:
                pd.DataFrame({"conteudo": [str(data)]}).to_excel(
                    writer, sheet_name=safe, index=False
                )
    return buf.getvalue()


# =========================================================================== #
#                                                                             #
#   SEÇÃO B — PLANILHA MULTI-ABAS (contínuas + periódicos + alvo),            #
#   excursões de limites críticos e texto do relatório final                  #
#   --------------------------------------------------------------           #
#   O usuário classifica cada aba do Excel como variáveis CONTÍNUAS           #
#   (minuto a minuto), PERIÓDICAS (informadas a cada x horas — laboratório    #
#   OU qualquer leitura periódica; o valor em T representa a média da janela  #
#   (T−x, T]) ou ALVO.                                                        #
#                                                                             #
# =========================================================================== #

ROLE_CONTINUA = "continua"
ROLE_PERIODICO = "periodico"
ROLE_ALVO = "alvo"
ROLE_IGNORAR = "ignorar"


# --------------------------------------------------------------------------- #
# B.1  Detecção auxiliar (coluna de data/hora e papel provável da aba)
# --------------------------------------------------------------------------- #
def guess_datetime_column(df: pd.DataFrame) -> str | None:
    """Primeira coluna com mais de 80% de valores interpretáveis como data/hora."""
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_datetime64_any_dtype(col):
            return str(c)
        if pd.api.types.is_numeric_dtype(col):
            continue
        sample = col.dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, dayfirst=True, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return str(c)
    return None


def guess_sheet_role(name: str, df: pd.DataFrame) -> str:
    """Sugere o papel de uma aba: alvo, periódico ou contínua (heurística).

    - Nome contendo alvo/produção/rendimento/kpi → alvo;
    - espaçamento mediano ≥ 60 min entre registros → periódico;
    - caso contrário → contínua.
    """
    n = _clean_token(name).lower()
    if any(t in n for t in ("alvo", "produc", "rendiment", "kpi", "meta", "target", "sazonal")):
        return ROLE_ALVO
    dt_col = guess_datetime_column(df)
    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce").dropna()
        if len(dt) >= 2:
            step = estimate_step_minutes(dt)
            if step >= 60.0:
                return ROLE_PERIODICO
    return ROLE_CONTINUA


# --------------------------------------------------------------------------- #
# B.2  Múltiplas abas de variáveis contínuas
# --------------------------------------------------------------------------- #
def build_period_features_multi(
    cont_sheets: list[dict],
    anchors,
    period_min: float,
    *,
    use_turnos: bool = False,
    osc_k: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Agrega N abas de variáveis contínuas nas janelas do alvo e junta tudo.

    ``cont_sheets`` é uma lista de ``{"name": str, "df": DataFrame,
    "dt_col": str, "limits": dict}``. Com mais de uma aba, as colunas são
    prefixadas com o nome da aba (``Moenda_Vazao_mean``) para evitar colisão.

    Retorna ``(period_feat, {aba: passo_em_minutos})``.
    """
    if not cont_sheets:
        raise ValueError("Nenhuma aba de variáveis contínuas informada.")
    prefix = len(cont_sheets) > 1
    frames: list[pd.DataFrame] = []
    steps: dict[str, float] = {}
    for sh in cont_sheets:
        feat, step_min = build_period_features(
            sh["df"], sh["dt_col"], anchors, period_min,
            limits=sh.get("limits") or {}, use_turnos=use_turnos, osc_k=osc_k,
        )
        if prefix:
            tok = _clean_token(sh["name"])
            feat.columns = [f"{tok}_{c}" for c in feat.columns]
        frames.append(feat)
        steps[sh["name"]] = step_min
    out = pd.concat(frames, axis=1)
    return out, steps


# --------------------------------------------------------------------------- #
# B.3  Variáveis PERIÓDICAS (laboratório ou qualquer leitura a cada x horas)
#      alinhadas às janelas do alvo
# --------------------------------------------------------------------------- #
def parse_lab_sheet(
    df: pd.DataFrame, dt_col: str, value_cols: list[str] | None = None
) -> tuple[pd.DataFrame, float]:
    """Lê uma aba de indicadores periódicos como série dos instantes de leitura.

    Retorna ``(lab_vals, lab_period_min)``: ``lab_vals`` indexado pelos horários
    de leitura (duplicatas agrupadas pela média) e ``lab_period_min`` = mediana
    do espaçamento entre leituras. Cada valor informado em T representa a
    média da janela ``(T − lab_period, T]``.
    """
    dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    cols = value_cols or [c for c in df.columns if c != dt_col]
    vals = pd.DataFrame({c: _coerce_numeric_brl(df[c]) for c in cols})
    vals.index = dt
    vals = vals[vals.index.notna()]
    vals = vals.groupby(level=0).mean().sort_index()
    vals = vals.dropna(how="all")
    if len(vals) < 2:
        raise ValueError("São necessárias ao menos 2 leituras do indicador periódico.")
    diffs = np.diff(vals.index.values).astype("timedelta64[s]").astype(float)
    diffs = diffs[diffs > 0]
    lab_period_min = float(np.median(diffs)) / 60.0 if len(diffs) else 240.0
    vals.index.name = "leitura"
    return vals, lab_period_min


def _lab_step_grid(
    lab_vals: pd.DataFrame, lab_period_min: float, grid_step_min: float
) -> pd.DataFrame:
    """Expande as leituras periódicas numa série-degrau em grade fina.

    Cada valor informado em T vale para todo o intervalo ``(T − x, T]``; pontos
    da grade sem leitura dentro dessa tolerância ficam NaN.
    """
    start = lab_vals.index.min() - pd.Timedelta(minutes=lab_period_min)
    end = lab_vals.index.max()
    grid = pd.date_range(start=start, end=end, freq=pd.Timedelta(minutes=grid_step_min))
    return lab_vals.reindex(grid, method="bfill",
                            tolerance=pd.Timedelta(minutes=lab_period_min))


def lab_window_features(
    lab_vals: pd.DataFrame,
    lab_period_min: float,
    anchors,
    period_min: float,
    *,
    limits: dict | None = None,
    lag_shift_min: float = 0.0,
) -> pd.DataFrame:
    """Alinha indicadores periódicos às janelas do alvo (semântica (T−x, T]).

    A série de cada variável é expandida em degrau numa grade fina e agregada na
    janela ``(T_{i-1}, T_i]`` de cada âncora do alvo. Com ``lag_shift_min`` > 0,
    as leituras são deslocadas para FRENTE no tempo — a janela do alvo passa a
    "enxergar" o indicador periódico de ``lag_shift_min`` minutos atrás.

    Por variável: ``{col}_per_mean`` (média ponderada pelo tempo), ``{col}_per_n``
    (nº de leituras na janela) e, com limites ``{col: (lo, hi)}``:
    ``_per_pct_fora``, ``_per_min_abaixo/acima`` (minutos), ``_per_area_abaixo/
    acima`` (desvio × tempo). Sem std/cv/oscilações — seriam artificiais numa
    série-degrau.
    """
    limits = limits or {}
    vals = lab_vals.copy()
    if lag_shift_min:
        vals.index = vals.index + pd.Timedelta(minutes=lag_shift_min)

    grid_step = max(1.0, min(lab_period_min, period_min) / 10.0)
    stepd = _lab_step_grid(vals, lab_period_min, grid_step)
    grid_times = stepd.index.to_numpy(dtype="datetime64[ns]")
    disc_times = vals.index.to_numpy(dtype="datetime64[ns]")

    anchors = pd.DatetimeIndex(pd.to_datetime(anchors)).sort_values()
    period = pd.Timedelta(minutes=period_min)

    records: list[dict] = []
    for i, T in enumerate(anchors):
        start = anchors[i - 1] if i > 0 else (T - period)
        g_lo = int(np.searchsorted(grid_times, np.datetime64(pd.Timestamp(start)), side="right"))
        g_hi = int(np.searchsorted(grid_times, np.datetime64(pd.Timestamp(T)), side="right"))
        d_lo = int(np.searchsorted(disc_times, np.datetime64(pd.Timestamp(start)), side="right"))
        d_hi = int(np.searchsorted(disc_times, np.datetime64(pd.Timestamp(T)), side="right"))
        win = stepd.iloc[g_lo:g_hi]
        disc = vals.iloc[d_lo:d_hi]
        rec: dict = {}
        for c in lab_vals.columns:
            base = _clean_token(c)
            arr = win[c].dropna().to_numpy(dtype=float)
            rec[f"{base}_per_mean"] = float(np.mean(arr)) if arr.size else np.nan
            dvals = disc[c].dropna()
            rec[f"{base}_per_n"] = int(len(dvals))
            if c in limits and arr.size:
                lo, hi = limits[c]
                has_lo = lo is not None and not (isinstance(lo, float) and np.isnan(lo))
                has_hi = hi is not None and not (isinstance(hi, float) and np.isnan(hi))
                out_mask = np.zeros(arr.shape, dtype=bool)
                if has_lo:
                    below = arr < lo
                    out_mask |= below
                    rec[f"{base}_per_min_abaixo"] = float(np.sum(below) * grid_step)
                    rec[f"{base}_per_area_abaixo"] = float(np.sum((lo - arr)[below]) * grid_step)
                if has_hi:
                    above = arr > hi
                    out_mask |= above
                    rec[f"{base}_per_min_acima"] = float(np.sum(above) * grid_step)
                    rec[f"{base}_per_area_acima"] = float(np.sum((arr - hi)[above]) * grid_step)
                rec[f"{base}_per_pct_fora"] = float(np.sum(out_mask) / arr.size * 100.0)
        records.append(rec)

    return pd.DataFrame(records, index=pd.DatetimeIndex(anchors, name="periodo"))


def add_lab_lags(
    lab_vals: pd.DataFrame,
    lab_period_min: float,
    anchors,
    period_min: float,
    *,
    max_lag_min: float,
    limits: dict | None = None,
) -> pd.DataFrame:
    """Cria blocos de lag do indicador periódico em múltiplos do período entre leituras.

    ``k = 0..floor(max_lag_min / lab_period_min)``; cada bloco desloca as
    leituras em ``k·lab_period_min`` minutos. O bloco do período ATUAL mantém o
    nome original (sem sufixo); os defasados recebem ``{col}_per_..._lag_{m}min``
    (m > 0, mesmo padrão reconhecido pelo diagnóstico).
    """
    lab_period_min = lab_period_min if lab_period_min and lab_period_min > 0 else 1.0
    n_lags = int(np.floor(max_lag_min / lab_period_min)) if max_lag_min > 0 else 0
    frames: list[pd.DataFrame] = []
    for k in range(0, n_lags + 1):
        m = int(round(k * lab_period_min))
        feat = lab_window_features(
            lab_vals, lab_period_min, anchors, period_min,
            limits=limits, lag_shift_min=float(m),
        )
        if m != 0:
            feat.columns = [f"{c}_lag_{m}min" for c in feat.columns]
        frames.append(feat)
    return pd.concat(frames, axis=1)


# --------------------------------------------------------------------------- #
# B.4  Excursões de limites críticos
# --------------------------------------------------------------------------- #
def detect_excursion_events(
    df: pd.DataFrame,
    dt_col: str,
    col: str,
    lo: float | None,
    hi: float | None,
    step_min: float,
) -> pd.DataFrame:
    """Eventos contíguos fora de faixa de UM indicador (na série bruta).

    Retorna um DataFrame com ``inicio, fim, duracao_min, lado, pico_desvio,
    area`` (desvio × tempo, em unidade·min) — um evento por sequência contígua
    abaixo do mínimo ou acima do máximo.
    """
    dt = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    vals = _coerce_numeric_brl(df[col])
    ok = dt.notna() & vals.notna()
    t = dt[ok].to_numpy(dtype="datetime64[ns]")
    v = vals[ok].to_numpy(dtype=float)
    order = np.argsort(t)
    t, v = t[order], v[order]

    has_lo = lo is not None and not (isinstance(lo, float) and np.isnan(lo))
    has_hi = hi is not None and not (isinstance(hi, float) and np.isnan(hi))

    events: list[dict] = []

    def _runs(mask: np.ndarray, lado: str, dev: np.ndarray) -> None:
        if not mask.any():
            return
        idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
        for s, e in zip(idx[::2], idx[1::2]):  # [s, e) é um trecho contíguo
            d = dev[s:e]
            events.append({
                "inicio": pd.Timestamp(t[s]),
                "fim": pd.Timestamp(t[e - 1]),
                "duracao_min": float((e - s) * step_min),
                "lado": lado,
                "pico_desvio": float(np.max(d)),
                "area": float(np.sum(d) * step_min),
            })

    if has_lo:
        _runs(v < lo, "abaixo", lo - v)
    if has_hi:
        _runs(v > hi, "acima", v - hi)

    out = pd.DataFrame(events, columns=["inicio", "fim", "duracao_min", "lado",
                                        "pico_desvio", "area"])
    return out.sort_values("inicio").reset_index(drop=True) if not out.empty else out


def excursion_summary(
    events_by_ind: dict[str, pd.DataFrame], total_min_by_ind: dict[str, float]
) -> pd.DataFrame:
    """Resumo de excursões por indicador (nº, tempo fora, duração, severidade)."""
    rows = []
    for ind, ev in events_by_ind.items():
        total_min = float(total_min_by_ind.get(ind, 0.0)) or np.nan
        if ev is None or ev.empty:
            rows.append({
                "Indicador": ind, "Nº eventos": 0, "Tempo fora (min)": 0.0,
                "Tempo fora (%)": 0.0, "Duração média (min)": np.nan,
                "Duração máx (min)": np.nan, "Maior desvio": np.nan,
            })
            continue
        fora = float(ev["duracao_min"].sum())
        rows.append({
            "Indicador": ind,
            "Nº eventos": int(len(ev)),
            "Tempo fora (min)": round(fora, 1),
            "Tempo fora (%)": round(fora / total_min * 100.0, 2) if total_min else np.nan,
            "Duração média (min)": round(float(ev["duracao_min"].mean()), 1),
            "Duração máx (min)": round(float(ev["duracao_min"].max()), 1),
            "Maior desvio": round(float(ev["pico_desvio"].max()), 4),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Tempo fora (%)", ascending=False).reset_index(drop=True)
    return out


def excursion_vs_target(merged: pd.DataFrame, target: str = "Sazonal") -> pd.DataFrame:
    """Associação entre violações de limite e o alvo, por indicador com limite.

    Para cada coluna ``*_pct_fora`` em lag 0 da base de modelagem: correlação de
    Spearman (com p-value) entre a fração fora de faixa e o alvo, e teste de
    Mann-Whitney U comparando o alvo nos períodos COM violação (pct_fora > 0)
    vs. SEM violação. Cobre contínuas (``_pct_fora``) e periódicas
    (``_per_pct_fora``).
    """
    if target not in merged.columns:
        return pd.DataFrame()
    yv = merged[target].astype(float)
    cols = [c for c in merged.columns
            if "_pct_fora" in c and (c.endswith("_lag_0min") or "_lag_" not in c)]
    rows = []
    for c in cols:
        x = merged[c].astype(float)
        mask = x.notna() & yv.notna()
        if mask.sum() < 4:
            continue
        xa, ya = x[mask], yv[mask]
        if _HAS_SCIPY and xa.std() > 0:
            sr, sp = _scipy_stats.spearmanr(xa, ya)
        else:
            sr, sp = float("nan"), float("nan")
        com = ya[xa > 0]
        sem = ya[xa == 0]
        if _HAS_SCIPY and len(com) >= 2 and len(sem) >= 2:
            try:
                _u, mw_p = _scipy_stats.mannwhitneyu(com, sem, alternative="two-sided")
            except ValueError:
                mw_p = float("nan")
        else:
            mw_p = float("nan")
        med_com = float(com.median()) if len(com) else np.nan
        med_sem = float(sem.median()) if len(sem) else np.nan
        rows.append({
            "Variável": re.sub(r"_lag_0min$", "", c),
            "Spearman r": round(float(sr), 4),
            "Spearman p": round(float(sp), 4),
            "Mann-Whitney p": round(float(mw_p), 4),
            "Mediana alvo COM violação": round(med_com, 4) if np.isfinite(med_com) else np.nan,
            "Mediana alvo SEM violação": round(med_sem, 4) if np.isfinite(med_sem) else np.nan,
            "Δ mediana": round(med_com - med_sem, 4)
            if np.isfinite(med_com) and np.isfinite(med_sem) else np.nan,
            "N com violação": int(len(com)),
            "N sem violação": int(len(sem)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Mann-Whitney p").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# B.5  Legenda dos sufixos e nomes legíveis de variáveis
# --------------------------------------------------------------------------- #
# Cada entrada: (rótulo curto p/ frases, categoria, o que mede, como interpretar)
SUFFIX_INFO: dict[str, tuple[str, str, str, str]] = {
    # --- estatísticas básicas da janela --------------------------------- #
    "mean": ("média", "Estatística básica",
             "Valor médio do indicador na janela do período.",
             "Mostra o patamar de operação. Se aparece no topo do ranking, o NÍVEL "
             "do indicador influencia o alvo (suba/desça o setpoint conforme o sentido)."),
    "median": ("valor típico (mediana)", "Estatística básica",
               "Valor central da janela (metade das leituras acima, metade abaixo).",
               "Como a média, porém pouco afetado por picos isolados."),
    "min": ("valor mínimo", "Estatística básica",
            "Menor leitura registrada na janela.",
            "Captura quedas pontuais; relevante quando mergulhos momentâneos prejudicam o processo."),
    "max": ("valor máximo", "Estatística básica",
            "Maior leitura registrada na janela.",
            "Captura picos pontuais; relevante quando estouros momentâneos prejudicam o processo."),
    "range": ("variação total", "Instabilidade",
              "Diferença entre o maior e o menor valor da janela.",
              "Alta = janela com grande variação; associação com o alvo indica "
              "que oscilações amplas impactam o resultado."),
    "std": ("instabilidade", "Instabilidade",
            "O quanto as leituras variam em torno da média (desvio padrão).",
            "Associação (em geral negativa) com o alvo indica que operar instável "
            "prejudica o resultado — estabilize o controle."),
    "cv": ("instabilidade relativa", "Instabilidade",
           "Instabilidade proporcional à média (comparável entre escalas diferentes).",
           "Como a instabilidade, mas comparável entre indicadores de grandezas diferentes."),
    "p10": ("valores baixos (P10)", "Estatística básica",
            "Patamar abaixo do qual ficam 10% das leituras da janela.",
            "Representa os 'vales típicos' do período (mínimos sustentados, sem o ruído "
            "de um único pico). Útil quando operar baixo demais prejudica."),
    "p25": ("valores médio-baixos (P25)", "Estatística básica",
            "Patamar abaixo do qual ficam 25% das leituras da janela.",
            "Parte baixa da operação típica do período."),
    "p75": ("valores médio-altos (P75)", "Estatística básica",
            "Patamar abaixo do qual ficam 75% das leituras da janela.",
            "Parte alta da operação típica do período."),
    "p90": ("valores altos (P90)", "Estatística básica",
            "Patamar abaixo do qual ficam 90% das leituras da janela.",
            "Representa os 'picos típicos' do período (máximos sustentados)."),
    "sum": ("total", "Estatística básica",
            "Soma de todas as leituras da janela.",
            "Em vazões/contagens equivale ao volume/total do período; segue a média quando "
            "o nº de leituras é constante."),
    "mean_abs_diff": ("oscilação", "Instabilidade",
                      "O quanto o sinal varia de uma leitura para a outra, em média.",
                      "Alto = sinal ruidoso ou com muitas manobras; o vai-e-vem do controle "
                      "pode impactar o resultado."),
    "max_rate": ("variação brusca", "Instabilidade",
                 "Maior salto entre duas leituras consecutivas na janela.",
                 "Captura degraus/transientes bruscos (partidas, paradas, manobras)."),
    "n_oscilacoes": ("saltos bruscos", "Instabilidade",
                     "Quantidade de variações muito acima do normal do período.",
                     "Conta eventos anormais; alto = controle oscilando ou distúrbios repetidos."),
    "n_valid": ("nº de leituras válidas", "Qualidade do dado",
                "Quantidade de leituras não vazias na janela.",
                "Baixo = falha de medição/comunicação no período; os demais atributos ficam menos confiáveis."),
    "pct_missing": ("% de leituras ausentes", "Qualidade do dado",
                    "Percentual de leituras esperadas que não chegaram na janela.",
                    "Alto = sensor/coletor com falha; avalie antes de confiar no período."),
    # --- limites críticos ------------------------------------------------ #
    "min_abaixo": ("tempo abaixo do limite", "Limites críticos",
                   "Tempo (min) que o indicador ficou ABAIXO do limite mínimo definido.",
                   "Quanto maior, mais tempo fora da condição segura; associação com o alvo "
                   "quantifica o prejuízo de operar abaixo do limite."),
    "area_abaixo": ("gravidade abaixo do limite", "Limites críticos",
                    "Junta numa só medida DUAS coisas: o quanto o valor passou do "
                    "limite mínimo e por quanto tempo ficou assim.",
                    "Separa uma violação leve de uma grave. Ex.: ficar só 0,1 abaixo "
                    "do limite por 10 min conta pouco; ficar 2,0 abaixo por 10 min "
                    "conta muito mais. Quanto maior o número, pior a violação."),
    "min_acima": ("tempo acima do limite", "Limites críticos",
                  "Tempo (min) que o indicador ficou ACIMA do limite máximo definido.",
                  "Quanto maior, mais tempo fora da condição segura, no lado de cima."),
    "area_acima": ("gravidade acima do limite", "Limites críticos",
                   "Junta numa só medida o quanto o valor passou do limite máximo e "
                   "por quanto tempo ficou assim.",
                   "Separa um estouro leve de um grave. Quanto maior o número, pior a "
                   "violação pelo lado de cima."),
    "pct_fora": ("% do tempo fora da faixa", "Limites críticos",
                 "Percentual do período em que o indicador violou os limites (abaixo ou acima).",
                 "Resumo direto da disciplina operacional; compare períodos com e sem violação "
                 "na aba 🚦 (teste de Mann-Whitney)."),
    # --- indicadores periódicos (laboratório ou leituras a cada x horas) - #
    "per_mean": ("média", "Indicadores periódicos",
                 "Média (ponderada pelo tempo) das leituras do indicador periódico vigentes "
                 "na janela — cada valor vale para a janela (T−x, T].",
                 "Nível médio do indicador periódico durante o período do alvo."),
    "per_n": ("nº de leituras", "Indicadores periódicos",
              "Quantas leituras do indicador periódico caíram na janela.",
              "Baixo = período com pouca cobertura (resultado menos confiável)."),
    "per_pct_fora": ("% do tempo fora da faixa", "Periódico + Limites",
                     "Percentual do tempo em que o indicador periódico esteve fora "
                     "dos limites (abaixo do mínimo ou acima do máximo).",
                     "Resumo simples da disciplina do indicador periódico; compare "
                     "com o alvo na aba 🚦."),
    "per_min_abaixo": ("tempo abaixo do limite", "Periódico + Limites",
                       "Quanto tempo (min) o indicador periódico ficou ABAIXO do "
                       "limite mínimo definido.",
                       "Quanto maior, mais tempo operando abaixo do especificado."),
    "per_min_acima": ("tempo acima do limite", "Periódico + Limites",
                      "Quanto tempo (min) o indicador periódico ficou ACIMA do "
                      "limite máximo definido.",
                      "Quanto maior, mais tempo operando acima do especificado."),
    "per_area_abaixo": ("gravidade abaixo do limite", "Periódico + Limites",
                        "Junta numa só medida o quanto o valor passou do limite "
                        "mínimo e por quanto tempo ficou assim.",
                        "Separa uma violação leve de uma grave: passar só um pouco "
                        "do limite por pouco tempo conta pouco; passar muito e por "
                        "muito tempo conta muito mais. Quanto maior, pior."),
    "per_area_acima": ("gravidade acima do limite", "Periódico + Limites",
                       "Junta numa só medida o quanto o valor passou do limite "
                       "máximo e por quanto tempo ficou assim.",
                       "Separa um estouro leve de um grave. Quanto maior, pior."),
    # --- turnos ----------------------------------------------------------- #
    "mean_t1": ("média no turno 00–08", "Turnos",
                "Média do indicador apenas no turno 00h–08h.",
                "Permite isolar diferenças entre turnos (equipe, carga, matéria-prima)."),
    "mean_t2": ("média no turno 08–16", "Turnos",
                "Média do indicador apenas no turno 08h–16h.",
                "Permite isolar diferenças entre turnos."),
    "mean_t3": ("média no turno 16–24", "Turnos",
                "Média do indicador apenas no turno 16h–24h.",
                "Permite isolar diferenças entre turnos."),
    "std_t1": ("instabilidade no turno 00–08", "Turnos",
               "Instabilidade do indicador apenas no turno 00h–08h.",
               "Instabilidade por turno — aponta o turno que mais oscila."),
    "std_t2": ("instabilidade no turno 08–16", "Turnos",
               "Instabilidade do indicador apenas no turno 08h–16h.",
               "Instabilidade por turno."),
    "std_t3": ("instabilidade no turno 16–24", "Turnos",
               "Instabilidade do indicador apenas no turno 16h–24h.",
               "Instabilidade por turno."),
}

# sufixos ordenados do mais longo para o mais curto (match correto de
# "mean_abs_diff" antes de "mean", "per_mean" antes de "mean", etc.)
_SUFFIXES_BY_LEN = sorted(SUFFIX_INFO, key=len, reverse=True)


def suffix_legend_table() -> pd.DataFrame:
    """Tabela-legenda dos sufixos para exibição na UI e no relatório."""
    rows = [
        {"Categoria": cat, "Sufixo": f"_{suf}", "Nome": rotulo.capitalize(),
         "O que mede": mede, "Como interpretar / impacto": impacto}
        for suf, (rotulo, cat, mede, impacto) in SUFFIX_INFO.items()
    ]
    order = ["Estatística básica", "Instabilidade", "Limites críticos",
             "Indicadores periódicos", "Periódico + Limites", "Turnos",
             "Qualidade do dado"]
    df = pd.DataFrame(rows)
    df["__o__"] = df["Categoria"].map({c: i for i, c in enumerate(order)}).fillna(99)
    return df.sort_values(["__o__", "Sufixo"]).drop(columns="__o__").reset_index(drop=True)


def humanize_variable(name: str, sheet_names: list[str] | None = None) -> str:
    """Converte um nome técnico em frase legível.

    Ex.: ``Moenda_Vazao_p10`` → ``10º percentil de Vazao (Moenda)``;
    ``Brix_per_mean_lag_240min`` → ``média de Brix, com lag de 240 min (4 h antes)``.
    Variáveis sem defasagem (período atual) NÃO recebem texto de lag.
    ``sheet_names`` (nomes das abas usadas como prefixo) permite separar a aba do
    indicador; sem ela, o prefixo permanece junto ao nome do indicador.
    """
    base = str(name)

    # 1) defasagem — só menciona quando há lag (m > 0); período atual fica sem texto
    lag_txt = ""
    m = re.search(r"_lag_(\d+)min$", base)
    if m:
        mins = int(m.group(1))
        base = base[: m.start()]
        if mins == 0:
            lag_txt = ""
        elif mins % 60 == 0:
            h = mins // 60
            lag_txt = f", com lag de {mins} min ({h:g} h antes)"
        else:
            lag_txt = f", com lag de {mins} min"
    else:
        d = re.search(r"_lag_(\d+)d$", base)
        if d:
            dias = int(d.group(1))
            base = base[: d.start()]
            lag_txt = "" if dias == 0 else (
                f", com lag de {dias} dia" + ("s" if dias > 1 else ""))

    # 2) sufixo estatístico
    rotulo = None
    for suf in _SUFFIXES_BY_LEN:
        if base.endswith(f"_{suf}"):
            rotulo = SUFFIX_INFO[suf][0]
            base = base[: -(len(suf) + 1)]
            break

    # 3) prefixo de aba (quando os nomes das abas são conhecidos)
    aba_txt = ""
    if sheet_names:
        for sh in sorted(sheet_names, key=len, reverse=True):
            tok = _clean_token(sh)
            if base.startswith(f"{tok}_"):
                base = base[len(tok) + 1:]
                aba_txt = f" ({sh})"
                break

    indicador = base.replace("_", " ").strip() or name
    if rotulo:
        return f"{rotulo} de {indicador}{aba_txt}{lag_txt}"
    return f"{indicador}{aba_txt}{lag_txt}"


# --------------------------------------------------------------------------- #
# B.6  Ranking por INDICADOR (sem sufixo/lag/métrica) e investigação em cadeia
# --------------------------------------------------------------------------- #
def base_indicator(name: str, sheet_names: list[str] | None = None) -> tuple[str, str]:
    """Reduz um nome de feature ao seu indicador-base.

    Retorna ``(key, display)``: ``key`` agrupa as métricas/defasagens do mesmo
    indicador (mantém o prefixo da aba, ex.: ``Moenda_Vazao``); ``display`` é o
    nome limpo para exibição (ex.: ``Vazao (Moenda)``), sem sufixo estatístico,
    sem lag e sem o tipo de métrica.
    """
    base = str(name)
    # remove a defasagem (período atual não tem sufixo)
    m = re.search(r"_lag_(\d+)min$", base)
    if m:
        base = base[: m.start()]
    else:
        d = re.search(r"_lag_(\d+)d$", base)
        if d:
            base = base[: d.start()]
    # remove o sufixo estatístico (mais longo primeiro)
    for suf in _SUFFIXES_BY_LEN:
        if base.endswith(f"_{suf}"):
            base = base[: -(len(suf) + 1)]
            break
    key = base
    # separa o prefixo de aba para o display
    aba_txt = ""
    if sheet_names:
        for sh in sorted(sheet_names, key=len, reverse=True):
            tok = _clean_token(sh)
            if base.startswith(f"{tok}_"):
                base = base[len(tok) + 1:]
                aba_txt = f" ({sh})"
                break
    indicador = base.replace("_", " ").strip() or str(name)
    return key, f"{indicador}{aba_txt}"


def indicator_ranking(
    consolidated: pd.DataFrame, sheet_names: list[str] | None = None
) -> pd.DataFrame:
    """Agrega o ranking consolidado (por feature) ao nível de INDICADOR.

    Cada indicador recebe como ``Impacto`` o **máximo** do "Score consolidado"
    entre as suas métricas/defasagens (a evidência mais forte). Retorna
    ``["Indicador", "Impacto", "Nº métricas"]`` ordenado do maior impacto ao menor.
    """
    cols = ["Indicador", "Impacto", "Nº métricas"]
    if (consolidated is None or consolidated.empty
            or "Score consolidado" not in consolidated.columns):
        return pd.DataFrame(columns=cols)
    df = consolidated.copy()
    pairs = [base_indicator(v, sheet_names) for v in df["Variável"]]
    df["__key"] = [k for k, _ in pairs]
    df["__disp"] = [d for _, d in pairs]
    agg = df.groupby("__key", sort=False).agg(
        _disp=("__disp", "first"),
        _imp=("Score consolidado", "max"),
        _n=("__disp", "size"),
    )
    out = pd.DataFrame({
        "Indicador": agg["_disp"].values,
        "Impacto": agg["_imp"].round(4).values,
        "Nº métricas": agg["_n"].astype(int).values,
    })
    return out.sort_values("Impacto", ascending=False).reset_index(drop=True)


def drilldown_ranking(
    X: pd.DataFrame, target_col: str, sheet_names: list[str] | None = None,
    *, model_type: str | None = None, run_shap: bool = False,
) -> pd.DataFrame:
    """Refaz a análise usando ``target_col`` (o indicador raiz) como novo alvo.

    Trata ``target_col`` (tipicamente a média/nível do indicador escolhido) como
    o novo indicador a explicar, remove todas as colunas do próprio indicador
    (métricas e defasagens — evita autoexplicação) e ranqueia quais OUTROS
    indicadores mais impactam nele. Reaproveita ``statistical_ranking`` e, se
    ``model_type`` for informado (mesma profundidade do run principal),
    ``train_model``/``shap_importance``. Retorna o ranking por indicador.
    """
    empty = pd.DataFrame(columns=["Indicador", "Impacto", "Nº métricas"])
    if target_col not in X.columns:
        return empty
    y2 = X[target_col].astype(float)
    tgt_key, _ = base_indicator(target_col, sheet_names)
    drop_cols = [c for c in X.columns
                 if base_indicator(c, sheet_names)[0] == tgt_key]
    X2 = X.drop(columns=drop_cols)
    if X2.shape[1] == 0:
        return empty

    stat2 = statistical_ranking(X2, y2)
    importances = shap_imp = None
    if model_type:
        try:
            mr = train_model(X2, y2, model_type=model_type)
            importances = mr.importances
            if run_shap:
                shap_imp = shap_importance(mr, X2)
        except Exception:  # noqa: BLE001 - drill-down degrada para só estatística
            importances = shap_imp = None
    cons2 = consolidated_ranking(stat2, importances, shap_imp)
    return indicator_ranking(cons2, sheet_names)


def indicator_shapley(
    X: pd.DataFrame, y: pd.Series, sheet_names: list[str] | None = None,
    *, stat_df: pd.DataFrame | None = None, max_indicators: int = 16,
) -> tuple[pd.DataFrame, float]:
    """Decomposição de Shapley/LMG da variância explicada (R²) por INDICADOR.

    É o "SHAP estatístico": reparte de forma **justa e não-sobreposta** (valores
    de Shapley, da teoria dos jogos) quanto cada indicador explica da variação do
    alvo — **sem treinar modelo de ML**, então funciona inclusive no modo "Só
    estatística". Cada indicador é representado pela sua feature de relação linear
    mais forte; as fatias somam o R² total explicado.

    Retorna ``(tabela, r2_total)`` com colunas
    ``["Indicador", "Contribuição (%)", "Parcela (R²)"]`` (ordenado desc).
    """
    cols_out = ["Indicador", "Contribuição (%)", "Parcela (R²)"]
    if not _HAS_SKLEARN or X is None or X.shape[1] == 0 or len(y) < 5:
        return pd.DataFrame(columns=cols_out), 0.0

    # força linear (|Pearson|) de cada feature, para escolher o representante
    if stat_df is not None and not stat_df.empty and "Pearson r" in stat_df.columns:
        strength = {r["Variável"]: abs(float(r["Pearson r"]))
                    for _, r in stat_df.iterrows()}
    else:
        strength = {}
        yv = y.to_numpy(dtype=float)
        for c in X.columns:
            xv = X[c].to_numpy(dtype=float)
            strength[c] = (abs(float(np.corrcoef(xv, yv)[0, 1]))
                           if np.std(xv) > 0 else 0.0)

    # agrupa por indicador e escolhe a feature mais forte de cada
    reps: dict[str, tuple[str, float]] = {}
    disp: dict[str, str] = {}
    for c in X.columns:
        key, d = base_indicator(c, sheet_names)
        s = strength.get(c, 0.0)
        if key not in reps or s > reps[key][1]:
            reps[key] = (c, s)
            disp[key] = d

    # com muitos indicadores, mantém os mais fortes (Shapley fica caro)
    items = sorted(reps.items(), key=lambda kv: kv[1][1], reverse=True)[:max_indicators]
    keys = [k for k, _ in items]
    if not keys:
        return pd.DataFrame(columns=cols_out), 0.0

    Xr = pd.DataFrame({k: X[reps[k][0]].to_numpy(dtype=float) for k in keys})
    shares, r2 = relative_importance(Xr, y)

    rows = []
    for k in keys:
        sh = max(0.0, float(shares.get(k, 0.0)))
        rows.append({
            "Indicador": disp[k],
            "Contribuição (%)": round(sh / r2 * 100, 1) if r2 > 1e-9 else 0.0,
            "Parcela (R²)": round(sh, 4),
        })
    out = (pd.DataFrame(rows, columns=cols_out)
           .sort_values("Parcela (R²)", ascending=False).reset_index(drop=True))
    return out, round(float(r2), 4)


# --------------------------------------------------------------------------- #
# B.7  Texto do relatório final (por que estas variáveis são as principais)
# --------------------------------------------------------------------------- #
def _fmt_lag_legivel(name: str) -> str | None:
    m = re.search(r"_lag_(\d+)min", name)
    if not m:
        return None
    mins = int(m.group(1))
    if mins == 0:
        return "no mesmo período"
    if mins % 60 == 0:
        h = mins // 60
        return f"{h} h antes" if h > 1 else "1 h antes"
    return f"{mins} min antes"


def generate_report_text(
    consolidated: pd.DataFrame,
    model_result: ModelResult | None,
    stat_df: pd.DataFrame,
    *,
    excursion_tbl: pd.DataFrame | None = None,
    period_min: float | None = None,
    n_periodos: int | None = None,
    top_n: int = 5,
    sheet_names: list[str] | None = None,
) -> str:
    """Texto (markdown) do relatório final: por que cada variável entrou no topo.

    Escrito em linguagem clara para o usuário final: os nomes técnicos das
    variáveis são convertidos em frases legíveis (ex.: "instabilidade da
    temperatura, sem defasagem") e a evidência estatística é descrita em
    palavras (associação forte/moderada, direta/inversa, significativa).
    """
    if consolidated is None or consolidated.empty:
        return "Não há evidência estatística suficiente para um relatório."

    stat_idx = (stat_df.set_index("Variável")
                if stat_df is not None and not stat_df.empty else pd.DataFrame())

    def _forca(v: float) -> str:
        v = abs(v)
        return "forte" if v >= 0.5 else "moderada" if v >= 0.3 else "leve"

    linhas: list[str] = ["## Principais variáveis associadas ao indicador-alvo", ""]
    ctx = []
    if n_periodos:
        ctx.append(f"**{n_periodos} períodos** analisados")
    if period_min:
        ctx.append(f"cada período cobre **{period_min:g} min** (~{period_min / 60:g} h)")
    if ctx:
        linhas += ["Base da análise: " + "; ".join(ctx) + ".", ""]

    top = consolidated.head(top_n)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        var = str(row["Variável"])
        partes: list[str] = []
        sr = stat_idx.loc[var].get("Spearman r") if var in stat_idx.index else None
        sp = stat_idx.loc[var].get("Spearman p") if var in stat_idx.index else None
        if pd.notna(sr):
            sentido = ("direta — sobe junto com o alvo" if sr > 0
                       else "inversa — sobe quando o alvo cai")
            sig = (", estatisticamente significativa"
                   if pd.notna(sp) and sp < 0.05 else "")
            partes.append(f"associação {_forca(sr)} e {sentido}{sig}")
        elif ("SHAP" in consolidated.columns and pd.notna(row.get("SHAP"))) or (
                "Importância modelo" in consolidated.columns
                and pd.notna(row.get("Importância modelo"))):
            partes.append("peso relevante no modelo preditivo")
        tema = _theme_of(var)
        if tema:
            partes.append(f"relacionada a **{tema}**")
        lag_txt = _fmt_lag_legivel(var)
        if lag_txt and lag_txt != "no mesmo período":
            partes.append(f"**efeito defasado** — o comportamento de {lag_txt} "
                          "influencia o período atual")

        hum = humanize_variable(var, sheet_names)
        score = row.get("Score consolidado")
        linha = f"**{rank}. {hum[:1].upper() + hum[1:]}**"
        if pd.notna(score):
            linha += f" — relevância {score:.2f}"
        frase = "; ".join(partes) if partes else \
            "evidência combinada das análises e do modelo"
        linha += f". {frase[:1].upper() + frase[1:]}."
        linhas.append(linha)
    linhas.append("")

    if excursion_tbl is not None and not excursion_tbl.empty:
        sig = excursion_tbl[excursion_tbl["Mann-Whitney p"] < 0.05]
        if not sig.empty:
            linhas.append("## Limites críticos com impacto detectado")
            linhas.append("")
            for _, r in sig.iterrows():
                hum = humanize_variable(r["Variável"], sheet_names)
                delta = r.get("Δ mediana")
                if pd.notna(delta):
                    direc = "menor" if delta < 0 else "maior"
                    txt = (f"períodos com violação tiveram o indicador-alvo "
                           f"significativamente **{direc}** (em média, {delta:+.2f})")
                else:
                    txt = ("períodos com violação tiveram o indicador-alvo "
                           "significativamente diferente")
                linhas.append(f"- **{hum[:1].upper() + hum[1:]}**: {txt}.")
            linhas.append("")

    if model_result is not None and not np.isnan(model_result.metrics.get("R2", np.nan)):
        r2 = max(0.0, model_result.metrics["R2"])
        linhas.append("## Desempenho do modelo")
        linhas.append("")
        linhas.append(
            f"O modelo conseguiu explicar cerca de **{r2 * 100:.0f}%** da variação "
            f"do indicador-alvo nos períodos de teste "
            f"(erro médio de {model_result.metrics['MAE']:.2f})."
        )
        linhas.append("")

    linhas.append(
        "*Observação: as variáveis acima são as de maior **evidência estatística** "
        "de influência — prováveis causas a investigar, não causalidade comprovada.*"
    )
    return "\n".join(linhas)
