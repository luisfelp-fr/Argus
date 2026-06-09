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

import itertools
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
