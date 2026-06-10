"""
plotting.py — Helpers de gráficos Plotly do Argus Analytics
===========================================================

Todas as funções retornam um objeto ``plotly.graph_objects.Figure`` já
estilizado com o tema visual do Argus (paleta azul-petróleo + dourado, fundo
claro). Nenhuma função chama Streamlit diretamente — assim podem ser
reaproveitadas no relatório HTML/PDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Paleta Argus -------------------------------------------------------------- #
ARGUS_PRIMARY = "#0E4D64"      # azul-petróleo
ARGUS_SECONDARY = "#1B98B5"    # azul claro
ARGUS_ACCENT = "#E8A33D"       # dourado (referência ao "olho de Argus")
ARGUS_DANGER = "#D7263D"       # vermelho (alertas/outliers)
ARGUS_OK = "#2A9D8F"           # verde (positivo)
ARGUS_GRID = "#E6ECEF"
ARGUS_SEQ = px.colors.sequential.Teal


def _apply_theme(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    """Aplica layout padrão do Argus a uma figura."""
    fig.update_layout(
        title=dict(text=title or "", font=dict(size=18, color=ARGUS_PRIMARY)),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=60 if title else 30, b=50),
        font=dict(family="Segoe UI, Arial, sans-serif", color="#243B43"),
        colorway=[ARGUS_PRIMARY, ARGUS_ACCENT, ARGUS_SECONDARY, ARGUS_OK, ARGUS_DANGER],
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor=ARGUS_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=ARGUS_GRID, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Distribuição
# --------------------------------------------------------------------------- #
def histogram(series: pd.Series, nbins: int = 30, title: str | None = None) -> go.Figure:
    fig = go.Figure(
        go.Histogram(x=series.dropna(), nbinsx=nbins, marker_color=ARGUS_SECONDARY,
                     marker_line_color=ARGUS_PRIMARY, marker_line_width=1, opacity=0.85)
    )
    fig.update_layout(bargap=0.03)
    fig.update_xaxes(title=series.name)
    fig.update_yaxes(title="Frequência")
    return _apply_theme(fig, title or f"Histograma — {series.name}")


def boxplot(series: pd.Series, title: str | None = None) -> go.Figure:
    fig = go.Figure(
        go.Box(y=series.dropna(), name=str(series.name), boxpoints="outliers",
               marker_color=ARGUS_PRIMARY, line_color=ARGUS_PRIMARY,
               marker=dict(color=ARGUS_DANGER, size=5))
    )
    fig.update_yaxes(title=series.name)
    return _apply_theme(fig, title or f"Boxplot — {series.name}")


def density(series: pd.Series, title: str | None = None) -> go.Figure:
    """Curva de densidade (KDE) via scipy, com histograma normalizado ao fundo."""
    data = series.dropna().to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=data, histnorm="probability density", nbinsx=30,
                     marker_color=ARGUS_GRID, name="Histograma", opacity=0.7)
    )
    try:
        from scipy.stats import gaussian_kde

        if len(data) > 2 and np.std(data) > 0:
            kde = gaussian_kde(data)
            xs = np.linspace(data.min(), data.max(), 200)
            fig.add_trace(
                go.Scatter(x=xs, y=kde(xs), mode="lines", name="Densidade (KDE)",
                           line=dict(color=ARGUS_ACCENT, width=3))
            )
    except Exception:
        pass
    fig.update_xaxes(title=series.name)
    fig.update_yaxes(title="Densidade")
    return _apply_theme(fig, title or f"Densidade — {series.name}")


def histogram_with_normal(series: pd.Series, title: str | None = None) -> go.Figure:
    """Histograma com curva normal teórica sobreposta (para normalidade)."""
    data = series.dropna().to_numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=data, histnorm="probability density", nbinsx=30,
                     marker_color=ARGUS_SECONDARY, opacity=0.7, name="Dados")
    )
    if len(data) > 2 and np.std(data) > 0:
        mu, sigma = float(np.mean(data)), float(np.std(data, ddof=1))
        xs = np.linspace(data.min(), data.max(), 200)
        ys = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((xs - mu) ** 2) / (2 * sigma ** 2))
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name="Normal teórica",
                       line=dict(color=ARGUS_DANGER, width=3))
        )
    fig.update_xaxes(title=series.name)
    fig.update_yaxes(title="Densidade")
    return _apply_theme(fig, title or f"Distribuição vs. Normal — {series.name}")


def qq_plot(series: pd.Series, title: str | None = None) -> go.Figure:
    """Gráfico Q-Q (quantis amostrais vs. teóricos da normal)."""
    from scipy import stats

    data = series.dropna().to_numpy()
    (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Quantis",
                             marker=dict(color=ARGUS_PRIMARY, size=6)))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines",
                             name="Referência normal", line=dict(color=ARGUS_ACCENT, width=2)))
    fig.update_xaxes(title="Quantis teóricos")
    fig.update_yaxes(title="Quantis amostrais")
    return _apply_theme(fig, title or f"Gráfico Q-Q — {series.name}")


# --------------------------------------------------------------------------- #
# Correlação
# --------------------------------------------------------------------------- #
def corr_heatmap(corr: pd.DataFrame, title: str | None = None) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(height=max(400, 60 * len(corr)))
    return _apply_theme(fig, title or "Matriz de Correlação")


def scatter(x: pd.Series, y: pd.Series, trendline: bool = True,
            title: str | None = None) -> go.Figure:
    pair = pd.concat([x, y], axis=1).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pair.iloc[:, 0], y=pair.iloc[:, 1], mode="markers",
                             marker=dict(color=ARGUS_PRIMARY, size=7, opacity=0.7),
                             name="Observações"))
    if trendline and len(pair) > 2:
        xv = pair.iloc[:, 0].to_numpy()
        yv = pair.iloc[:, 1].to_numpy()
        if np.std(xv) > 0:
            coef = np.polyfit(xv, yv, 1)
            xs = np.array([xv.min(), xv.max()])
            fig.add_trace(go.Scatter(x=xs, y=np.polyval(coef, xs), mode="lines",
                                     name="Tendência", line=dict(color=ARGUS_ACCENT, width=3)))
    fig.update_xaxes(title=x.name)
    fig.update_yaxes(title=y.name)
    return _apply_theme(fig, title or f"Dispersão — {x.name} × {y.name}")


def ranking_bar(labels: list[str], values: list[float], title: str | None = None,
                xlabel: str = "Valor") -> go.Figure:
    colors = [ARGUS_OK if v >= 0 else ARGUS_DANGER for v in values]
    fig = go.Figure(
        go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
               text=[f"{v:.3f}" for v in values], textposition="auto")
    )
    fig.update_layout(height=max(320, 36 * len(labels)))
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(autorange="reversed")
    return _apply_theme(fig, title or "Ranking")


# --------------------------------------------------------------------------- #
# Regressão
# --------------------------------------------------------------------------- #
def real_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color=ARGUS_PRIMARY, size=7, opacity=0.7),
                             name="Observações"))
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             name="Ajuste perfeito", line=dict(color=ARGUS_ACCENT, dash="dash")))
    fig.update_xaxes(title="Valor real")
    fig.update_yaxes(title="Valor previsto")
    return _apply_theme(fig, title or "Real vs. Previsto")


def residuals_plot(y_pred: np.ndarray, residuals: np.ndarray, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color=ARGUS_PRIMARY, size=7, opacity=0.7), name="Resíduos"))
    fig.add_hline(y=0, line=dict(color=ARGUS_DANGER, dash="dash"))
    fig.update_xaxes(title="Valor previsto")
    fig.update_yaxes(title="Resíduo")
    return _apply_theme(fig, title or "Gráfico de Resíduos")


# --------------------------------------------------------------------------- #
# Comparação entre grupos
# --------------------------------------------------------------------------- #
def grouped_box(df: pd.DataFrame, value_col: str, group_col: str,
                title: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (grp, sub) in enumerate(df.groupby(group_col, observed=True)):
        fig.add_trace(go.Box(y=sub[value_col].dropna(), name=str(grp), boxpoints="outliers"))
    fig.update_yaxes(title=value_col)
    fig.update_xaxes(title=group_col)
    return _apply_theme(fig, title or f"{value_col} por {group_col}")


# --------------------------------------------------------------------------- #
# Controle estatístico de processo (CEP)
# --------------------------------------------------------------------------- #
def control_chart(values: pd.Series, center: float, ucl: float, lcl: float,
                  x: pd.Series | None = None, title: str | None = None) -> go.Figure:
    xs = x if x is not None else pd.Series(range(len(values)))
    vals = values.to_numpy()
    out_mask = (vals > ucl) | (vals < lcl)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=vals, mode="lines+markers", name="Observações",
                             line=dict(color=ARGUS_PRIMARY),
                             marker=dict(color=np.where(out_mask, ARGUS_DANGER, ARGUS_PRIMARY), size=7)))
    fig.add_hline(y=center, line=dict(color=ARGUS_OK, width=2),
                  annotation_text="Média (LC)", annotation_position="right")
    fig.add_hline(y=ucl, line=dict(color=ARGUS_DANGER, dash="dash"),
                  annotation_text="LSC", annotation_position="right")
    fig.add_hline(y=lcl, line=dict(color=ARGUS_DANGER, dash="dash"),
                  annotation_text="LIC", annotation_position="right")
    fig.update_yaxes(title=values.name)
    fig.update_xaxes(title="Observação / Tempo")
    return _apply_theme(fig, title or f"Carta de Controle — {values.name}")


def capability_hist(series: pd.Series, lie: float | None, lse: float | None,
                    title: str | None = None) -> go.Figure:
    fig = histogram(series, title=title or f"Capabilidade — {series.name}")
    if lie is not None:
        fig.add_vline(x=lie, line=dict(color=ARGUS_DANGER, width=2, dash="dash"),
                      annotation_text="LIE", annotation_position="top")
    if lse is not None:
        fig.add_vline(x=lse, line=dict(color=ARGUS_DANGER, width=2, dash="dash"),
                      annotation_text="LSE", annotation_position="top")
    fig.add_vline(x=float(series.mean()), line=dict(color=ARGUS_OK, width=2),
                  annotation_text="Média", annotation_position="bottom")
    return fig


# --------------------------------------------------------------------------- #
# Séries temporais
# --------------------------------------------------------------------------- #
def time_trend(time: pd.Series, value: pd.Series, rolling: pd.Series | None = None,
               title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=value, mode="lines", name="Valor",
                             line=dict(color=ARGUS_SECONDARY)))
    if rolling is not None:
        fig.add_trace(go.Scatter(x=time, y=rolling, mode="lines", name="Média móvel",
                                 line=dict(color=ARGUS_ACCENT, width=3)))
    fig.update_xaxes(title="Tempo")
    fig.update_yaxes(title=value.name)
    return _apply_theme(fig, title or f"Tendência Temporal — {value.name}")


def lag_curve(lags_min: np.ndarray, corrs: np.ndarray, best_lag: float | None = None,
              title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags_min, y=corrs, mode="lines+markers",
                             line=dict(color=ARGUS_PRIMARY), name="Correlação"))
    fig.add_hline(y=0, line=dict(color="#999", dash="dot"))
    if best_lag is not None:
        fig.add_vline(x=best_lag, line=dict(color=ARGUS_ACCENT, width=2),
                      annotation_text=f"Melhor lag = {best_lag:g}", annotation_position="top")
    fig.update_xaxes(title="Lag (min)")
    fig.update_yaxes(title="Correlação")
    return _apply_theme(fig, title or "Correlação por Lag")


def outlier_scatter(series: pd.Series, mask: np.ndarray, title: str | None = None) -> go.Figure:
    idx = np.arange(len(series))
    colors = np.where(mask, ARGUS_DANGER, ARGUS_PRIMARY)
    fig = go.Figure(
        go.Scatter(x=idx, y=series.to_numpy(), mode="markers",
                   marker=dict(color=colors, size=7), name="Valores")
    )
    fig.update_xaxes(title="Índice")
    fig.update_yaxes(title=series.name)
    return _apply_theme(fig, title or f"Outliers — {series.name}")
