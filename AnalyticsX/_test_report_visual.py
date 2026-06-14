"""Gera um PDF de inspeção visual com gráficos de nomes longos (como no app real).

Uso: python _test_report_visual.py  -> _test_report_visual.pdf
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import report

rng = np.random.default_rng(3)

# ranking de barras horizontais com nomes longos reais
nomes = [
    "Moenda_Temperatura_mean_abs_diff_lag_0min", "Fermentacao_pH_area_abaixo_lag_0min",
    "Moenda_Vazao_mean_lag_0min", "Brix_per_last_lag_240min",
    "Fermentacao_pH_pct_fora_lag_0min", "Moenda_Temperatura_std_lag_480min",
    "Moenda_Pressao_max_rate_lag_960min", "Fermentacao_Nivel_p90_lag_0min",
    "Pol_per_mean_lag_240min", "Moenda_Vazao_n_oscilacoes_lag_480min",
]
vals = np.sort(rng.uniform(0.1, 0.95, len(nomes)))
fig_rank = go.Figure(go.Bar(
    x=vals, y=nomes, orientation="h",
    text=[f"{v:.3f}" for v in vals], textposition="auto",
    marker_color=vals, marker_colorscale="Purples",
))
fig_rank.update_layout(title="Score consolidado", xaxis_title="Score",
                       yaxis_title="Variável", height=max(320, 26 * len(nomes)))

# série temporal real x previsto
idx = pd.date_range("2026-05-01", periods=120, freq="8h")
y = 85 + np.cumsum(rng.normal(0, 0.4, 120))
fig_serie = go.Figure()
fig_serie.add_trace(go.Scatter(x=idx, y=y, mode="lines+markers", name="Rendimento (real)"))
fig_serie.add_trace(go.Scatter(x=idx, y=y + rng.normal(0, 0.6, 120), mode="lines",
                               name="Previsto (treino)", line=dict(dash="dot")))
fig_serie.update_layout(title="Indicador alvo: real × previsto",
                        xaxis_title="Período", yaxis_title="Rendimento", height=420,
                        legend=dict(orientation="h", y=1.1))

# matriz de correlação
cols = nomes[:8]
corr = np.round(np.clip(rng.uniform(-1, 1, (8, 8)), -1, 1), 2)
fig_corr = go.Figure(go.Heatmap(
    z=corr, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
    text=corr, texttemplate="%{text:.2f}", textfont=dict(size=9),
))
fig_corr.update_layout(title="Matriz de correlação", height=max(400, 32 * 8))

items = [
    report.ReportItem(id="r", kind="figure", title="Ranking (nomes longos)", payload=fig_rank),
    report.ReportItem(id="s", kind="figure", title="Série real × previsto", payload=fig_serie),
    report.ReportItem(id="c", kind="figure", title="Matriz de correlação", payload=fig_corr),
]
meta = {"titulo": "Inspeção visual do PDF", "autor": "teste"}
pdf = report.build_pdf(items, meta, "Verificação de legibilidade dos gráficos.")
with open("_test_report_visual.pdf", "wb") as f:
    f.write(pdf)
print(f"PDF gerado ({len(pdf)//1024} KB) -> _test_report_visual.pdf")
