"""Smoke test do report.py: HTML interativo, HTML estático (kaleido) e PDF.

Uso: python _test_report.py
"""

import pandas as pd
import plotly.graph_objects as go

import report

fig = go.Figure(go.Bar(x=[1, 2, 3], y=["Vazao", "Temperatura", "Brix"],
                       orientation="h"))
fig.update_layout(title="Ranking de exemplo")

tbl = pd.DataFrame({"Variável": ["Vazao_mean", "Temp_std"],
                    "Spearman r": [0.51, -0.52], "p": [0.0, 0.0]})

items = [
    report.ReportItem(id="fig1", kind="figure", title="Ranking de exemplo",
                      payload=fig, comment="Comentário do usuário."),
    report.ReportItem(id="tbl1", kind="table", title="Tabela de exemplo", payload=tbl),
    report.ReportItem(id="met1", kind="metrics", title="Métricas",
                      payload={"MAE": "1.16", "R²": "0.577"}),
    report.ReportItem(id="txt1", kind="text", title="Observações",
                      payload="## Título\n- item **negrito**\n- outro item"),
]
meta = {"titulo": "Relatório de Teste", "autor": "Smoke Test",
        "observacoes": "Gerado automaticamente."}
diag = "## Análise\n**Variável X** apresentou correlação de Spearman +0.51."

html_i = report.build_html(items, meta, diag, interactive=True)
assert "plotly" in html_i.lower() and "Ranking de exemplo" in html_i
with open("_test_report.html", "w", encoding="utf-8") as f:
    f.write(html_i)
print(f"HTML interativo OK ({len(html_i)//1024} KB) -> _test_report.html")

png = report._fig_to_png_b64(fig)
assert len(png) > 1000
print(f"kaleido OK (PNG base64 de {len(png)//1024} KB)")

pdf = report.build_pdf(items, meta, diag)
assert pdf[:4] == b"%PDF"
with open("_test_report.pdf", "wb") as f:
    f.write(pdf)
print(f"PDF OK ({len(pdf)//1024} KB) -> _test_report.pdf")
print("TUDO OK")
