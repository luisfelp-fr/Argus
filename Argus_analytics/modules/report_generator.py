"""
Módulo 14 — Relatório Final
===========================

Lista as análises adicionadas, permite reordenar/remover e gera um relatório
profissional em HTML (autocontido, com gráficos Plotly interativos). Opcionalmente
gera PDF, se ``kaleido`` e ``reportlab`` estiverem disponíveis (degradação graciosa).
"""

from __future__ import annotations

import base64
import io

import pandas as pd
import streamlit as st

from utils.helpers import now_str

# Jinja2 (template do HTML)
try:
    from jinja2 import Template
    _HAS_JINJA = True
except Exception:  # pragma: no cover
    _HAS_JINJA = False

# kaleido (exportar figuras para imagem) — usado apenas no PDF
try:
    import kaleido  # noqa: F401
    import plotly.io as pio  # noqa: F401
    _HAS_KALEIDO = True
except Exception:  # pragma: no cover
    _HAS_KALEIDO = False

# reportlab (PDF)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table as RLTable,
        TableStyle, PageBreak,
    )
    _HAS_REPORTLAB = True
except Exception:  # pragma: no cover
    _HAS_REPORTLAB = False


# --------------------------------------------------------------------------- #
# Template HTML
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatório Argus Analytics</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; color:#243B43; margin:0; padding:0; }
  .cover { background:linear-gradient(135deg,#0E4D64,#1B98B5); color:#fff; padding:60px 50px; }
  .cover h1 { font-size:3rem; margin:0; }
  .cover p { font-size:1.2rem; opacity:.95; }
  .content { padding:30px 50px; max-width:1100px; margin:auto; }
  h2 { color:#0E4D64; border-bottom:3px solid #E8A33D; padding-bottom:6px; margin-top:40px; }
  h3 { color:#1B98B5; }
  .meta { background:#E6ECEF; padding:14px 18px; border-radius:8px; font-size:.95rem; }
  .interp { background:#FFF8EC; border-left:4px solid #E8A33D; padding:12px 16px; margin:14px 0; }
  table.data { border-collapse:collapse; width:100%; margin:14px 0; font-size:.9rem; }
  table.data th { background:#0E4D64; color:#fff; padding:8px; text-align:left; }
  table.data td { border:1px solid #E6ECEF; padding:6px 8px; }
  table.data tr:nth-child(even){ background:#F5F8F9; }
  .footer { text-align:center; color:#789; font-size:.85rem; padding:30px; }
  .badge { display:inline-block; background:#0E4D64; color:#fff; padding:3px 10px;
           border-radius:12px; font-size:.8rem; margin-right:6px; }
</style>
</head>
<body>
  <div class="cover">
    <h1>👁️ Argus Analytics</h1>
    <p>Relatório de Análise Estatística</p>
  </div>
  <div class="content">
    <div class="meta">
      <strong>Arquivo analisado:</strong> {{ file_name }}<br>
      <strong>Data do relatório:</strong> {{ report_date }}<br>
      <strong>Resumo da base:</strong> {{ n_rows }} linhas × {{ n_cols }} colunas
      ({{ n_numeric }} numéricas, {{ n_categorical }} categóricas, {{ n_datetime }} data/hora)<br>
      <strong>Análises incluídas:</strong> {{ items|length }}
    </div>

    {% for item in items %}
    <h2>{{ loop.index }}. {{ item.name }}</h2>
    <p>
      {% for k, v in item.variables.items() %}<span class="badge">{{ k }}: {{ v }}</span>{% endfor %}
      {% for k, v in item.params.items() %}<span class="badge">{{ k }}: {{ v }}</span>{% endfor %}
    </p>
    <p style="color:#789;font-size:.85rem;">🕒 {{ item.timestamp }}</p>

    {% for fig_html in item.figures_html %}
      <div>{{ fig_html|safe }}</div>
    {% endfor %}

    {% for tname, thtml in item.tables_html.items() %}
      <h3>{{ tname }}</h3>
      {{ thtml|safe }}
    {% endfor %}

    {% if item.interpretation %}
      <div class="interp"><strong>Interpretação:</strong><br>{{ item.interpretation|replace('\n','<br>')|safe }}</div>
    {% endif %}
    {% endfor %}

    <h2>Conclusões principais</h2>
    <ul>
      {% for c in conclusions %}<li>{{ c }}</li>{% endfor %}
    </ul>

    <h2>Recomendações práticas</h2>
    <ul>
      {% for r in recommendations %}<li>{{ r }}</li>{% endfor %}
    </ul>
  </div>
  <div class="footer">
    Gerado por Argus Analytics — {{ report_date }}
  </div>
</body>
</html>
"""


def _fig_to_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _table_to_html(df: pd.DataFrame) -> str:
    return df.to_html(classes="data", border=0, index=True, justify="left")


def _build_conclusions(items: list[dict]) -> list[str]:
    if not items:
        return ["Nenhuma análise foi adicionada ao relatório."]
    out = [f"Foram realizadas {len(items)} análises sobre a base de dados."]
    names = ", ".join(sorted({i["name"] for i in items}))
    out.append(f"Análises incluídas: {names}.")
    # Primeira linha de interpretação de cada análise como destaque
    for i in items:
        first = (i["interpretation"] or "").split("\n")[0].strip()
        if first:
            out.append(f"**{i['name']}**: {first}")
    return out


def _build_recommendations(items: list[dict]) -> list[str]:
    recs = [
        "Valide a qualidade dos dados (ausentes, duplicados, outliers) antes de decisões.",
        "Lembre-se de que correlação não implica causalidade.",
        "Acompanhe os indicadores ao longo do tempo para confirmar tendências.",
    ]
    names = {i["name"] for i in items}
    if any("Normalidade" in n for n in names):
        recs.append("Escolha testes paramétricos ou não paramétricos conforme o resultado de normalidade.")
    if any("Controle" in n for n in names):
        recs.append("Investigue causas especiais nos pontos fora dos limites de controle.")
    if any("Capabilidade" in n for n in names):
        recs.append("Atue na centralização e/ou redução de variação para melhorar o Cpk.")
    return recs


def build_html(items: list[dict], meta: dict) -> str:
    if not _HAS_JINJA:
        return "<html><body><h1>Jinja2 não disponível.</h1></body></html>"

    render_items = []
    for it in items:
        render_items.append({
            "name": it["name"],
            "variables": it.get("variables", {}),
            "params": it.get("params", {}),
            "timestamp": it.get("timestamp", ""),
            "interpretation": it.get("interpretation", ""),
            "figures_html": [_fig_to_html(f) for f in it.get("figures", [])],
            "tables_html": {k: _table_to_html(v) for k, v in it.get("tables", {}).items()},
        })

    template = Template(HTML_TEMPLATE)
    # Plotly.js incluído uma única vez no <head> (via CDN); figuras usam include_plotlyjs=False
    return template.render(
        items=render_items,
        conclusions=_build_conclusions(items),
        recommendations=_build_recommendations(items),
        **meta,
    )


def build_pdf(items: list[dict], meta: dict) -> bytes | None:
    """Gera o PDF via reportlab + kaleido. Retorna None se faltar dependência."""
    if not (_HAS_REPORTLAB and _HAS_KALEIDO):
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ArgusTitle", parent=styles["Title"],
                                 textColor=colors.HexColor("#0E4D64"))
    h2 = ParagraphStyle("ArgusH2", parent=styles["Heading2"],
                        textColor=colors.HexColor("#0E4D64"))
    story = []

    story.append(Paragraph("Argus Analytics", title_style))
    story.append(Paragraph("Relatório de Análise Estatística", styles["Heading3"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f"<b>Arquivo:</b> {meta['file_name']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Data:</b> {meta['report_date']}", styles["Normal"]))
    story.append(Paragraph(
        f"<b>Base:</b> {meta['n_rows']} linhas × {meta['n_cols']} colunas", styles["Normal"]))
    story.append(PageBreak())

    for idx, it in enumerate(items, 1):
        story.append(Paragraph(f"{idx}. {it['name']}", h2))
        if it.get("interpretation"):
            story.append(Paragraph(it["interpretation"].replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        for fig in it.get("figures", []):
            try:
                img_bytes = fig.to_image(format="png", width=900, height=450, scale=1)
                story.append(RLImage(io.BytesIO(img_bytes), width=16 * cm, height=8 * cm))
                story.append(Spacer(1, 0.3 * cm))
            except Exception:
                continue
        for tname, tdf in it.get("tables", {}).items():
            story.append(Paragraph(tname, styles["Heading4"]))
            data = [list(map(str, [""] + list(tdf.columns)))]
            for ridx, row in tdf.head(20).iterrows():
                data.append([str(ridx)] + [str(v) for v in row.values])
            tbl = RLTable(data)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0E4D64")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3 * cm))
        story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _meta(state) -> dict:
    df = state.get("df")
    col_types = state.get("col_types")
    return {
        "file_name": state.get("file_name") or "—",
        "report_date": now_str(),
        "n_rows": len(df) if df is not None else 0,
        "n_cols": df.shape[1] if df is not None else 0,
        "n_numeric": len(col_types.numeric) if col_types else 0,
        "n_categorical": len(col_types.categorical) if col_types else 0,
        "n_datetime": len(col_types.datetime) if col_types else 0,
    }


def render(state) -> None:
    items = st.session_state.get("report_items", [])

    st.info(
        "Aqui você monta o relatório final. Adicione análises pelos botões "
        "**➕ Adicionar ao relatório** em cada tela, organize a ordem e baixe em HTML/PDF."
    )

    if not items:
        st.warning("📭 Nenhuma análise adicionada ainda. Volte às análises e clique em "
                   "**➕ Adicionar ao relatório**.")
        return

    st.markdown(f"### 📑 {len(items)} análise(s) no relatório")

    # Lista com reordenação/remoção
    for i, item in enumerate(items):
        with st.container(border=True):
            cols = st.columns([0.06, 0.06, 0.06, 0.82])
            if cols[0].button("⬆️", key=f"up_{i}", disabled=(i == 0)):
                items[i - 1], items[i] = items[i], items[i - 1]
                st.rerun()
            if cols[1].button("⬇️", key=f"down_{i}", disabled=(i == len(items) - 1)):
                items[i + 1], items[i] = items[i], items[i + 1]
                st.rerun()
            if cols[2].button("🗑️", key=f"del_{i}"):
                items.pop(i)
                st.rerun()
            with cols[3]:
                st.markdown(f"**{i+1}. {item['name']}**  ·  🕒 {item.get('timestamp','')}")
                vs = ", ".join(f"{k}={v}" for k, v in item.get("variables", {}).items())
                if vs:
                    st.caption(vs)

    st.divider()
    if st.button("🧹 Limpar relatório", key="clear_report"):
        st.session_state["report_items"] = []
        st.rerun()

    # Geração
    st.markdown("### ⬇️ Baixar relatório")
    meta = _meta(state)

    col1, col2 = st.columns(2)
    with col1:
        try:
            html = build_html(items, meta)
            st.download_button(
                "📄 Baixar relatório HTML",
                data=html.encode("utf-8"),
                file_name="relatorio_argus.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Falha ao gerar HTML: {exc}")

    with col2:
        if _HAS_REPORTLAB and _HAS_KALEIDO:
            if st.button("🧾 Gerar PDF", use_container_width=True, key="gen_pdf"):
                with st.spinner("Gerando PDF (pode levar alguns segundos)..."):
                    pdf = build_pdf(items, meta)
                if pdf:
                    st.download_button(
                        "📥 Baixar PDF", data=pdf, file_name="relatorio_argus.pdf",
                        mime="application/pdf", use_container_width=True, key="dl_pdf",
                    )
                else:
                    st.warning("Não foi possível gerar o PDF.")
        else:
            st.info("📄 Exportação em PDF indisponível neste ambiente "
                    "(requer `reportlab` + `kaleido`). O relatório HTML é interativo "
                    "e abre em qualquer navegador.")

    with st.expander("👁️ Pré-visualizar relatório (HTML)"):
        try:
            st.components.v1.html(build_html(items, meta), height=600, scrolling=True)
        except Exception:
            st.caption("Pré-visualização indisponível — baixe o HTML para visualizar.")
