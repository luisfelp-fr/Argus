"""
Relatório final — montagem, prévia HTML e exportação PDF
--------------------------------------------------------
O usuário "anexa" gráficos/tabelas/textos das abas de análise (botão 📎). Os itens
ficam em ``st.session_state[REPORT_KEY]`` e são montados num HTML único:

- **Prévia / download HTML**: gráficos Plotly interativos (plotly.js via CDN).
- **Download PDF**: mesmo template em variante "pdf-safe" — gráficos convertidos
  em PNG (kaleido) e HTML renderizado por xhtml2pdf (puro Python, funciona no
  Streamlit Cloud).

Sem dependência de Streamlit: as funções recebem/retornam dados puros, e a UI
(daily_ui.py) cuida do session_state.
"""

from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import html as _html
import io
import re
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go

try:
    from xhtml2pdf import pisa
    _HAS_PDF = True
except Exception:  # noqa: BLE001
    _HAS_PDF = False

REPORT_KEY = "report_items"
META_KEY = "report_meta"


# --------------------------------------------------------------------------- #
# Estrutura dos itens
# --------------------------------------------------------------------------- #
@dataclass
class ReportItem:
    id: str                    # chave única (dedupe)
    kind: str                  # "figure" | "table" | "text" | "metrics"
    title: str
    payload: object            # go.Figure | pd.DataFrame | str | dict[str, str]
    comment: str = ""
    source_tab: str = field(default="")


def get_items(state) -> list[ReportItem]:
    return state.setdefault(REPORT_KEY, [])


def has_item(state, item_id: str) -> bool:
    return any(it.id == item_id for it in get_items(state))


def add_item(state, item: ReportItem) -> bool:
    """Adiciona se ainda não existe; retorna True se adicionou."""
    items = get_items(state)
    if any(it.id == item.id for it in items):
        return False
    items.append(item)
    return True


def remove_item(state, item_id: str) -> None:
    state[REPORT_KEY] = [it for it in get_items(state) if it.id != item_id]


def move_item(state, idx: int, delta: int) -> None:
    items = get_items(state)
    j = idx + delta
    if 0 <= idx < len(items) and 0 <= j < len(items):
        items[idx], items[j] = items[j], items[idx]


def items_fingerprint(items: list[ReportItem], meta: dict, diagnosis: str) -> str:
    """Hash p/ invalidar o cache do PDF quando itens/comentários/textos mudarem."""
    h = hashlib.sha1()
    for it in items:
        h.update(it.id.encode()); h.update(it.comment.encode())
    for k in sorted(meta):
        h.update(str(meta[k]).encode())
    h.update(diagnosis.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Conversões
# --------------------------------------------------------------------------- #
def _fig_to_png_b64(fig: go.Figure, *, scale: float = 2.0) -> str:
    png = fig.to_image(format="png", scale=scale)  # requer kaleido
    return base64.b64encode(png).decode()


def _md_to_html(text: str) -> str:
    """Conversor mínimo de markdown (títulos, **negrito**, *itálico*, listas)."""
    out: list[str] = []
    in_list = False
    for raw in text.splitlines():
        line = _html.escape(raw.rstrip())
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", line)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if in_list:
                out.append("</ul>"); in_list = False
            lvl = min(len(m.group(1)) + 2, 6)   # h3..h6 dentro do relatório
            out.append(f"<h{lvl}>{m.group(2)}</h{lvl}>")
            continue
        if re.match(r"^\s*[-•]\s+", line):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{re.sub(r'^\s*[-•]\s+', '', line)}</li>")
            continue
        if in_list:
            out.append("</ul>"); in_list = False
        if line.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _table_html(df: pd.DataFrame, max_rows: int = 60) -> str:
    shown = df.head(max_rows)
    note = ""
    if len(df) > max_rows:
        note = f"<p class='note'>Mostrando {max_rows} de {len(df)} linhas.</p>"
    return shown.to_html(index=False, classes="rep-table", border=0,
                         float_format=lambda v: f"{v:.4g}") + note


def _metrics_html(metrics: dict) -> str:
    cells = "".join(
        f"<td class='metric'><div class='metric-label'>{_html.escape(str(k))}</div>"
        f"<div class='metric-value'>{_html.escape(str(v))}</div></td>"
        for k, v in metrics.items()
    )
    return f"<table class='metrics-row'><tr>{cells}</tr></table>"


# --------------------------------------------------------------------------- #
# Template HTML
# --------------------------------------------------------------------------- #
_CSS_BASE = """
body { font-family: Helvetica, Arial, sans-serif; color: #222; margin: 24px; }
h1 { font-size: 22px; color: #14406b; margin-bottom: 2px; }
h2 { font-size: 17px; color: #14406b; border-bottom: 2px solid #14406b;
     padding-bottom: 3px; margin-top: 28px; }
h3, h4, h5, h6 { color: #1a5a93; margin-bottom: 4px; }
p { font-size: 12px; line-height: 1.5; }
ul { font-size: 12px; line-height: 1.5; }
.subtitle { color: #666; font-size: 12px; margin-top: 0; }
.comment { background: #f4f8fc; border-left: 3px solid #1a5a93; padding: 6px 10px;
           font-size: 11px; color: #333; margin: 6px 0; }
.note { color: #888; font-size: 10px; }
.rep-table { border-collapse: collapse; font-size: 10px; width: 100%; }
.rep-table th { background: #14406b; color: #fff; padding: 4px 6px; text-align: left; }
.rep-table td { border-bottom: 1px solid #ddd; padding: 3px 6px; }
.metrics-row { width: 100%; border-collapse: separate; border-spacing: 8px 0; }
.metric { background: #f4f8fc; border: 1px solid #cfe0f0; border-radius: 6px;
          padding: 8px; text-align: center; }
.metric-label { font-size: 10px; color: #555; }
.metric-value { font-size: 16px; font-weight: bold; color: #14406b; }
.source-tag { color: #999; font-size: 10px; }
"""

# xhtml2pdf não aceita largura percentual em <img>; o HTML interativo aceita.
_CSS_FIG_HTML = ".fig-img { width: 100%; }"
_CSS_FIG_PDF = ".fig-img { width: 480pt; }"


def _render_item(it: ReportItem, *, interactive: bool) -> str:
    parts = [f"<h2>{_html.escape(it.title)}</h2>"]
    if it.source_tab:
        parts.append(f"<p class='source-tag'>Origem: {_html.escape(it.source_tab)}</p>")
    if it.comment.strip():
        parts.append(f"<div class='comment'>💬 {_html.escape(it.comment)}</div>")

    if it.kind == "figure" and isinstance(it.payload, go.Figure):
        if interactive:
            parts.append(it.payload.to_html(full_html=False, include_plotlyjs=False,
                                            default_height=420))
        else:
            b64 = _fig_to_png_b64(it.payload)
            parts.append(f"<img class='fig-img' src='data:image/png;base64,{b64}'/>")
    elif it.kind == "table" and isinstance(it.payload, pd.DataFrame):
        parts.append(_table_html(it.payload))
    elif it.kind == "metrics" and isinstance(it.payload, dict):
        parts.append(_metrics_html(it.payload))
    else:  # "text" ou payload inesperado
        parts.append(_md_to_html(str(it.payload)))
    return "\n".join(parts)


def build_html(items: list[ReportItem], meta: dict, diagnosis_md: str,
               *, interactive: bool) -> str:
    """Monta o relatório completo. ``interactive=True`` usa Plotly.js (CDN)."""
    title = meta.get("titulo") or "Relatório de Análise de Causas"
    author = meta.get("autor") or ""
    obs = meta.get("observacoes") or ""
    date_str = _dt.datetime.now().strftime("%d/%m/%Y %H:%M")

    head_extra = (
        "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js' charset='utf-8'></script>"
        if interactive else ""
    )
    body: list[str] = [
        f"<h1>{_html.escape(title)}</h1>",
        f"<p class='subtitle'>Gerado em {date_str}"
        + (f" · {_html.escape(author)}" if author else "") + "</p>",
    ]
    if obs.strip():
        body.append(f"<div class='comment'>{_html.escape(obs)}</div>")
    if diagnosis_md.strip():
        body.append("<h2>🧭 Por que estas variáveis foram consideradas as principais</h2>")
        body.append(_md_to_html(diagnosis_md))
    for it in items:
        body.append(_render_item(it, interactive=interactive))
    if not items:
        body.append("<p class='note'>Nenhum item anexado — use os botões "
                    "📎 nas abas de análise.</p>")

    css = _CSS_BASE + (_CSS_FIG_HTML if interactive else _CSS_FIG_PDF)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_html.escape(title)}</title>{head_extra}"
        f"<style>{css}</style></head><body>"
        + "\n".join(body) + "</body></html>"
    )


def build_pdf(items: list[ReportItem], meta: dict, diagnosis_md: str) -> bytes:
    """Gera o PDF a partir da variante estática (figuras em PNG via kaleido)."""
    if not _HAS_PDF:
        raise RuntimeError("xhtml2pdf não está instalado — adicione ao requirements.txt.")
    html_doc = build_html(items, meta, diagnosis_md, interactive=False)
    buf = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html_doc), dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError("Falha na conversão do HTML para PDF (xhtml2pdf).")
    return buf.getvalue()
