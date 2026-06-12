"""
Identidade visual do Argus AnalyticsX (olho verde no cabeçalho e no login).
Gera o HTML dos cabeçalhos reutilizando o mesmo ícone do app.
"""

from __future__ import annotations

import base64
import os

_ICON = os.path.join(os.path.dirname(__file__), "assets", "eye_icon.png")


def eye_icon_data_uri() -> str:
    """Retorna o olho verde como data URI base64 (ou "" se a imagem faltar)."""
    try:
        with open(_ICON, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


def app_header_html(title: str, subtitle: str = "") -> str:
    """Cabeçalho do app: olho verde + título (cor adapta ao tema)."""
    uri = eye_icon_data_uri()
    img = (f"<img src='{uri}' width='50' height='50' "
           "style='display:block;flex:none' alt='Argus'/>"
           if uri else "<span style='font-size:2.2rem'>👁️</span>")
    sub = (f"<div style='font-size:0.98rem;color:#7A8794;margin-top:1px'>"
           f"{subtitle}</div>" if subtitle else "")
    return (
        "<div style='display:flex;align-items:center;gap:14px;"
        "margin:0.1rem 0 0.7rem'>"
        f"{img}"
        "<div>"
        "<div style='font-size:1.95rem;font-weight:700;line-height:1.1'>"
        f"{title}</div>{sub}</div>"
        "</div>"
    )


def login_header_html() -> str:
    """Cabeçalho centralizado da tela de login."""
    uri = eye_icon_data_uri()
    img = (f"<img src='{uri}' width='72' height='72' "
           "style='display:block;margin:0 auto' alt='Argus'/>"
           if uri else "<div style='font-size:3rem'>👁️</div>")
    return (
        "<div style='text-align:center'>"
        f"{img}"
        "<h1 style='margin:.45rem 0 0;color:#0E2A47'>"
        "Argus AnalyticsX</h1>"
        "<p style='color:#5C7185;margin-top:4px'>Análise Avançada de Processo</p>"
        "</div>"
    )
