# -*- coding: utf-8 -*-
"""
keep_awake.py — Mantém o app do Streamlit Community Cloud acordado.

Abre a URL do app num navegador headless (Chromium). Se a página de "soneca"
do Streamlit aparecer, clica no botão "Yes, get this app back up!" para acordar
o app. É executado pelo GitHub Actions em intervalos regulares — a visita
periódica de um navegador real evita que o app durma e o reativa caso já tenha
dormido.

A URL vem da variável de ambiente APP_URL (configurada no workflow a partir da
variável de repositório STREAMLIT_APP_URL).
"""

import os
import re
import sys

from playwright.sync_api import sync_playwright

URL = (os.environ.get("APP_URL") or "").strip()

# Texto do botão que reativa o app na tela de soneca do Streamlit.
WAKE_PATTERN = re.compile(r"get this app back up|wake|acordar", re.I)
# Sinais de que o app está dormindo / indisponível.
ASLEEP_PATTERN = re.compile(
    r"get this app back up|gone to sleep|app is asleep|is sleeping", re.I
)


def main() -> int:
    if not URL:
        print("::error::Variável STREAMLIT_APP_URL não definida. Configure em "
              "Settings > Secrets and variables > Actions > Variables (aba "
              "'Variables'), com o nome STREAMLIT_APP_URL e o valor da URL do app.")
        return 1

    print(f"Visitando: {URL}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()

        try:
            # 'domcontentloaded' (e não 'networkidle') porque o app aberto mantém
            # um websocket ativo — networkidle nunca dispararia.
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        except Exception as exc:
            print(f"::warning::Falha ao abrir a página: {exc}")

        page.wait_for_timeout(5000)

        body = page.content() or ""
        if ASLEEP_PATTERN.search(body):
            print("App parece estar DORMINDO — tentando acordar...")
            clicked = _click_wake(page)
            if clicked:
                print("Botão de acordar clicado. Aguardando o app subir...")
                page.wait_for_timeout(45000)
            else:
                print("::warning::Não encontrei o botão de acordar (layout pode ter mudado).")
        else:
            print("App já está ACORDADO.")

        # Screenshot para depuração (enviado como artefato pelo workflow).
        try:
            page.screenshot(path="status.png")
        except Exception:
            pass

        asleep = bool(ASLEEP_PATTERN.search(page.content() or ""))
        print("Estado final:", "DORMINDO" if asleep else "ACORDADO")
        browser.close()

    # Não falha o job só por continuar dormindo — evita ruído de notificações.
    return 0


def _click_wake(page) -> bool:
    """Tenta clicar no botão que reativa o app, por várias estratégias."""
    # 1) Botão pelo papel/nome.
    try:
        btn = page.get_by_role("button", name=WAKE_PATTERN)
        if btn.count() > 0:
            btn.first.click(timeout=10000)
            return True
    except Exception:
        pass
    # 2) Qualquer elemento clicável que contenha o texto.
    try:
        el = page.get_by_text(WAKE_PATTERN)
        if el.count() > 0:
            el.first.click(timeout=10000)
            return True
    except Exception:
        pass
    return False


if __name__ == "__main__":
    sys.exit(main())
