"""
Guia Interativo de Data Wrangling — App Streamlit
-------------------------------------------------
Renderiza o guia interativo (originalmente um arquivo HTML autocontido) dentro
do Streamlit, preservando todos os recursos interativos: menu com barra de
progresso (checkboxes salvos no navegador), abas, quizzes, botões de "copiar"
nos blocos de código, realce de sintaxe e destaque da seção atual (scroll-spy).

Rodar localmente:
    streamlit run guia_data_wrangling.py
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Caminho do HTML relativo a este arquivo (funciona em qualquer diretório de
# trabalho, inclusive no Streamlit Community Cloud).
HTML_PATH = Path(__file__).parent / "guia_data_wrangling.html"

st.set_page_config(
    page_title="Guia Interativo de Data Wrangling",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Remove o espaçamento padrão do Streamlit para o guia ocupar toda a largura.
st.markdown(
    """
    <style>
      .block-container {padding: 0 !important; max-width: 100% !important;}
      header[data-testid="stHeader"] {display: none;}
      #MainMenu, footer {visibility: hidden;}
      div[data-testid="stAppViewContainer"] > .main {padding: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def carregar_guia() -> str:
    """Lê o HTML do guia. Cacheado para não reler o arquivo a cada interação."""
    return HTML_PATH.read_text(encoding="utf-8")


if not HTML_PATH.exists():
    st.error(
        f"Arquivo do guia não encontrado: `{HTML_PATH.name}`. "
        "Certifique-se de que ele está na mesma pasta deste script."
    )
else:
    # scrolling=True permite rolar todo o conteúdo dentro do componente,
    # mantendo o menu lateral fixo (position: sticky) como no HTML original.
    components.html(carregar_guia(), height=1000, scrolling=True)
