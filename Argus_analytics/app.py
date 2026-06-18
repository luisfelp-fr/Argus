"""
Argus Analytics — Analisador estatístico guiado (Streamlit)
===========================================================

Aplicativo principal: configura a página, gerencia a navegação por
``session_state`` e roteia para cada módulo de análise.

Execução:
    streamlit run app.py

O nome "Argus" faz referência ao gigante mitológico de muitos olhos: a ideia é
"enxergar" os dados por vários ângulos.
"""

from __future__ import annotations

import streamlit as st

from utils.helpers import init_session, has_data
from modules import (
    data_loader,
    data_quality,
    descriptive_stats,
    distribution,
    normality_tests,
    correlation,
    regression,
    group_comparison,
    control_charts,
    capability,
    time_series,
    lag_analysis,
    outliers,
    report_generator,
)

# --------------------------------------------------------------------------- #
# Configuração da página
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Argus Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()

# Catálogo de análises: chave -> (rótulo, ícone, descrição, função render, precisa_dados)
ANALYSES: dict[str, dict] = {
    "data_overview": {
        "label": "Visão Geral dos Dados",
        "icon": "📋",
        "desc": "Importe sua planilha, veja a prévia e o diagnóstico automático dos tipos de coluna.",
        "render": data_loader.render,
        "needs_data": False,
    },
    "outliers": {
        "label": "Outliers",
        "icon": "🚨",
        "desc": "Identifique valores muito diferentes do padrão (IQR, Z-score e MAD).",
        "render": outliers.render,
        "needs_data": True,
    },
    "data_quality": {
        "label": "Qualidade dos Dados",
        "icon": "🧹",
        "desc": "Verifique valores ausentes, duplicidades, colunas constantes e problemas de formatação.",
        "render": data_quality.render,
        "needs_data": True,
    },
    "descriptive": {
        "label": "Estatística Descritiva",
        "icon": "📊",
        "desc": "Resuma o comportamento das variáveis: média, mediana, dispersão e limites.",
        "render": descriptive_stats.render,
        "needs_data": True,
    },
    "distribution": {
        "label": "Distribuição dos Dados",
        "icon": "📈",
        "desc": "Histograma, boxplot e densidade para ver concentração, assimetria e outliers.",
        "render": distribution.render,
        "needs_data": True,
    },
    "normality": {
        "label": "Teste de Normalidade",
        "icon": "🔔",
        "desc": "Descubra se uma variável segue distribuição normal (Shapiro-Wilk e Anderson-Darling).",
        "render": normality_tests.render,
        "needs_data": True,
    },
    "correlation": {
        "label": "Correlação",
        "icon": "🔗",
        "desc": "Verifique se duas ou mais variáveis se movimentam juntas. Útil para relações entre indicadores.",
        "render": correlation.render,
        "needs_data": True,
    },
    "regression": {
        "label": "Regressão",
        "icon": "📐",
        "desc": "Entenda como uma ou mais variáveis influenciam uma variável resposta.",
        "render": regression.render,
        "needs_data": True,
    },
    "group_comparison": {
        "label": "Comparação entre Grupos",
        "icon": "⚖️",
        "desc": "Compare se diferentes grupos (turnos, máquinas, fornecedores) se comportam de forma distinta.",
        "render": group_comparison.render,
        "needs_data": True,
    },
    "control_charts": {
        "label": "Controle Estatístico de Processo",
        "icon": "🚦",
        "desc": "Verifique se o processo está estável ao longo do tempo (carta de controle).",
        "render": control_charts.render,
        "needs_data": True,
    },
    "capability": {
        "label": "Capabilidade Cp / Cpk",
        "icon": "🎯",
        "desc": "Avalie se a variação do processo cabe nos limites de especificação e se está centralizado.",
        "render": capability.render,
        "needs_data": True,
    },
    "time_series": {
        "label": "Análise Temporal",
        "icon": "⏱️",
        "desc": "Observe como uma variável evolui no tempo: tendências, oscilações e médias móveis.",
        "render": time_series.render,
        "needs_data": True,
    },
    "lag_analysis": {
        "label": "Análise com Lag",
        "icon": "⏳",
        "desc": "Descubra se uma variável afeta outra após um atraso (defasagem temporal).",
        "render": lag_analysis.render,
        "needs_data": True,
    },
    "report": {
        "label": "Relatório Final",
        "icon": "📑",
        "desc": "Monte, organize e baixe o relatório com todas as análises adicionadas.",
        "render": report_generator.render,
        "needs_data": False,
    },
}


# --------------------------------------------------------------------------- #
# Estilo (CSS leve)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .argus-title { color:#0E4D64; font-weight:800; }
      .argus-sub   { color:#1B98B5; }
      div[data-testid="stMetricValue"] { color:#0E4D64; }
      .stButton>button { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar — navegação
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 👁️ Argus Analytics")
        st.caption("Seu assistente estatístico guiado")
        st.divider()

        if st.button("🏠 Início", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

        st.markdown("**Análises**")
        for key, meta in ANALYSES.items():
            disabled = meta["needs_data"] and not has_data()
            if st.button(
                f"{meta['icon']} {meta['label']}",
                key=f"nav_{key}",
                use_container_width=True,
                disabled=disabled,
            ):
                st.session_state["page"] = key
                st.rerun()

        st.divider()
        if has_data():
            n_items = len(st.session_state.get("report_items", []))
            st.success(f"✅ Dados carregados\n\n📑 {n_items} análise(s) no relatório")
            if st.session_state.get("outliers_applied"):
                st.warning("🧹 Analisando a base **sem outliers**")
        else:
            st.info("📂 Nenhum dado carregado ainda.")


# --------------------------------------------------------------------------- #
# Tela inicial (home) com cards
# --------------------------------------------------------------------------- #
def home() -> None:
    st.markdown("<h1 class='argus-title'>👁️ Argus Analytics</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='argus-sub' style='font-size:1.15rem;'>"
        "Enxergue seus dados por todos os ângulos — análise estatística simples, "
        "visual e guiada para engenharia e processos.</p>",
        unsafe_allow_html=True,
    )

    if not has_data():
        st.info(
            "👋 **Bem-vindo!** Para começar, clique em **📋 Visão Geral dos Dados** "
            "e importe um arquivo Excel ou CSV. Em seguida escolha a análise desejada "
            "nos cards abaixo."
        )
        if st.button("📋 Importar dados agora", type="primary"):
            st.session_state["page"] = "data_overview"
            st.rerun()
    else:
        df = st.session_state["df"]
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Arquivo", st.session_state.get("file_name") or "—")
        c2.metric("🔢 Linhas", f"{len(df):,}".replace(",", "."))
        c3.metric("📊 Colunas", df.shape[1])

    st.divider()
    st.markdown("### Escolha uma análise")
    st.caption("Cada card abre uma tela dedicada, com seleção simples de variáveis e explicações.")

    # Grid de cards (4 colunas)
    keys = list(ANALYSES.keys())
    cols_per_row = 4
    for i in range(0, len(keys), cols_per_row):
        row = keys[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, key in zip(cols, row):
            meta = ANALYSES[key]
            disabled = meta["needs_data"] and not has_data()
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {meta['icon']} {meta['label']}")
                    st.caption(meta["desc"])
                    if st.button(
                        "Abrir ▶",
                        key=f"card_{key}",
                        use_container_width=True,
                        disabled=disabled,
                    ):
                        st.session_state["page"] = key
                        st.rerun()
                    if disabled:
                        st.caption("🔒 Requer dados carregados")


# --------------------------------------------------------------------------- #
# Roteamento
# --------------------------------------------------------------------------- #
def main() -> None:
    sidebar()
    page = st.session_state.get("page", "home")

    if page == "home":
        home()
        return

    meta = ANALYSES.get(page)
    if meta is None:
        st.session_state["page"] = "home"
        st.rerun()
        return

    # Cabeçalho da análise + botão voltar
    top = st.columns([0.85, 0.15])
    with top[0]:
        st.markdown(f"<h2 class='argus-title'>{meta['icon']} {meta['label']}</h2>",
                    unsafe_allow_html=True)
    with top[1]:
        if st.button("← Início", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()
    st.caption(meta["desc"])
    st.divider()

    try:
        meta["render"](st.session_state)
    except Exception as exc:  # captura amigável de qualquer erro inesperado
        st.error(
            "⚠️ Ocorreu um erro ao executar esta análise. Verifique se as variáveis "
            "escolhidas são adequadas e tente novamente."
        )
        with st.expander("Detalhes técnicos (para depuração)"):
            st.exception(exc)


if __name__ == "__main__":
    main()
