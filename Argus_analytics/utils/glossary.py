"""
glossary.py — Glossário estatístico + cards de faixas de referência
===================================================================

Reúne:
  * ``GLOSSARY``: textos curtos e didáticos para os ícones de ajuda "?"
    (parâmetro ``help=`` dos widgets do Streamlit) ao lado de conceitos e testes.
  * ``band_cards``: componente visual que mostra, em cards, as faixas de
    consideração de um indicador (CV, assimetria, curtose) e destaca em qual
    faixa o valor atual se encontra.
"""

from __future__ import annotations

import math

try:
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False

from .plotting import (
    ARGUS_OK, ARGUS_ACCENT, ARGUS_DANGER, ARGUS_SECONDARY, ARGUS_PRIMARY,
)

# Cores em CSS (com '#') ----------------------------------------------------- #
C_OK = "#" + ARGUS_OK.lstrip("#")
C_ACCENT = "#" + ARGUS_ACCENT.lstrip("#")
C_DANGER = "#" + ARGUS_DANGER.lstrip("#")
C_BLUE = "#" + ARGUS_SECONDARY.lstrip("#")
C_PRIMARY = "#" + ARGUS_PRIMARY.lstrip("#")


# --------------------------------------------------------------------------- #
# Glossário — textos dos ícones de ajuda "?"
# --------------------------------------------------------------------------- #
GLOSSARY: dict[str, str] = {
    # Estatística descritiva
    "media": "Média: soma dos valores ÷ quantidade. É o valor típico, mas é "
             "sensível a valores extremos (outliers).",
    "mediana": "Mediana: valor central quando os dados são ordenados. É robusta "
               "a outliers — metade dos dados fica abaixo, metade acima.",
    "desvio_padrao": "Desvio padrão: dispersão média em torno da média, na mesma "
                     "unidade dos dados. Quanto maior, mais espalhados os valores.",
    "variancia": "Variância: média dos desvios ao quadrado (é o desvio padrão²). "
                 "Mede dispersão, mas na unidade ao quadrado.",
    "quartis": "Quartis dividem os dados ordenados em 4 partes iguais: Q1 deixa "
               "25% abaixo e Q3 deixa 75% abaixo.",
    "amplitude": "Amplitude = valor máximo − valor mínimo. Mostra a faixa total "
                 "coberta pelos dados.",
    "cv": "Coeficiente de Variação (CV) = desvio padrão ÷ média, em %. Mede a "
          "dispersão RELATIVA: permite comparar a variação de grandezas de "
          "escalas diferentes. Quanto maior, mais heterogêneos os dados.",
    "assimetria": "Assimetria (skewness): mede se a distribuição é simétrica. "
                  "Positiva = cauda longa à direita (valores altos); negativa = "
                  "cauda longa à esquerda (valores baixos); ~0 = simétrica.",
    "curtose": "Curtose (em excesso): mede o pico e o peso das caudas. 0 ≈ normal; "
               "positiva = pico acentuado e caudas pesadas (mais outliers); "
               "negativa = distribuição mais achatada.",
    # Normalidade
    "p_valor": "p-valor: probabilidade de observar um resultado tão extremo quanto "
               "o atual se a hipótese nula fosse verdadeira. Por convenção, "
               "p ≤ 0,05 indica resultado estatisticamente significativo.",
    "shapiro": "Shapiro-Wilk: teste de normalidade. p-valor > 0,05 sugere que os "
               "dados NÃO fogem da normalidade (não rejeita a normal).",
    "anderson": "Anderson-Darling: teste de normalidade que dá mais peso às caudas. "
                "Se a estatística A² < valor crítico, não se rejeita a normalidade.",
    "normalidade": "Normalidade: indica se os dados seguem a distribuição normal "
                   "(curva em sino). É premissa de muitos testes paramétricos.",
    # Correlação
    "correlacao": "Correlação varia de −1 a +1: o sinal indica a direção e o módulo "
                  "indica a força. Atenção: correlação NÃO implica causalidade.",
    "pearson": "Pearson: mede a relação LINEAR entre duas variáveis numéricas. "
               "É sensível a outliers e assume aproximada normalidade.",
    "spearman": "Spearman: mede relação MONOTÔNICA usando postos (ranks). Mais "
                "robusto a outliers e a relações não lineares.",
    "kendall": "Kendall (tau): baseia-se na concordância de ordenação entre pares. "
               "Robusto e indicado para amostras pequenas.",
    # Regressão
    "r2": "R² (coeficiente de determinação): fração da variação da resposta "
          "explicada pelo modelo, de 0 a 1. Ex.: 0,75 ≈ 75% explicado.",
    "r2_aj": "R² ajustado: corrige o R² pelo número de variáveis, penalizando "
             "variáveis irrelevantes. Útil para comparar modelos.",
    "coeficiente": "Coeficiente: variação esperada na resposta para cada +1 unidade "
                   "da variável explicativa, mantendo as demais constantes.",
    "residuo": "Resíduo: diferença entre o valor real e o previsto pelo modelo. "
               "Idealmente devem se espalhar aleatoriamente em torno de zero.",
    "regressao": "Regressão: modela como uma ou mais variáveis explicativas (X) "
                 "influenciam uma variável resposta (Y).",
    # Comparação de grupos
    "teste_t": "Teste t: compara as MÉDIAS de 2 grupos. É paramétrico (assume "
               "aproximada normalidade dos dados).",
    "mann_whitney": "Mann-Whitney: alternativa NÃO paramétrica ao teste t; compara "
                    "as distribuições de 2 grupos sem exigir normalidade.",
    "anova": "ANOVA: compara as médias de 3 ou mais grupos simultaneamente "
             "(paramétrico).",
    "kruskal": "Kruskal-Wallis: alternativa NÃO paramétrica à ANOVA, para 3 ou "
               "mais grupos.",
    # CEP / Capabilidade
    "cep": "Carta de controle (CEP): acompanha o processo ao longo do tempo. "
           "Pontos entre os limites = variação comum; fora = possível causa especial.",
    "limites_controle": "LSC/LIC: limites de controle a ±3 sigma da média, "
                        "estimados pela amplitude móvel. NÃO confundir com limites "
                        "de especificação — descrevem o que o processo faz, não o "
                        "que o cliente exige.",
    "cp": "Cp: compara a largura das especificações com a variação do processo "
          "(6σ). NÃO considera a centralização. Cp ≥ 1,33 costuma ser bom.",
    "cpk": "Cpk: como o Cp, mas considera a centralização do processo. "
           "Cpk bem menor que Cp indica processo descentralizado.",
    "lie_lse": "LIE/LSE: limites INFERIOR e SUPERIOR de especificação definidos "
               "pelo cliente ou projeto (a faixa do que é aceitável).",
    # Temporal / Lag
    "media_movel": "Média móvel: média de uma janela deslizante de períodos. "
                   "Suaviza o ruído e evidencia a tendência.",
    "lag": "Lag (defasagem): atraso entre uma variável (causa) e o efeito em "
           "outra. Lag positivo: a variável explicativa antecede o alvo.",
    "cross_correlation": "Correlação cruzada: correlação entre as variáveis em "
                         "vários atrasos; o pico indica o lag mais provável.",
    # Outliers
    "iqr": "IQR (1,5×amplitude interquartílica): marca como outlier valores abaixo "
           "de Q1 − 1,5·IQR ou acima de Q3 + 1,5·IQR.",
    "zscore": "Z-score: nº de desvios padrão em relação à média. |z| > 3 costuma "
              "indicar outlier (assume distribuição aproximadamente normal).",
    "mad": "MAD (desvio absoluto mediano): método robusto baseado na mediana, "
           "indicado quando há muitos outliers.",
    "outlier": "Outlier: valor muito diferente do padrão geral. Pode ser erro de "
               "medição, falha de processo ou evento real importante.",
}


def help_for(key: str) -> str | None:
    """Retorna o texto de ajuda para um conceito (ou None se inexistente)."""
    return GLOSSARY.get(key)


# --------------------------------------------------------------------------- #
# Faixas de referência (cards)
# --------------------------------------------------------------------------- #
def _band(name, rng, desc, color, active):
    return {"name": name, "range": rng, "desc": desc, "color": color, "active": active}


def cv_bands(cv: float) -> list[dict]:
    """Faixas do Coeficiente de Variação (%)."""
    ok = cv is not None and not (isinstance(cv, float) and math.isnan(cv))
    a = abs(cv) if ok else None
    return [
        _band("Baixa", "CV < 15%",
              "Dados homogêneos: pouca variação relativa à média.",
              C_OK, ok and a < 15),
        _band("Moderada", "15% – 30%",
              "Variação relativa intermediária; vale acompanhar.",
              C_ACCENT, ok and 15 <= a < 30),
        _band("Alta", "CV > 30%",
              "Dados heterogêneos: muita dispersão em relação ao valor típico.",
              C_DANGER, ok and a >= 30),
    ]


def skew_bands(skew: float) -> list[dict]:
    """Faixas de assimetria (skewness)."""
    ok = skew is not None and not (isinstance(skew, float) and math.isnan(skew))
    a = abs(skew) if ok else None
    return [
        _band("Simétrica", "|s| < 0,5",
              "Distribuição aproximadamente simétrica.",
              C_OK, ok and a < 0.5),
        _band("Assimetria leve", "0,5 ≤ |s| < 1",
              "Uma cauda é um pouco mais longa que a outra.",
              C_ACCENT, ok and 0.5 <= a < 1),
        _band("Assimetria forte", "|s| ≥ 1",
              "Cauda nitidamente longa de um dos lados.",
              C_DANGER, ok and a >= 1),
    ]


def kurtosis_bands(kurt: float) -> list[dict]:
    """Faixas de curtose em excesso (0 ≈ normal)."""
    ok = kurt is not None and not (isinstance(kurt, float) and math.isnan(kurt))
    return [
        _band("Achatada", "k < −0,5",
              "Mais achatada que a normal; caudas leves (platicúrtica).",
              C_BLUE, ok and kurt < -0.5),
        _band("Próxima da normal", "−0,5 ≤ k ≤ 0,5",
              "Pico e caudas parecidos com a distribuição normal.",
              C_OK, ok and -0.5 <= kurt <= 0.5),
        _band("Pico / caudas pesadas", "k > 0,5",
              "Pico acentuado e/ou caudas pesadas: mais outliers (leptocúrtica).",
              C_DANGER, ok and kurt > 0.5),
    ]


def band_cards(title: str, value_display: str, bands: list[dict],
               help_text: str | None = None) -> None:
    """Renderiza o valor (com ícone de ajuda) + cards das faixas de consideração.

    O card da faixa em que o valor se encontra é destacado.
    """
    if not _HAS_ST:
        return
    st.metric(title, value_display, help=help_text)
    cols = st.columns(len(bands))
    for col, b in zip(cols, bands):
        with col:
            with st.container(border=True):
                chip = (
                    f"<span style='background:{b['color']};color:white;"
                    f"padding:2px 9px;border-radius:9px;font-size:0.78rem;"
                    f"font-weight:700'>{b['range']}</span>"
                )
                st.markdown(chip, unsafe_allow_html=True)
                badge = "✅ " if b["active"] else ""
                st.markdown(f"**{badge}{b['name']}**")
                st.caption(b["desc"])
                if b["active"]:
                    st.markdown(
                        f"<span style='color:{b['color']};font-weight:700;"
                        f"font-size:0.8rem'>◄ seu valor está nesta faixa</span>",
                        unsafe_allow_html=True,
                    )
