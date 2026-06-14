"""
Textos de ajuda (tooltips) e glossário em PT-BR
-----------------------------------------------
Centraliza as explicações de conceitos estatísticos e de aprendizado de máquina
usadas em toda a interface (parâmetro ``help=`` dos widgets do Streamlit) e no
expander "ℹ️ Glossário". Mantém a linguagem acessível ao time de processo.
"""

from __future__ import annotations

HELP: dict[str, str] = {
    # ----------------------------------------------------------------- #
    # Associação estatística
    # ----------------------------------------------------------------- #
    "pearson": (
        "Pearson r: correlação LINEAR entre a variável e o alvo (−1 a +1; "
        "0 = sem relação linear). Simples e direta, mas sensível a valores extremos."
    ),
    "spearman": (
        "Spearman r: correlação de ORDEM (ranking); capta relações monotônicas "
        "(sobe-sobe ou sobe-desce) mesmo não lineares e é mais robusta a outliers (−1 a +1)."
    ),
    "p_value": (
        "p-value: probabilidade de a associação observada ter surgido por acaso. "
        "Quanto MENOR (ex.: < 0,05), mais significativa estatisticamente."
    ),
    "mutual_information": (
        "Mutual Information (Informação Mútua): mede qualquer tipo de dependência "
        "(inclusive não linear) entre a variável e o alvo (≥ 0; 0 = independentes). "
        "Não indica o sentido do efeito."
    ),
    "estatistica": (
        "Pearson r: correlação LINEAR (−1 a +1). "
        "Spearman r: correlação de ORDEM, capta relações monotônicas não lineares (−1 a +1). "
        "p-value: probabilidade de a associação ser por acaso — quanto MENOR, mais significativa. "
        "Mutual Information: dependência geral, inclusive não linear (≥ 0)."
    ),
    # ----------------------------------------------------------------- #
    # Tempo / lag / período
    # ----------------------------------------------------------------- #
    "lag": (
        "Maior atraso (em minutos) a considerar entre o processo e o indicador alvo. "
        "O sistema cria blocos de períodos anteriores até esse limite (passo = período)."
    ),
    "periodo_sazonal": (
        "Janela em que o processo é resumido para cada leitura do indicador alvo. "
        "Detectado automaticamente pelos horários do indicador; ajuste se necessário."
    ),
    "split_temporal": (
        "Split temporal: os dados são divididos respeitando a ordem do tempo — o modelo "
        "treina no passado (75%) e é avaliado no futuro (25%). Evita 'prever o passado "
        "com o futuro', que inflaria artificialmente o desempenho."
    ),
    "lab_janela": (
        "Indicador periódico (laboratório ou qualquer leitura a cada x horas): o valor "
        "informado no horário T representa a condição MÉDIA entre a leitura anterior e T "
        "(janela (T−x, T], onde x é o intervalo entre leituras, detectado pela mediana do "
        "espaçamento). O sistema distribui cada valor nessa janela antes de alinhá-lo aos "
        "períodos do alvo."
    ),
    "lab_lag": (
        "Lag do indicador periódico: defasagens criadas em múltiplos do intervalo entre "
        "leituras (ex.: leitura a cada 4 h → lags de 0, 240, 480 min...), até o lag máximo "
        "definido. Permite captar efeitos da leitura anterior sobre o período atual do alvo."
    ),
    # ----------------------------------------------------------------- #
    # Modelos e explicabilidade
    # ----------------------------------------------------------------- #
    "tipo_analise": (
        "Só estatística: correlações e informação mútua. "
        "Estatística + Modelo: adiciona um modelo preditivo (importância das variáveis). "
        "Completa (com SHAP): adiciona a explicabilidade SHAP."
    ),
    "modelo": (
        "RandomForest: floresta de árvores de decisão — robusto e estável. "
        "XGBoost: gradient boosting — costuma ser mais preciso, porém mais sensível ao ajuste."
    ),
    "random_forest": (
        "RandomForest: treina muitas árvores de decisão, cada uma vendo amostras e "
        "variáveis diferentes, e combina as previsões pela média. Captura relações não "
        "lineares e interações, com pouco ajuste e boa robustez a ruído."
    ),
    "xgboost": (
        "XGBoost: treina árvores em sequência, cada nova árvore corrigindo o erro das "
        "anteriores, com regularização contra sobreajuste. Costuma ser mais preciso em "
        "dados tabulares, porém mais sensível ao ajuste de parâmetros."
    ),
    "importancia_modelo": (
        "Importância do modelo: o quanto cada variável reduz o erro nas árvores do modelo. "
        "Mostra quem o modelo mais usou, mas não o sentido do efeito."
    ),
    "shap": (
        "SHAP: impacto médio (em módulo) de cada variável nas previsões do modelo — "
        "o quanto cada uma 'empurra' o resultado para cima ou para baixo (teoria dos jogos)."
    ),
    "score_consolidado": (
        "Score consolidado: combina Spearman, Informação Mútua, importância do modelo e SHAP "
        "(normalizados de 0 a 1) num único índice de evidência."
    ),
    # ----------------------------------------------------------------- #
    # Métricas de desempenho
    # ----------------------------------------------------------------- #
    "mae": ("Erro Absoluto Médio: diferença média (nas unidades do indicador) entre "
            "previsto e real. Menor é melhor."),
    "rmse": ("Raiz do Erro Quadrático Médio: como o MAE, mas penaliza mais os erros "
             "grandes. Menor é melhor."),
    "mape": "Erro Percentual Absoluto Médio: erro médio em %. Menor é melhor.",
    "r2": ("R² (coeficiente de determinação): fração da variação explicada pelo modelo "
           "(1 = perfeito; ≤ 0 = não explica)."),
    # ----------------------------------------------------------------- #
    # Estabilidade / multicolinearidade
    # ----------------------------------------------------------------- #
    "cv": (
        "CV (coeficiente de variação): desvio-padrão dividido pela média (em %). Mede a "
        "instabilidade relativa do indicador — quanto maior, mais instável."
    ),
    "vif": (
        "VIF (fator de inflação de variância): mede o quanto uma variável é explicada "
        "pelas demais. VIF alto (> 10) indica redundância — a variável traz pouca "
        "informação nova."
    ),
    "multicolinearidade": (
        "Multicolinearidade: quando variáveis explicativas são fortemente correlacionadas "
        "entre si, dividindo o mesmo 'crédito' e distorcendo a atribuição de importância. "
        "O sistema pode remover as redundantes antes da atribuição."
    ),
    # ----------------------------------------------------------------- #
    # Limites e excursões
    # ----------------------------------------------------------------- #
    "limites": (
        "Limites críticos: faixas mínima e/ou máxima de operação definidas por você. "
        "Para cada indicador com limite, o sistema gera variáveis extras (tempo e área "
        "fora da faixa) e análises adicionais de excursão."
    ),
    "pct_fora": (
        "% fora de faixa: fração do tempo do período em que o indicador ficou fora dos "
        "limites definidos. Para indicadores periódicos, é calculada sobre a série "
        "expandida em janelas (ponderação por tempo), não sobre as leituras pontuais."
    ),
    "area_fora": (
        "Área fora de faixa: soma de (desvio além do limite × tempo). Combina a "
        "intensidade e a duração da violação numa única medida de severidade."
    ),
    "excursao": (
        "Excursão: evento contíguo em que o indicador permaneceu fora da faixa definida. "
        "Cada evento tem início, fim, duração, lado (abaixo/acima), pico de desvio e área."
    ),
    "mann_whitney": (
        "Teste de Mann-Whitney U: compara o alvo nos períodos COM violação de limite vs. "
        "SEM violação, sem assumir distribuição normal. p-value < 0,05 sugere que a "
        "violação está associada a uma mudança real no alvo."
    ),
    "boxplot": (
        "Boxplot: resume a distribuição (mediana, quartis e extremos). Comparar as caixas "
        "com e sem violação mostra se o alvo muda quando o limite é desrespeitado."
    ),
    # ----------------------------------------------------------------- #
    # Relatório
    # ----------------------------------------------------------------- #
    "anexar_relatorio": (
        "Adiciona este gráfico/tabela ao relatório final (aba 📄 Relatório), onde você "
        "pode ordenar, comentar e exportar em HTML ou PDF."
    ),
}


def glossario_markdown() -> str:
    """Conteúdo do expander "ℹ️ Glossário" — organizado por tema."""
    return (
        "#### 🤖 Modelos preditivos\n"
        f"- **RandomForest** — {HELP['random_forest']}\n"
        f"- **XGBoost** — {HELP['xgboost']}\n"
        "\n"
        "#### 📊 Análises estatísticas de associação\n"
        f"- **Pearson r** — {HELP['pearson']}\n"
        f"- **Spearman r** — {HELP['spearman']} É o critério principal de ordenação do ranking.\n"
        f"- **p-value** — {HELP['p_value']}\n"
        f"- **Informação Mútua** — {HELP['mutual_information']}\n"
        "\n"
        "#### 🧠 Explicabilidade (por que o modelo previu aquilo)\n"
        f"- **Importância do modelo** — {HELP['importancia_modelo']}\n"
        f"- **SHAP** — {HELP['shap']}\n"
        "\n"
        "#### 🎯 Métricas de desempenho do modelo (medidas no teste)\n"
        f"- **MAE** — {HELP['mae']}\n"
        f"- **RMSE** — {HELP['rmse']}\n"
        f"- **MAPE** — {HELP['mape']}\n"
        f"- **R²** — {HELP['r2']}\n"
        f"- **Split temporal** — {HELP['split_temporal']}\n"
        "\n"
        "#### 🚦 Limites críticos e excursões\n"
        f"- **Limites críticos** — {HELP['limites']}\n"
        f"- **Excursão** — {HELP['excursao']}\n"
        f"- **% fora de faixa** — {HELP['pct_fora']}\n"
        f"- **Área fora de faixa** — {HELP['area_fora']}\n"
        f"- **Mann-Whitney U** — {HELP['mann_whitney']}\n"
        "\n"
        "#### 🔁 Indicadores periódicos (laboratório ou leituras a cada x horas)\n"
        f"- **Janela da leitura** — {HELP['lab_janela']}\n"
        f"- **Lag do indicador periódico** — {HELP['lab_lag']}\n"
        "\n"
        "#### 🧩 Outros conceitos\n"
        f"- **Lag (minutos)** — {HELP['lag']}\n"
        f"- **Score consolidado** — {HELP['score_consolidado']}\n"
        f"- **Período sazonal** — {HELP['periodo_sazonal']}\n"
        f"- **CV (estabilidade)** — {HELP['cv']}\n"
        f"- **VIF / multicolinearidade** — {HELP['multicolinearidade']}\n"
    )
