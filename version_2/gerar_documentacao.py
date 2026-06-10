"""
Gera a documentação técnica do programa em PDF, com fórmulas e modelos estatísticos.

As fórmulas são tipografadas pelo renderizador matemático do matplotlib (mathtext),
sem necessidade de LaTeX instalado, e embutidas no PDF montado com reportlab.

Uso:
    python gerar_documentacao.py            # gera Documentacao_Analise_Correlacao.pdf
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer,
)

_TMP = Path(tempfile.mkdtemp(prefix="doc_formulas_"))
_styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=_styles["Heading1"], fontSize=15, spaceBefore=14,
                    spaceAfter=6, textColor="#0b3d61")
H2 = ParagraphStyle("H2", parent=_styles["Heading2"], fontSize=12, spaceBefore=10,
                    spaceAfter=4, textColor="#15598c")
BODY = ParagraphStyle("Body", parent=_styles["BodyText"], fontSize=10.5, leading=15,
                      alignment=TA_JUSTIFY, spaceAfter=6)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9.5, textColor="#444444",
                      leftIndent=10, spaceAfter=6)
TITLE = ParagraphStyle("Title", parent=_styles["Title"], fontSize=22,
                       textColor="#0b3d61")
SUBTITLE = ParagraphStyle("Sub", parent=_styles["Normal"], fontSize=11,
                          textColor="#555555", spaceAfter=2)

_formula_counter = [0]


def formula(latex: str, fontsize: int = 17):
    """Renderiza uma fórmula LaTeX (mathtext) para um flowable de imagem centrado.

    Se o mathtext não conseguir interpretar, cai para texto monoespaçado.
    """
    _formula_counter[0] += 1
    path = _TMP / f"f{_formula_counter[0]}.png"
    dpi = 220
    try:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#111111")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                    transparent=False, facecolor="white")
        plt.close(fig)
        w, h = ImageReader(str(path)).getSize()
        wpt, hpt = w * 72.0 / dpi, h * 72.0 / dpi
        max_w = 16 * cm
        if wpt > max_w:
            hpt *= max_w / wpt
            wpt = max_w
        img = Image(str(path), width=wpt, height=hpt)
        img.hAlign = "CENTER"
        return img
    except Exception:  # noqa: BLE001 - fallback se algum símbolo não for suportado
        plt.close("all")
        mono = ParagraphStyle("mono", parent=BODY, fontName="Courier",
                              alignment=1, fontSize=10)
        return Paragraph(latex.replace("\\", ""), mono)


def p(text: str):
    return Paragraph(text, BODY)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BODY), leftIndent=6) for t in items],
        bulletType="bullet", start="•", leftIndent=14,
    )


# --------------------------------------------------------------------------- #
# Conteúdo
# --------------------------------------------------------------------------- #
def build_story():
    s = []
    s += [Paragraph("Análise de Correlação Multivariada", TITLE)]
    s += [Paragraph("Documentação técnica — modelos estatísticos, fórmulas e "
                    "cálculo de estabilidade", SUBTITLE)]
    s += [Spacer(1, 0.3 * cm)]
    s += [p("Este documento descreve o que o programa faz e detalha os modelos "
            "estatísticos internos. A ferramenta foi concebida para engenheiros "
            "de processo identificarem a <b>causa</b> de aumentos ou reduções de um "
            "<b>indicador-chave</b> a partir de uma série de indicadores de processo, "
            "considerando o <b>atraso (lag)</b> entre causa e efeito.")]

    # 1. Visão geral
    s += [Paragraph("1. Visão geral e fluxo de processamento", H1)]
    s += [p("O programa recebe uma planilha com uma coluna de data/hora e várias "
            "colunas de indicadores. O usuário escolhe o indicador-chave (o efeito a "
            "explicar) e o sistema executa, em sequência:")]
    s += [bullets([
        "<b>Leitura e padronização</b> dos dados (numéricos, ordenados no tempo);",
        "<b>Reamostragem</b> numa grade temporal regular (ou agregação composta, "
        "no caso de variáveis de laboratório);",
        "<b>Detecção do lag ótimo</b> de cada indicador via correlação cruzada;",
        "<b>Quantificação da relação</b> por correlação de Pearson e Informação Mútua;",
        "<b>Atribuição rigorosa de variância</b> (Shapley/LMG) com tratamento de "
        "multicolinearidade;",
        "<b>Avaliação de estabilidade</b> de cada indicador e geração de "
        "gatilhos acionáveis.",
    ])]

    # 2. Reamostragem
    s += [Paragraph("2. Reamostragem temporal", H1)]
    s += [p("O passo temporal é estimado pela mediana das diferenças entre "
            "instantes consecutivos:")]
    s += [formula(r"\Delta t = \mathrm{mediana}\,\{\, t_{i+1}-t_i \,\}")]
    s += [p("Os dados são reamostrados para uma grade regular de passo "
            "<i>Δt</i> (média dentro de cada intervalo). Lacunas curtas (até 3 "
            "amostras consecutivas) são interpoladas linearmente; lacunas maiores "
            "permanecem vazias e são ignoradas nos cálculos.")]

    # 3. Variáveis de laboratório
    s += [Paragraph("3. Variáveis de laboratório (amostras compostas)", H1)]
    s += [p("Variáveis medidas em laboratório por composição (ex.: a cada 4 horas) "
            "têm um valor que representa a <b>média do período</b>, e por isso "
            "<b>não podem ser interpoladas</b>. Sendo <i>T</i> o instante registrado "
            "(fim do período) e <i>N</i> o número de horas da composição, define-se a "
            "janela do período como:")]
    s += [formula(r"W_T = (\,T - N,\; T\,]")]
    s += [p("Para cada coleta de laboratório em <i>T</i>, todos os indicadores são "
            "agregados pela média na janela, colocando processo e laboratório na "
            "mesma base composta:")]
    s += [formula(r"\bar{X}_j(T) = \frac{1}{|W_T|}\sum_{t\,\in\,W_T} X_j(t)")]
    s += [Paragraph("As variáveis de laboratório, tendo um único valor na janela, "
                    "resultam no próprio valor reportado. As demais análises seguem "
                    "sobre esse conjunto composto (passo = N horas).", NOTE)]

    # 4. Lag ótimo
    s += [Paragraph("4. Detecção do lag ótimo (correlação cruzada / TLCC)", H1)]
    s += [p("Para cada indicador, varre-se um conjunto de defasagens (lags) e "
            "calcula-se a correlação de Pearson entre o indicador-chave e o "
            "indicador deslocado no tempo. A correlação de Pearson é:")]
    s += [formula(r"r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}"
                  r"{\sqrt{\sum_i (x_i-\bar{x})^2}\;\sqrt{\sum_i (y_i-\bar{y})^2}}")]
    s += [p("A função de correlação cruzada em função do lag ℓ é:")]
    s += [formula(r"r(\ell) = \mathrm{corr}\left(\,\mathrm{chave}_t,\; "
                  r"\mathrm{indicador}_{t-\ell}\,\right)")]
    s += [p("O lag ótimo é aquele que <b>maximiza o módulo</b> da correlação "
            "(captando tanto relações diretas quanto inversas):")]
    s += [formula(r"\ell^{*} = \mathrm{argmax}_{\ell}\; \left| r(\ell) \right|")]
    s += [p("E é convertido em minutos pelo passo temporal:")]
    s += [formula(r"\mathrm{lag}_{\min} = \ell^{*}\cdot \Delta t")]
    s += [Paragraph("Convenção: lag positivo significa que o indicador "
                    "<b>antecede</b> o indicador-chave (a causa ocorre antes do "
                    "efeito).", NOTE)]

    # 5. Informação Mútua
    s += [Paragraph("5. Informação Mútua (relações não-lineares)", H1)]
    s += [p("Além da correlação linear, mede-se a dependência (inclusive "
            "não-linear) entre o indicador no lag ótimo e o indicador-chave pela "
            "Informação Mútua:")]
    s += [formula(r"I(X;Y) = \sum_{x}\sum_{y} p(x,y)\,"
                  r"\log\frac{p(x,y)}{p(x)\,p(y)}")]
    s += [p("Valores próximos de zero indicam independência; valores maiores "
            "indicam dependência mais forte. É estimada por k-vizinhos "
            "(scikit-learn, <i>mutual_info_regression</i>).")]

    s += [PageBreak()]

    # 6. Atribuição rigorosa
    s += [Paragraph("6. Atribuição rigorosa de variância (Shapley / LMG)", H1)]
    s += [p("A correlação individual superestima o impacto quando há indicadores "
            "correlacionados entre si. Para dar a cada driver uma <b>fatia real e "
            "não-sobreposta</b> da variância do indicador-chave, monta-se uma "
            "regressão linear múltipla com os indicadores alinhados pelo seu lag "
            "ótimo e decompõe-se o coeficiente de determinação.")]

    s += [Paragraph("6.1. Coeficiente de determinação (R²)", H2)]
    s += [p("Para um conjunto de preditores, o R² mede a fração da variância do "
            "indicador-chave explicada pelo modelo:")]
    s += [formula(r"R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}"
                  r"{\sum_i (y_i-\bar{y})^2}")]

    s += [Paragraph("6.2. Multicolinearidade (VIF) e sua eliminação", H2)]
    s += [p("O Fator de Inflação da Variância mede o quanto um indicador é "
            "redundante em relação aos demais. Sendo <i>R²ⱼ</i> o R² da regressão do "
            "indicador <i>j</i> contra os outros:")]
    s += [formula(r"\mathrm{VIF}_j = \frac{1}{1 - R_j^{2}}")]
    s += [p("VIF alto (tipicamente &gt; 10) indica forte redundância. Quando a "
            "eliminação está ativa, o programa: (i) agrupa indicadores com "
            "correlação mútua |corr| ≥ 0,90, mantendo em cada grupo o mais "
            "relacionado ao indicador-chave; e (ii) remove iterativamente o de maior "
            "VIF enquanto exceder o limiar. Os removidos são reportados.")]

    s += [Paragraph("6.3. Decomposição de Shapley / LMG", H2)]
    s += [p("A fatia de cada indicador <i>i</i> é a média da sua contribuição "
            "marginal ao R² sobre todos os subconjuntos <i>S</i> dos demais "
            "preditores (valor de Shapley aplicado ao R²). Com <i>p</i> preditores e "
            "<i>N</i> o conjunto completo:")]
    s += [formula(r"\phi_i = \sum_{S \subseteq N\setminus\{i\}} "
                  r"\frac{|S|!\,(p-|S|-1)!}{p!}\,"
                  r"\left[\,R^2(S\cup\{i\}) - R^2(S)\,\right]")]
    s += [p("Essa decomposição tem três propriedades fundamentais: as fatias são "
            "<b>não-negativas</b>, <b>não-sobrepostas</b> e <b>somam exatamente o R² "
            "do modelo completo</b>:")]
    s += [formula(r"\sum_{i=1}^{p} \phi_i = R^2(N)")]
    s += [Paragraph("Para até 10 preditores o cálculo é exato (todos os "
                    "subconjuntos); acima disso, aproxima-se por amostragem de "
                    "permutações aleatórias (mesma esperança).", NOTE)]

    s += [Paragraph("6.4. Métricas exibidas", H2)]
    s += [p("A partir das fatias φᵢ, o painel mostra:")]
    s += [p("Fatia da variância (parcela real da variação do indicador-chave "
            "atribuída ao driver):")]
    s += [formula(r"\mathrm{Fatia}_i\,(\%) = 100 \cdot \phi_i")]
    s += [p("Contribuição no modelo (a mesma fatia normalizada para somar 100%):")]
    s += [formula(r"\mathrm{Contrib}_i\,(\%) = 100 \cdot "
                  r"\frac{\phi_i}{R^2(N)}")]
    s += [p("Na aba de análise detalhada também é exibido o impacto simples por "
            "indicador (R² individual = correlação ao quadrado):")]
    s += [formula(r"\mathrm{Impacto}_i\,(\%) = 100 \cdot r_i^{2}")]

    # 7. Estabilidade
    s += [Paragraph("7. Cálculo de estabilidade de um indicador", H1)]
    s += [p("A estabilidade é avaliada no <b>período todo analisado</b> pelo "
            "<b>coeficiente de variação</b> (CV), métrica padrão de estabilidade de "
            "processo: mede o quanto o indicador oscilou em relação ao seu nível "
            "típico. Sendo σ o desvio padrão e μ a média da série:")]
    s += [formula(r"\mathrm{CV} = \frac{\sigma}{\left|\mu\right|}")]
    s += [p("A classificação usa dois limiares (o limiar de variação forte é "
            "ajustável no aplicativo; valores padrão abaixo):")]
    s += [bullets([
        "<b>Estável</b>: CV &lt; 5%;",
        "<b>Moderada</b>: 5% ≤ CV &lt; 15%;",
        "<b>Instável (variação forte)</b>: CV ≥ 15%.",
    ])]
    s += [p("A tendência líquida do período é o sinal da diferença entre a média do "
            "último e do primeiro terço da série:")]
    s += [formula(r"\mathrm{tend} = \mathrm{sinal}\left("
                  r"\bar{x}_{\mathrm{fim}} - \bar{x}_{\mathrm{ini}}\right)")]
    s += [Paragraph("O CV não é uma medida válida quando a série cruza o zero (nível "
                    "mal definido / média próxima de zero); nesses casos a "
                    "estabilidade é reportada como '—'.", NOTE)]

    # 8. Maior distúrbio
    s += [Paragraph("8. Detecção do maior distúrbio (evidência causa→efeito)", H1)]
    s += [p("Para o gráfico de evidência, localiza-se o maior distúrbio histórico do "
            "driver pelo z-score ponto a ponto:")]
    s += [formula(r"z_t = \frac{x_t - \bar{x}}{\sigma_x}, \qquad "
                  r"t^{*} = \mathrm{argmax}_t\, |z_t|")]
    s += [p("A janela da causa é o trecho contíguo em torno do pico onde o z-score "
            "permanece relevante, |zₜ| ≥ max(0,5·|z(t*)|, 1,5). A janela de efeito é "
            "a janela da causa deslocada para frente pelo lag — evidenciando, na "
            "linha de tempo real, qual trecho do driver impactou o indicador-chave.")]

    # 9. Gatilhos
    s += [Paragraph("9. Gatilhos acionáveis", H1)]
    s += [p("Um gatilho é emitido quando um driver que <b>antecede</b> o "
            "indicador-chave (lag &gt; 0) apresentou <b>forte variação no período todo</b> "
            "(CV ≥ limiar configurável). O sentido provável do indicador-chave combina a "
            "tendência do driver no período com o sinal da correlação no lag ótimo:")]
    s += [formula(r"\mathrm{dir}_{\mathrm{chave}} = "
                  r"\mathrm{sinal}(\mathrm{tend})\,\cdot\,"
                  r"\mathrm{sinal}\left(r(\ell^{*})\right)")]
    s += [p("Assim, o painel destaca os indicadores que mais variaram no período e que, "
            "por anteciparem o indicador-chave, são as prováveis causas a monitorar — "
            "indicando o sentido da influência e a fatia de impacto correspondente.")]

    # 10. Premissas
    s += [Paragraph("10. Premissas e limitações", H1)]
    s += [bullets([
        "Correlação não implica causalidade; o lag e o conhecimento de processo "
        "ajudam a sustentar a interpretação causal.",
        "A detecção de lag pressupõe séries razoavelmente estacionárias; séries com "
        "forte tendência podem deslocar o pico de correlação.",
        "No modo laboratório, a resolução de lag é limitada ao período de composição "
        "(ex.: 4 horas) — atrasos menores não são detectáveis.",
        "A atribuição de Shapley pressupõe relação aproximadamente linear no modelo "
        "de regressão usado para o R².",
    ])]

    s += [Spacer(1, 0.4 * cm)]
    s += [Paragraph("Documento gerado automaticamente a partir do código-fonte do "
                    "programa (analysis.py / app.py).", NOTE)]
    return s


def main(out="Documentacao_Analise_Correlacao.pdf"):
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Análise de Correlação Multivariada — Documentação técnica",
        author="Análise de Correlação Multivariada",
    )
    doc.build(build_story())
    print(f"Gerado: {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Documentacao_Analise_Correlacao.pdf")
