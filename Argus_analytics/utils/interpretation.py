"""
interpretation.py — Geração automática de textos interpretativos
================================================================

Funções puras (sem Streamlit) que convertem resultados numéricos em frases
claras e aplicadas, no estilo "assistente estatístico". Cada função devolve uma
string em português, pronta para exibir na tela e no relatório.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Correlação
# --------------------------------------------------------------------------- #
def correlation_strength(r: float) -> str:
    """Classifica a força da correlação pela magnitude (|r|)."""
    a = abs(r)
    if a < 0.20:
        return "muito fraca"
    if a < 0.40:
        return "fraca"
    if a < 0.60:
        return "moderada"
    if a < 0.80:
        return "forte"
    return "muito forte"


def interpret_correlation(var_x: str, var_y: str, r: float, method: str = "Pearson") -> str:
    if np.isnan(r):
        return (f"Não foi possível calcular a correlação entre **{var_x}** e "
                f"**{var_y}** (dados insuficientes ou variância nula).")
    direction = "positiva" if r >= 0 else "negativa"
    strength = correlation_strength(r)
    sense = ("quando uma aumenta, a outra tende a aumentar"
             if r >= 0 else
             "quando uma aumenta, a outra tende a diminuir")
    return (
        f"A variável **{var_x}** apresentou correlação {direction} {strength} "
        f"com **{var_y}** ({method} r = {r:.3f}). Isso indica que {sense}. "
        "No entanto, esta relação não prova causalidade — correlação indica "
        "associação, não causa."
    )


# --------------------------------------------------------------------------- #
# Normalidade
# --------------------------------------------------------------------------- #
def interpret_normality(p_value: float, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "Não foi possível avaliar a normalidade (dados insuficientes)."
    if p_value > alpha:
        return (
            f"Com p-valor = {p_value:.4f} (> {alpha:g}), **não há evidência "
            "estatística suficiente para afirmar que os dados fogem da "
            "normalidade**. Testes paramétricos (t, ANOVA, regressão) tendem a "
            "ser apropriados."
        )
    return (
        f"Com p-valor = {p_value:.4f} (≤ {alpha:g}), **os dados provavelmente não "
        "seguem distribuição normal**. Para este caso, pode ser mais adequado "
        "utilizar análises não paramétricas (ex.: Mann-Whitney, Kruskal-Wallis, "
        "correlação de Spearman)."
    )


# --------------------------------------------------------------------------- #
# Regressão
# --------------------------------------------------------------------------- #
def interpret_r2(r2: float) -> str:
    pct = r2 * 100
    quality = (
        "muito alto" if r2 >= 0.8 else
        "alto" if r2 >= 0.6 else
        "moderado" if r2 >= 0.4 else
        "baixo"
    )
    return (
        f"O modelo apresentou R² de {r2:.3f}, indicando que aproximadamente "
        f"**{pct:.1f}% da variação da variável resposta** foi explicada pelas "
        f"variáveis selecionadas (poder explicativo {quality}). O restante "
        "decorre de fatores não incluídos no modelo ou de variabilidade natural."
    )


def interpret_coefficients(names: list[str], coefs: list[float], pvalues: list[float],
                           alpha: float = 0.05) -> str:
    sig = [(n, c) for n, c, p in zip(names, coefs, pvalues) if p <= alpha and n != "const"]
    if not sig:
        return ("Nenhuma variável explicativa apresentou efeito estatisticamente "
                f"significativo (p ≤ {alpha:g}). Considere revisar as variáveis escolhidas.")
    parts = []
    for n, c in sig:
        sense = "aumenta" if c >= 0 else "reduz"
        parts.append(f"a cada unidade de **{n}**, a resposta {sense} em {abs(c):.4g}")
    return ("Variáveis com efeito significativo (p ≤ "
            f"{alpha:g}): " + "; ".join(parts) + ".")


# --------------------------------------------------------------------------- #
# Comparação entre grupos
# --------------------------------------------------------------------------- #
def interpret_group_test(test_name: str, p_value: float, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "Não foi possível concluir o teste (dados insuficientes)."
    if p_value <= alpha:
        return (
            f"O teste {test_name} indicou diferença **estatisticamente significativa** "
            f"entre os grupos (p = {p_value:.4f} ≤ {alpha:g}). Os grupos provavelmente "
            "apresentam comportamento diferente para esta variável."
        )
    return (
        f"O teste {test_name} **não** encontrou diferença estatisticamente "
        f"significativa entre os grupos (p = {p_value:.4f} > {alpha:g}). Não há "
        "evidência de que os grupos difiram quanto a esta variável."
    )


# --------------------------------------------------------------------------- #
# Carta de controle (CEP)
# --------------------------------------------------------------------------- #
def interpret_control_chart(n_out: int, total: int) -> str:
    if n_out == 0:
        return (
            f"Todos os {total} pontos estão dentro dos limites de controle. O processo "
            "aparenta estar **estável** (sob controle estatístico), com apenas "
            "variação de causas comuns."
        )
    return (
        f"Foram identificados **{n_out} ponto(s) fora dos limites de controle** "
        f"(de {total}), indicando possível **instabilidade** no processo. Pontos fora "
        "podem sinalizar causas especiais de variação que merecem investigação."
    )


# --------------------------------------------------------------------------- #
# Capabilidade Cp / Cpk
# --------------------------------------------------------------------------- #
def capability_class(value: float) -> str:
    if value < 1.0:
        return "potencialmente incapaz"
    if value < 1.33:
        return "marginal"
    if value < 1.67:
        return "geralmente capaz"
    return "robusto"


def interpret_capability(cp: float, cpk: float) -> str:
    cls = capability_class(cpk)
    centered = ("bem centralizado" if abs(cp - cpk) < 0.1 * max(cp, 1e-9)
                else "deslocado em relação ao centro das especificações")
    return (
        f"O processo apresentou **Cp = {cp:.2f}** e **Cpk = {cpk:.2f}**, sendo "
        f"classificado como **{cls}**. A diferença entre Cp e Cpk mostra que o "
        f"processo está {centered}. Lembre-se: Cp avalia se a variação cabe nos "
        "limites; Cpk também considera a centralização."
    )


# --------------------------------------------------------------------------- #
# Análise com lag
# --------------------------------------------------------------------------- #
def interpret_lag(var: str, target: str, lag_min: float, corr: float) -> str:
    strength = correlation_strength(corr)
    direction = "positiva" if corr >= 0 else "negativa"
    if abs(lag_min) < 1e-9:
        timing = "sem atraso aparente (efeito praticamente imediato)"
    elif lag_min > 0:
        timing = f"com atraso de cerca de **{lag_min:g} min** (a variável antecede o alvo)"
    else:
        timing = f"com defasagem de {abs(lag_min):g} min"
    return (
        f"**{var}** mostrou correlação {direction} {strength} (r = {corr:.3f}) com "
        f"**{target}** {timing}. Em processos industriais, esse atraso pode refletir "
        "o tempo entre uma alteração operacional e seu efeito medido."
    )


def interpret_lag_validation(var: str, target: str, lag_min: float,
                             coef: float, exog_p: float, lb_p: float) -> str:
    """Interpreta a validação ARIMAX de um lag (significância + Ljung-Box)."""
    sig = bool(exog_p == exog_p and exog_p < 0.05)
    wn = bool(lb_p == lb_p and lb_p > 0.05)
    sentido = "aumenta" if coef >= 0 else "reduz"
    when = (f"defasada em **{lag_min:g} min**" if abs(lag_min) > 1e-9
            else "sem defasagem")

    if sig and wn:
        veredito = (
            f"✅ **Lag validado.** Mesmo controlando a dinâmica própria de "
            f"**{target}** (modelo ARIMAX), **{var}** {when} tem efeito "
            f"estatisticamente significativo (p = {exog_p:.4f}) e {sentido} o "
            f"alvo (coef. = {coef:.4g}). Os resíduos são ruído branco "
            f"(Ljung-Box p = {lb_p:.4f}), indicando que o modelo capturou bem a "
            "estrutura temporal — a relação não parece espúria."
        )
    elif sig and not wn:
        veredito = (
            f"⚠️ **Validação parcial.** **{var}** {when} é significativa "
            f"(p = {exog_p:.4f}), mas os resíduos ainda mostram autocorrelação "
            f"(Ljung-Box p = {lb_p:.4f}). Ajuste a ordem do modelo ou o lag — pode "
            "haver dinâmica não capturada."
        )
    elif not sig and wn:
        veredito = (
            f"❌ **Lag não confirmado.** Com a dinâmica de **{target}** modelada, "
            f"**{var}** {when} deixa de ser significativa (p = {exog_p:.4f}). A "
            "correlação cruzada provavelmente refletia tendência/autocorrelação "
            "comum, não um efeito direto."
        )
    else:
        veredito = (
            f"❌ **Não validado.** **{var}** {when} não é significativa "
            f"(p = {exog_p:.4f}) e os resíduos não são ruído branco "
            f"(Ljung-Box p = {lb_p:.4f}). Reavalie o lag, a ordem do modelo ou a "
            "qualidade da série."
        )
    return veredito


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def interpret_outliers(n_out: int, total: int, method: str) -> str:
    if n_out == 0:
        return (f"Nenhum outlier foi detectado pelo método **{method}**. Os valores "
                "estão dentro do comportamento esperado.")
    pct = 100 * n_out / total if total else 0
    return (
        f"Foram identificados **{n_out} possível(is) outlier(s)** ({pct:.1f}% dos "
        f"{total} valores) pelo método **{method}**. Outliers podem representar erro "
        "de medição, falha de processo ou eventos reais importantes — avalie antes "
        "de removê-los."
    )


# --------------------------------------------------------------------------- #
# Estatística descritiva / distribuição
# --------------------------------------------------------------------------- #
def interpret_descriptive(name: str, mean: float, median: float, cv: float) -> str:
    skew_note = ("a média e a mediana são próximas, sugerindo distribuição "
                 "aproximadamente simétrica"
                 if abs(mean - median) <= 0.1 * (abs(mean) + 1e-9) else
                 "há diferença entre média e mediana, sugerindo assimetria")
    # CV pode ser indefinido (NaN) quando a média está próxima de zero — nesse
    # caso a dispersão relativa não se aplica e não deve ser classificada.
    if cv != cv:  # NaN
        return (
            f"Para **{name}**, o valor típico gira em torno de {mean:.4g} "
            f"(mediana {median:.4g}); {skew_note}. O coeficiente de variação não se "
            "aplica aqui porque a média está próxima de zero — avalie a dispersão "
            "pelo **desvio padrão** ou pela amplitude."
        )
    disp = ("baixa dispersão" if cv < 15 else
            "dispersão moderada" if cv < 30 else "alta dispersão")
    return (
        f"Para **{name}**, o valor típico gira em torno de {mean:.4g} "
        f"(mediana {median:.4g}); {skew_note}. O coeficiente de variação de "
        f"{cv:.1f}% indica **{disp}** dos dados."
    )


def interpret_skewness(skew: float) -> str:
    if skew != skew:  # NaN — variável praticamente constante
        return ("A variável é praticamente **constante** (sem variação), então "
                "assimetria e curtose não se aplicam.")
    if abs(skew) < 0.5:
        return "A distribuição é aproximadamente **simétrica**."
    side = "direita (cauda longa para valores altos)" if skew > 0 else "esquerda (cauda longa para valores baixos)"
    intensity = "leve" if abs(skew) < 1 else "acentuada"
    return f"A distribuição apresenta assimetria {intensity} à {side} (skew = {skew:.2f})."
