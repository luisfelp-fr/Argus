"""Teste rápido de humanize_variable e suffix_legend_table."""

import analysis

sheets = ["Moenda", "Fermentacao", "Lab_Caldo"]
casos = {
    "Moenda_Vazao_p10_lag_0min":
        "10º percentil de Vazao (Moenda), sem lag",
    "Moenda_Temperatura_mean_abs_diff_lag_0min":
        "variação média entre leituras de Temperatura (Moenda), sem lag",
    "Fermentacao_pH_pct_fora_lag_0min":
        "% do tempo fora da faixa de pH (Fermentacao), sem lag",
    "Brix_lab_lab_last_lag_240min":
        "última análise de laboratório de Brix lab, com lag de 240 min (4 h antes)",
    "Fermentacao_pH_area_abaixo_lag_480min":
        "severidade abaixo do limite de pH (Fermentacao), com lag de 480 min (8 h antes)",
    "Vazao_mean_lag_0min":          # sem prefixo de aba (1 aba só)
        "média de Vazao, sem lag",
    "Fermentacao_pH_pct_fora":      # sem lag (tabela de excursões)
        "% do tempo fora da faixa de pH (Fermentacao)",
    "Temperatura_std_lag_1d":       # legado em dias
        "desvio padrão de Temperatura, com lag de 1 dia",
    "Nivel_mean_t2_lag_0min":
        "média no turno 08–16 de Nivel, sem lag",
}
falhas = 0
for nome, esperado in casos.items():
    obtido = analysis.humanize_variable(nome, sheets)
    status = "OK    " if obtido == esperado else "DIFERE"
    if obtido != esperado:
        falhas += 1
    print(f"[{status}] {nome}\n         -> {obtido}")
    if obtido != esperado:
        print(f"         esperado: {esperado}")

legend = analysis.suffix_legend_table()
print(f"\nLegenda: {len(legend)} sufixos em {legend['Categoria'].nunique()} categorias")
assert len(legend) >= 30
assert falhas == 0, f"{falhas} casos divergentes"
print("TUDO OK")
