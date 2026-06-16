"""Teste rápido de humanize_variable e suffix_legend_table."""

import analysis

sheets = ["Moenda", "Fermentacao", "Caldo"]
casos = {
    # período atual (sem sufixo de lag) → SEM texto de lag
    "Moenda_Vazao_p10":
        "valores baixos (P10) de Vazao (Moenda)",
    "Moenda_Temperatura_mean_abs_diff":
        "oscilação de Temperatura (Moenda)",
    "Fermentacao_pH_pct_fora":
        "% do tempo fora da faixa de pH (Fermentacao)",
    "Vazao_mean":                   # sem prefixo de aba (1 aba só)
        "média de Vazao",
    "Brix_per_mean":                # periódico, período atual
        "média de Brix",
    "Nivel_mean_t2":
        "média no turno 08–16 de Nivel",
    # com defasagem → menciona o lag
    "Brix_per_mean_lag_240min":
        "média de Brix, com lag de 240 min (4 h antes)",
    "Fermentacao_pH_area_abaixo_lag_480min":
        "gravidade abaixo do limite de pH (Fermentacao), com lag de 480 min (8 h antes)",
    "Temperatura_std_lag_1d":       # legado em dias
        "instabilidade de Temperatura, com lag de 1 dia",
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
