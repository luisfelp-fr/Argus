"""Smoke test headless do fluxo multi-abas (Seção B de analysis.py).

Roda o pipeline completo sobre sample_multiabas.xlsx e confere se as relações
conhecidas aparecem no topo dos rankings:
  +Vazao_mean (lag 0), -Temperatura_std (lag 0),
  -Brix da leitura anterior (lag 240 min), -pH < 4.8 (lag 0, pct_fora).

Uso: python _test_multiabas.py
"""

import pandas as pd

import analysis

XLSX = "sample_multiabas.xlsx"

# ---------------------------------------------------------------- leitura
sheets = {name: pd.read_excel(XLSX, sheet_name=name)
          for name in ("Moenda", "Fermentacao", "Caldo", "Alvo")}

print("== guess_sheet_role ==")
for name, df in sheets.items():
    print(f"  {name}: {analysis.guess_sheet_role(name, df)}"
          f" (dt_col detectada: {analysis.guess_datetime_column(df)})")

# ---------------------------------------------------------------- alvo
target_series, period_min = analysis.parse_target_series(
    sheets["Alvo"], "DataHora", "Rendimento")
print(f"\nPeriodo do alvo detectado: {period_min:g} min (esperado 480)")
assert abs(period_min - 480) < 1

# ---------------------------------------------------------------- continuas
limits_ferm = {"pH": (4.8, None)}
cont_feat, steps = analysis.build_period_features_multi(
    [
        {"name": "Moenda", "df": sheets["Moenda"], "dt_col": "DataHora", "limits": {}},
        {"name": "Fermentacao", "df": sheets["Fermentacao"], "dt_col": "DataHora",
         "limits": limits_ferm},
    ],
    target_series.index, period_min,
)
print(f"Passos detectados por aba: {steps} (esperado 1 e 5 min)")
assert abs(steps["Moenda"] - 1) < 0.01 and abs(steps["Fermentacao"] - 5) < 0.01

MAX_LAG = 480.0
cont_lagged = analysis.add_lag_features_min(cont_feat, MAX_LAG, period_min)

# ---------------------------------------------------------------- periodicos
lab_vals, lab_period = analysis.parse_lab_sheet(sheets["Caldo"], "DataHora")
print(f"Periodo do indicador periodico detectado: {lab_period:g} min (esperado 240)")
assert abs(lab_period - 240) < 1

lab_lagged = analysis.add_lab_lags(
    lab_vals, lab_period, target_series.index, period_min, max_lag_min=MAX_LAG)
print(f"Colunas periodicas geradas: {list(lab_lagged.columns)[:6]} ... "
      f"({lab_lagged.shape[1]} no total)")

# ---------------------------------------------------------------- merge + limpeza
all_feat = cont_lagged.join(lab_lagged)
merged = analysis.merge_with_target(all_feat, target_series)
print(f"Base de modelagem: {merged.shape[0]} periodos x {merged.shape[1]} colunas")

clean, rep = analysis.clean_modeling_table(merged)
X, y = analysis.split_features_target(clean)

# ---------------------------------------------------------------- estatistica
stat = analysis.statistical_ranking(X, y)
print("\n== Top 12 do ranking estatistico (|Spearman|) ==")
print(stat.head(12).to_string(index=False))

stat_idx = stat.set_index("Variável")


def _sr(var: str) -> float:
    return float(stat_idx.loc[var, "Spearman r"]) if var in stat_idx.index else float("nan")


# A relação do Brix usa a ÚLTIMA leitura até T−240min → é a feature
# ``per_last`` no lag 240 que a captura exatamente (a ``per_mean`` da janela de
# 480 min mistura duas leituras e não discrimina o lag).
checks = {
    "Vazao_mean lag 0 (direta)": _sr("Moenda_Vazao_mean_lag_0min") > 0.3,
    "Temperatura_std lag 0 (inversa)": _sr("Moenda_Temperatura_std_lag_0min") < -0.25,
    "Brix last lag 240 (inversa)": _sr("Brix_per_last_lag_240min") < -0.3,
    "Brix: lag 240 mais forte que lag 0 e 480": (
        abs(_sr("Brix_per_last_lag_240min")) > abs(_sr("Brix_per_last_lag_0min"))
        and abs(_sr("Brix_per_last_lag_240min")) > abs(_sr("Brix_per_last_lag_480min"))
    ),
    "pH pct_fora lag 0 (inversa)": _sr("Fermentacao_pH_pct_fora_lag_0min") < -0.3,
}
print("\n== Relacoes conhecidas (sinal e forca do Spearman) ==")
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FALHOU'}] {k}")
print("  Brix last lag 0 / 240 / 480:",
      f"{_sr('Brix_per_last_lag_0min'):+.3f} /",
      f"{_sr('Brix_per_last_lag_240min'):+.3f} /",
      f"{_sr('Brix_per_last_lag_480min'):+.3f}")

# ---------------------------------------------------------------- excursoes
ferm = sheets["Fermentacao"]
ev = analysis.detect_excursion_events(ferm, "DataHora", "pH", 4.8, None, 5.0)
total_min = len(ferm) * 5.0
summ = analysis.excursion_summary({"pH": ev}, {"pH": total_min})
print("\n== Resumo de excursoes (pH < 4.8) ==")
print(summ.to_string(index=False))
assert len(ev) > 0, "esperava eventos de excursao de pH"

exc_tbl = analysis.excursion_vs_target(merged)
print("\n== Excursao vs alvo (Mann-Whitney) ==")
print(exc_tbl.to_string(index=False))
ph_rows = exc_tbl[exc_tbl["Variável"].str.contains("pH", case=False)]
assert not ph_rows.empty and float(ph_rows["Mann-Whitney p"].iloc[0]) < 0.05, \
    "esperava Mann-Whitney p < 0.05 para o pH"

# ---------------------------------------------------------------- modelo + texto
model = analysis.train_model(X, y, model_type="RandomForest")
shap_imp = analysis.shap_importance(model, X)
consolidated = analysis.consolidated_ranking(stat, model.importances, shap_imp)
texto = analysis.generate_report_text(
    consolidated, model, stat, excursion_tbl=exc_tbl,
    period_min=period_min, n_periodos=len(clean),
    sheet_names=["Moenda", "Fermentacao", "Caldo"])
print(f"\nR2 do modelo (teste): {model.metrics['R2']:.3f}")
print("\n== Texto do relatorio ==")
print(texto)

assert all(checks.values()), f"relacoes nao confirmadas: {checks}"
print("\nTUDO OK")
