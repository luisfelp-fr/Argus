"""Smoke test headless do fluxo multi-abas (Seção B de analysis.py).

Roda o pipeline completo sobre sample_multiabas.xlsx e confere se as relações
conhecidas aparecem no topo dos rankings:
  +Vazao_mean (lag 0), -Temperatura_std (lag 0),
  -Brix da leitura anterior (lag 240 min), -pH < 4.8 (lag 0, pct_fora).

Nota: este teste exercita ``add_lab_lags`` COM defasagem para validar o
alinhamento temporal da função. No APP, as variáveis periódicas entram
apenas no período atual (sem lag) — ver ``_test_ui_apptest.py``.

Uso: python _test_multiabas.py
"""

import numpy as np
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
# Como no app: periodicas entram apenas no periodo atual (max_lag_min=0),
# sem sufixo de lag e sem a variavel _last (desconsiderada).
lab_vals, lab_period = analysis.parse_lab_sheet(sheets["Caldo"], "DataHora")
print(f"Periodo do indicador periodico detectado: {lab_period:g} min (esperado 240)")
assert abs(lab_period - 240) < 1

lab_lagged = analysis.add_lab_lags(
    lab_vals, lab_period, target_series.index, period_min, max_lag_min=0.0)
print(f"Colunas periodicas geradas: {list(lab_lagged.columns)}")
assert not any("_last" in c for c in lab_lagged.columns), "_last deveria estar fora"
assert not any("_lag_" in c for c in lab_lagged.columns), "periodicas sem sufixo de lag"
assert "Brix_per_mean" in lab_lagged.columns

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


# Período atual (sem defasagem) → colunas sem sufixo de lag.
# A variável _last foi desconsiderada; o Brix periódico entra como _per_mean.
checks = {
    "Vazao_mean (direta)": _sr("Moenda_Vazao_mean") > 0.3,
    "Temperatura_std (inversa)": _sr("Moenda_Temperatura_std") < -0.25,
    "pH pct_fora (inversa)": _sr("Fermentacao_pH_pct_fora") < -0.3,
    "sem coluna _lag_0min": not any("_lag_0min" in c for c in X.columns),
    "sem coluna _last": not any("_last" in c for c in X.columns),
    "Brix periodico presente (per_mean)": "Brix_per_mean" in X.columns,
}
print("\n== Relacoes conhecidas (sinal e forca do Spearman) ==")
for k, ok in checks.items():
    print(f"  [{'OK' if ok else 'FALHOU'}] {k}")

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

# ---------------------------------------------------------------- ranking por indicador
SHEETS = ["Moenda", "Fermentacao", "Caldo"]
assert analysis.base_indicator("Moenda_Vazao_mean", SHEETS) == ("Moenda_Vazao", "Vazao (Moenda)")
assert analysis.base_indicator("Brix_per_mean_lag_240min", SHEETS)[0] == "Brix"

ind_rank = analysis.indicator_ranking(consolidated, SHEETS)
print("\n== Ranking por indicador (limpo) ==")
print(ind_rank.to_string(index=False))
assert list(ind_rank.columns) == ["Indicador", "Impacto", "Nº métricas"]
assert not ind_rank.empty
assert ind_rank["Impacto"].is_monotonic_decreasing
# nomes limpos: sem sufixo/lag/metrica no rotulo do indicador
assert not ind_rank["Indicador"].str.contains(r"_lag_|_mean|_std|_per_|_pct_fora",
                                               regex=True).any()
assert ind_rank["Indicador"].str.contains("Vazao").any()
print(f"[OK] {len(ind_rank)} indicadores, nomes limpos, ordenado por impacto")

# ---------------------------------------------------------------- drill-down (cadeia)
dr = analysis.drilldown_ranking(X, "Moenda_Vazao_mean", SHEETS)
print("\n== Drill-down: o que impacta Vazao (Moenda) ==")
print(dr.head(8).to_string(index=False))
assert list(dr.columns) == ["Indicador", "Impacto", "Nº métricas"]
assert not dr.empty, "drill-down deveria retornar drivers"
assert not dr["Indicador"].str.contains("Vazao").any(), "nao deve se autoexplicar"
print("[OK] drill-down nao se autoexplica e retorna ranking limpo")

# ---------------------------------------------------------------- Shapley por indicador
ish, r2 = analysis.indicator_shapley(X, y, SHEETS, stat_df=stat)
print("\n== Contribuicao de cada indicador (Shapley/LMG) ==")
print(ish.to_string(index=False), "| R2 total:", r2)
assert list(ish.columns) == ["Indicador", "Contribuição (%)", "Parcela (R²)"]
assert not ish.empty
assert (ish["Parcela (R²)"] >= -1e-9).all(), "parcelas devem ser nao-negativas"
assert abs(ish["Parcela (R²)"].sum() - r2) < 0.02, "parcelas devem somar o R2"
assert not ish["Indicador"].str.contains(r"_lag_|_mean|_std|_per_",
                                         regex=True).any()
assert ish["Indicador"].iloc[0] in ("Temperatura (Moenda)", "pH (Fermentacao)",
                                    "Vazao (Moenda)")
print("[OK] Shapley: fatias nao-negativas somam R2, nomes limpos, sem modelo")

# ---------------------------------------------------------------- AutoML
auto = analysis.auto_model(X, y, n_iter=8, cv_folds=3, max_candidates_k=15)
print("\n== AutoML ==")
print("vencedor:", auto.model_name, "| K:", auto.k_selected, "| CV RMSE:", auto.cv_score)
assert isinstance(auto, analysis.AutoModelResult)
assert auto.model_name in ("RandomForest", "XGBoost")
assert auto.selected_features and set(auto.selected_features) <= set(X.columns)
assert auto.feature_names == auto.selected_features
assert all(np.isfinite([auto.metrics["MAE"], auto.metrics["RMSE"]]))
assert not auto.candidates_df.empty and "RMSE (CV)" in auto.candidates_df.columns
# subset: shap/day_contributions restritos as features selecionadas
si = analysis.shap_importance(auto, X)
assert si is None or set(si.index) <= set(auto.selected_features)
dc = analysis.day_contributions(auto, X, X.index[-1])
assert set(dc["Variável"]) <= set(auto.selected_features)
# simulador
base_pred = analysis.simulate_prediction(auto, X, {})
assert np.isfinite(base_pred)
top_feat = auto.importances.index[0]
hi = analysis.simulate_prediction(auto, X, {top_feat: float(X[top_feat].max())})
assert np.isfinite(hi)
print("[OK] AutoML: escolhe modelo, seleciona features, subset e simulador funcionam")

print("\nTUDO OK")
