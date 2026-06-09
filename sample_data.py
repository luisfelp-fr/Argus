"""
Gera uma planilha sintética com lag conhecido para validar a análise.

O indicador-chave ("Temperatura") é construído como soma de duas "causas"
DESLOCADAS no tempo (com atraso conhecido) + ruído. Indicadores irrelevantes
(ruído puro) também são incluídos.

Uso:
    python sample_data.py            # gera sample_data.xlsx
    python sample_data.py saida.csv  # gera CSV
"""

import sys

import numpy as np
import pandas as pd

# Lags verdadeiros, em número de amostras (passo de 1 min abaixo)
LAG_VAZAO = 5      # Vazão antecede a Temperatura em 5 min
LAG_PRESSAO = 12   # Pressão antecede a Temperatura em 12 min


def generate(n: int = 1000, step_min: int = 1, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq=f"{step_min}min")

    def smooth_noise(window: int) -> np.ndarray:
        """Ruído branco suavizado: estacionário e com autocorrelação curta,
        de modo que o pico do lag fica bem definido (ideal para validação)."""
        x = rng.normal(0, 1, n + window)
        x = np.convolve(x, np.ones(window) / window, mode="same")[:n]
        return x - x.mean()

    # Causas independentes entre si (evita correlação espúria), com níveis médios
    # físicos para que o coeficiente de variação (estabilidade) seja interpretável.
    vazao = smooth_noise(window=3) * 3 + 50
    pressao = smooth_noise(window=3) * 2 + 10

    # Efeito = combinação das causas DESLOCADAS + ruído (em torno de um nível)
    temperatura = (
        70.0
        + 0.8 * pd.Series(vazao - 50).shift(LAG_VAZAO).fillna(0).to_numpy()
        + 1.5 * pd.Series(pressao - 10).shift(LAG_PRESSAO).fillna(0).to_numpy()
        + rng.normal(0, 1.0, n)
    )

    df = pd.DataFrame(
        {
            "DataHora": idx,
            "Temperatura": temperatura,   # indicador-chave (efeito)
            "Vazao": vazao,               # causa com lag 5
            "Pressao": pressao,           # causa com lag 12
            "Ruido": rng.normal(0, 1, n), # irrelevante
        }
    )
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data.xlsx"
    df = generate()
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False, sep=";")
    else:
        df.to_excel(out, index=False, engine="openpyxl")
    print(f"Gerado: {out}")
    print(f"Lags verdadeiros -> Vazao: {LAG_VAZAO} min | Pressao: {LAG_PRESSAO} min")
