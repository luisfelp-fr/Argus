"""
Gera uma planilha de exemplo COM variável de laboratório (amostra composta).

- Indicadores de processo (1 em 1 min): Vazao, Pressao, Temperatura.
- Variável de laboratório `DQO_lab` amostrada de 4 em 4 h, cujo valor é a MÉDIA
  do período (T − 4 h, T] de um sinal dirigido pela Pressao + ruído. Nas demais
  linhas a coluna fica vazia (não pode ser interpolada).

Use no app: ligue "Há indicadores de laboratório", selecione `DQO_lab`, período 4 h,
e escolha `DQO_lab` como indicador-chave.

Uso:
    python sample_data_lab.py            # gera sample_data_lab.xlsx
"""

import sys

import numpy as np
import pandas as pd

PERIOD_H = 4  # período da composição (horas)


def generate(days: int = 5, step_min: int = 1, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60 // step_min
    idx = pd.date_range("2026-01-01 00:00", periods=n, freq=f"{step_min}min")

    def smooth_noise(window: int) -> np.ndarray:
        x = rng.normal(0, 1, n + window)
        x = np.convolve(x, np.ones(window) / window, mode="same")[:n]
        return x - x.mean()

    vazao = smooth_noise(5) * 3 + 50
    pressao = smooth_noise(5) * 2 + 10
    temperatura = smooth_noise(3) * 1.5 + 70

    df = pd.DataFrame(
        {"Vazao": vazao, "Pressao": pressao, "Temperatura": temperatura}, index=idx
    )

    # Variável de laboratório: média do período de um sinal dirigido pela Pressao
    driver = 4.0 * pressao + rng.normal(0, 2, n)
    period = pd.Timedelta(hours=PERIOD_H)
    lab_times = pd.date_range(
        idx[0] + period, idx[-1], freq=f"{PERIOD_H}h"
    )
    dqo = pd.Series(np.nan, index=idx)
    drv = pd.Series(driver, index=idx)
    for T in lab_times:
        win = drv.loc[(drv.index > T - period) & (drv.index <= T)]
        dqo.loc[T] = float(win.mean())
    df["DQO_lab"] = dqo

    df.index.name = "DataHora"
    return df.reset_index()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data_lab.xlsx"
    df = generate()
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False, sep=";")
    else:
        df.to_excel(out, index=False, engine="openpyxl")
    n_lab = int(df["DQO_lab"].notna().sum())
    print(f"Gerado: {out}  ({len(df)} linhas de processo, {n_lab} coletas de DQO_lab)")
    print(f"No app: ligue 'laboratório', selecione DQO_lab, período {PERIOD_H} h, chave = DQO_lab.")
