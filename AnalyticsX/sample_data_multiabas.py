"""
Gera um Excel de teste para o fluxo MULTI-ABAS do modo sazonal (Seção B).

Cria um arquivo com QUATRO abas:
  - ``Moenda``      (contínua, 1 min): Vazao, Temperatura, Pressao
  - ``Fermentacao`` (contínua, 5 min): pH, Nivel  — passo diferente p/ testar
                    a detecção por aba; pH com excursões abaixo de 4.8
  - ``Caldo``       (periódica, a cada 4 h): Brix, Pol — cada valor informado
                    em T é a média de um processo oculto nas 4 h anteriores
                    (semântica (T−4h, T])
  - ``Alvo``        (a cada 8 h): Rendimento, com relações CONHECIDAS:
      + a·Vazao_mean (janela atual)
      − b·Temperatura_std (janela atual)
      − c·Brix da LEITURA ANTERIOR (efeito defasado de 240 min)
      − d·minutos de pH < 4.8 na janela atual
      + ruído

Uso:
    python sample_data_multiabas.py                 # gera sample_multiabas.xlsx
    python sample_data_multiabas.py saida.xlsx
"""

import sys

import numpy as np
import pandas as pd

PH_LO = 4.8               # limite inferior crítico do pH (configure na UI)
LAB_PERIOD_H = 4          # leitura do indicador periódico a cada 4 h
TARGET_PERIOD_H = 8       # alvo divulgado a cada 8 h


def generate(days: int = 40, seed: int = 11):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-05-01 00:00")
    total_min = days * 24 * 60

    # ----------------------- Moenda (1 min) ----------------------------- #
    idx1 = pd.date_range(start, periods=total_min, freq="1min")
    n_win = days * 24 // TARGET_PERIOD_H          # nº de janelas do alvo
    win_len = TARGET_PERIOD_H * 60                # minutos por janela do alvo

    # nível de vazão e instabilidade térmica sorteados POR JANELA do alvo
    vaz_levels = 300.0 + rng.normal(0, 25, n_win)
    temp_stds = rng.uniform(0.4, 6.0, n_win)

    vazao = np.concatenate([
        lv + rng.normal(0, 5, win_len) + 8 * np.sin(np.linspace(0, 4 * np.pi, win_len))
        for lv in vaz_levels
    ])
    temperatura = np.concatenate([
        80 + (lambda w: w - w.mean())(np.cumsum(rng.normal(0, sd / 25, win_len)))
        for sd in temp_stds
    ])
    pressao = 2.1 + rng.normal(0, 0.1, total_min)
    moenda = pd.DataFrame({
        "DataHora": idx1, "Vazao": vazao, "Temperatura": temperatura, "Pressao": pressao,
    })

    # ----------------------- Fermentacao (5 min) ------------------------ #
    idx5 = pd.date_range(start, periods=total_min // 5, freq="5min")
    n5_win = win_len // 5
    # pH com centro sorteado por janela; ~25% das janelas com excursão < 4.8
    ph_centers = np.where(rng.random(n_win) < 0.25,
                          4.65 + rng.normal(0, 0.05, n_win),
                          5.20 + rng.normal(0, 0.15, n_win))
    ph = np.concatenate([c + rng.normal(0, 0.08, n5_win) for c in ph_centers])
    nivel = 60.0 + rng.normal(0, 3, len(idx5))
    fermentacao = pd.DataFrame({"DataHora": idx5, "pH": ph, "Nivel": nivel})

    # minutos de pH < 4.8 por janela do alvo (cada amostra de 5 min vale 5 min)
    ph_out_min = np.array([
        float(np.sum(ph[i * n5_win:(i + 1) * n5_win] < PH_LO) * 5)
        for i in range(n_win)
    ])

    # ----------------------- Caldo (periódico, 4 h) --------------------- #
    # leituras geradas como AR(1): sinal forte por leitura e baixa
    # autocorrelação entre janelas (permite distinguir o lag correto)
    def _ar1(mean: float, phi: float, sigma: float, n: int) -> np.ndarray:
        x = np.empty(n)
        x[0] = mean + rng.normal(0, sigma)
        for k in range(1, n):
            x[k] = mean + phi * (x[k - 1] - mean) + rng.normal(0, sigma)
        return x

    lab_step = LAB_PERIOD_H * 60
    lab_times = pd.date_range(start + pd.Timedelta(hours=LAB_PERIOD_H),
                              periods=total_min // lab_step, freq=f"{LAB_PERIOD_H}h")
    brix_lab = _ar1(18.0, 0.3, 0.35, len(lab_times))
    pol_lab = _ar1(14.5, 0.3, 0.30, len(lab_times))
    lab = pd.DataFrame({"DataHora": lab_times, "Brix": brix_lab, "Pol": pol_lab})

    # ----------------------- Alvo (8 h) --------------------------------- #
    alvo_times = pd.date_range(start + pd.Timedelta(hours=TARGET_PERIOD_H),
                               periods=n_win, freq=f"{TARGET_PERIOD_H}h")
    rows = []
    for i, T in enumerate(alvo_times):
        # leitura periódica ANTERIOR à janela atual: última com T_lab <= T − 4 h
        prev_pos = np.searchsorted(
            lab_times.values, np.datetime64(T - pd.Timedelta(hours=LAB_PERIOD_H)),
            side="right") - 1
        brix_prev = brix_lab[prev_pos] if prev_pos >= 0 else 18.0
        rendimento = (
            85.0
            + 0.060 * (vaz_levels[i] - 300.0)       # + vazão média (janela atual)
            - 1.200 * temp_stds[i]                   # − instabilidade térmica
            - 4.000 * (brix_prev - 18.0)             # − Brix da LEITURA ANTERIOR
            - 0.010 * ph_out_min[i]                  # − minutos de pH < 4.8
            + rng.normal(0, 0.35)
        )
        rows.append({"DataHora": T, "Rendimento": round(rendimento, 3)})
    alvo = pd.DataFrame(rows)

    return moenda, fermentacao, lab, alvo


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_multiabas.xlsx"
    moenda, fermentacao, lab, alvo = generate()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        moenda.to_excel(writer, sheet_name="Moenda", index=False)
        fermentacao.to_excel(writer, sheet_name="Fermentacao", index=False)
        lab.to_excel(writer, sheet_name="Caldo", index=False)
        alvo.to_excel(writer, sheet_name="Alvo", index=False)
    print(f"Gerado: {out}")
    print(f"  Aba 'Moenda'      (continua, 1 min):  {len(moenda)} linhas")
    print(f"  Aba 'Fermentacao' (continua, 5 min):  {len(fermentacao)} linhas")
    print(f"  Aba 'Caldo'       (periodico, 4 h):   {len(lab)} leituras")
    print(f"  Aba 'Alvo'        (8 h):              {len(alvo)} leituras")
    print(f"  Limite a configurar: pH minimo = {PH_LO} (aba Fermentacao)")
    print("  Relacoes esperadas: +Vazao_mean (lag 0), -Temperatura_std (lag 0),")
    print("    -Brix leitura anterior (lag 240 min), -pH abaixo de 4.8 (lag 0)")
