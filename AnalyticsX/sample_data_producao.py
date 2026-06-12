"""
Gera um Excel de teste para o modo "Produção Diária" (Versão 2).

Cria um arquivo com DUAS abas:
  - ``Processo``: dados minuto a minuto (DataHora + indicadores de processo)
  - ``Producao``: produção informada uma vez por dia (Data + Producao)

A produção diária é construída a partir de relações CONHECIDAS, para validar a
análise:
  + média diária de ``Vazao``            (relação direta, mesmo dia)
  − instabilidade diária de ``Temperatura`` (desvio padrão, mesmo dia)
  − tempo de ``pH`` fora da faixa [4.8, 5.6] do DIA ANTERIOR (efeito de lag 1 dia)
  + ruído

Uso:
    python sample_data_producao.py                 # gera sample_producao.xlsx
    python sample_data_producao.py saida.xlsx
"""

import sys

import numpy as np
import pandas as pd

PH_LO, PH_HI = 4.8, 5.6   # faixa ideal de pH usada na geração (e p/ configurar limites)


def generate(days: int = 50, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-05-01 00:00")
    mpd = 24 * 60  # minutos por dia

    proc_parts: list[pd.DataFrame] = []
    prod_rows: list[dict] = []
    prev_ph_out = 0.0  # tempo de pH fora de faixa do dia anterior (efeito de lag)

    for d in range(days):
        day0 = start + pd.Timedelta(days=d)
        idx = pd.date_range(day0, periods=mpd, freq="1min")

        # Alvos diários que variam de um dia para o outro
        vaz_mean = 300.0 + rng.normal(0, 25)          # nível médio de vazão do dia
        temp_std_target = rng.uniform(0.4, 6.0)        # instabilidade da temperatura
        ph_center = 5.2 + rng.normal(0, 0.3)           # centro do pH (pode sair da faixa)

        t = np.linspace(0, 4 * np.pi, mpd)
        vazao = vaz_mean + rng.normal(0, 5, mpd) + 8 * np.sin(t)
        temperatura = 80 + np.cumsum(rng.normal(0, temp_std_target / 25, mpd))
        temperatura = temperatura - temperatura.mean() + 80
        pressao = 2.1 + rng.normal(0, 0.1, mpd)
        ph = ph_center + rng.normal(0, 0.25, mpd)
        brix = 18.0 + rng.normal(0, 0.4, mpd)
        nivel = 60.0 + rng.normal(0, 3, mpd)

        ph_out = float(np.sum((ph < PH_LO) | (ph > PH_HI)))  # minutos fora da faixa

        proc_parts.append(pd.DataFrame({
            "DataHora": idx, "Vazao": vazao, "Temperatura": temperatura,
            "Pressao": pressao, "pH": ph, "Brix": brix, "Nivel": nivel,
        }))

        producao = (
            9000.0
            + 9.0 * (vaz_mean - 300.0)          # + vazão (mesmo dia)
            - 280.0 * temperatura.std(ddof=1)   # − instabilidade térmica (mesmo dia)
            - 6.0 * prev_ph_out                  # − pH fora da faixa (DIA ANTERIOR)
            + rng.normal(0, 120)
        )
        prod_rows.append({"Data": day0.normalize(), "Producao": round(producao, 1)})
        prev_ph_out = ph_out

    processo = pd.concat(proc_parts, ignore_index=True)
    producao = pd.DataFrame(prod_rows)
    return processo, producao


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_producao.xlsx"
    processo, producao = generate()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        processo.to_excel(writer, sheet_name="Processo", index=False)
        producao.to_excel(writer, sheet_name="Producao", index=False)
    print(f"Gerado: {out}")
    print(f"  Aba 'Processo': {len(processo)} linhas (minuto a minuto)")
    print(f"  Aba 'Producao': {len(producao)} dias")
    print(f"  Faixa de pH usada (configure como limite): [{PH_LO}, {PH_HI}]")
    print("  Relações: +Vazao_mean (D0), -Temperatura_std (D0), -pH fora de faixa (D-1)")
