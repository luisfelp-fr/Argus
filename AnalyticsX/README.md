# Argus AnalyticsX - Análise Avançada de Processo

App em Streamlit com **dois modos**, selecionados no topo da página:

- **📈 Análise Sazonal (novo):** planilha **multi-abas** (variáveis contínuas +
  periódicas + alvo, classificadas pelo usuário); correlaciona o processo com um
  indicador alvo divulgado em período de horas, com limites críticos, excursões e
  **relatório final HTML/PDF**. Veja a seção *"Modo Análise Sazonal"* abaixo.
- **📊 Correlação Clássica (v1):** a análise original (descrita no restante deste
  documento) — detecta a **causa** de aumento/redução de um **indicador-chave** a
  partir dos demais indicadores, com **lag em minutos**.

> Esta pasta `version 2` preserva todos os arquivos da versão 1; o modo Clássico é
> idêntico ao app original.

---

## Modo Análise Sazonal

Transforma os dados de processo em **variáveis explicativas por período do alvo** e
busca quais indicadores têm **maior evidência estatística de influência** sobre o
indicador alvo (prováveis causas — sem afirmar causalidade absoluta).

**Entrada:** um único arquivo Excel com **múltiplas abas, classificadas pelo
usuário** antes da análise:

- **Contínuas** (1 ou mais abas): indicadores em períodos curtos (minutos) —
  `DataHora` + indicadores; cada aba pode ter passo temporal próprio;
- **Periódicas** (0 ou mais abas): leituras informadas a cada *x* horas —
  laboratório **ou qualquer variável periódica** (apontamentos, análises manuais,
  médias horárias etc.); o valor no horário `T` representa a **média da janela
  anterior** `(T−x, T]` (o intervalo *x* é detectado pela mediana do espaçamento
  entre leituras);
- **Alvo** (exatamente 1 aba): o indicador a explicar (produção, análise que
  impacta o KPI etc.), divulgado em período de horas.

O sistema **sugere a classificação** por heurística (nome da aba + espaçamento dos
registros) e o usuário confirma/ajusta. Em CSV, sem abas, carregue o alvo num
segundo upload (fluxo de 2 bases preservado).

**O que o sistema faz:**
1. **Agrega minuto → período do alvo** por indicador: média, mediana, mín, máx,
   amplitude, desvio padrão, CV, percentis (10/25/75/90), soma, nº de válidos,
   % de ausentes.
2. **Instabilidade:** Δ médio entre leituras, taxa máxima de variação, nº de
   oscilações bruscas.
3. **Periódicos alinhados ao alvo:** cada leitura é expandida na sua janela
   `(T−x, T]` e agregada por período do alvo (média ponderada pelo tempo, última
   leitura, nº de leituras), com **lags em múltiplos do intervalo entre
   leituras**.
4. **Limites críticos** (mín/máx por indicador, contínuas e periódicas): tempo e
   área fora de faixa como variáveis explicativas **+ análise de excursões** —
   eventos contíguos fora de faixa, linha do tempo, e teste de **Mann-Whitney U**
   comparando o alvo em períodos com × sem violação (aba 🚦).
5. **Lag em minutos** das contínuas: blocos de períodos anteriores até o máximo.
6. **Turnos opcionais:** média/desvio por 00–08, 08–16, 16–24.
7. **Junta com o alvo**, limpa a base (colunas constantes, nulos, padroniza nomes).
8. **Estatística:** Pearson e Spearman (com *p-value*) e *mutual information*.
9. **Modelo:** Random Forest (ou XGBoost) com **split temporal** (sem embaralhar) e
   métricas MAE, RMSE, MAPE, R².
10. **Explicabilidade:** importância do modelo e **SHAP**.
11. **Ranking consolidado** (Spearman + MI + importância + SHAP) e **diagnóstico
    gerencial** automático.
12. **Diagnóstico do período:** alvo real, principais contribuintes e comparação
    com a média histórica.
13. **Relatório final (aba 📄):** anexe qualquer gráfico/tabela das abas com o botão
    **📎 Adicionar ao relatório**, comente e reordene os itens; o texto explica **por
    que** as variáveis foram consideradas principais (correlação, p-value, SHAP,
    lags e limites violados). Prévia **HTML interativa** + download em **HTML** e
    **PDF** (gerado no servidor com kaleido + xhtml2pdf).
14. **Tooltips** explicativos (ícones "?") em todos os conceitos estatísticos/ML,
    centralizados em `tooltips.py`, além do glossário no topo.
15. **Exporta** todos os resultados num Excel multi-aba (incluindo excursões).

**Validação rápida:**

```bash
python sample_data_multiabas.py    # cria sample_multiabas.xlsx (Moenda 1min, Fermentacao 5min,
                                   # Caldo 4h [periodico], Alvo 8h) com relacoes conhecidas
python _test_multiabas.py          # smoke test headless do pipeline completo
python _test_ui_apptest.py         # teste de UI (AppTest): classificacao + processamento
streamlit run app.py               # modo Sazonal: suba o arquivo, confirme a classificacao,
                                   # limite pH minimo 4.8 (Fermentacao), Processar
```

Espera-se confirmar: `Moenda_Vazao_mean_lag_0min` (direto), `Moenda_Temperatura_std_lag_0min`
(inverso), `Brix_per_last_lag_240min` (efeito defasado da leitura periódica anterior)
e `Fermentacao_pH_pct_fora_lag_0min` (Mann-Whitney p < 0,05 na aba 🚦).

---

## Modo Correlação Clássica (v1)

App em Streamlit para engenheiros **detectarem a causa** de aumento ou redução de
um **indicador-chave** a partir de uma série de indicadores de processo —
considerando o **atraso (lag)** entre causa e efeito (um distúrbio numa variável
pode só se refletir no indicador-chave alguns minutos depois).

## Abas

- **🚦 Painel de Decisão (principal):** diagnóstico com KPIs (valor/variação do
  indicador-chave, principal causa, **fatia real de impacto** e variância total
  explicada), medidor de impacto, **gatilhos acionáveis** (alerta quando um driver
  que antecede o indicador-chave está em distúrbio agora, prevendo o efeito e o
  tempo de antecedência), gráfico de **eixo duplo** (chave × driver principal em
  unidades reais), evidência causa→efeito e tabela de drivers com status.
- **🔬 Análise Detalhada:** ranking completo, curva de correlação × lag, séries
  alinhadas pelo lag e dispersão por indicador.

## Atribuição rigorosa de variância (Shapley/LMG)

O painel **não** usa apenas a correlação individual (que superestima quando os
indicadores são correlacionados entre si). Em vez disso:

1. **Alinha** cada indicador pelo seu lag ótimo e monta uma regressão multivariada
   contra o indicador-chave.
2. **Elimina a multicolinearidade** — indicadores redundantes entre si
   (`|corr| ≥ 0,9` ou VIF alto) são detectados; mantém-se um representante por grupo
   (o mais ligado ao indicador-chave) e os demais são removidos (reportados no app).
3. **Decompõe o R²** do modelo pelo método de **Shapley/LMG**: cada driver recebe
   uma **fatia real, não-negativa e não-sobreposta** da variância do indicador-chave.
   As fatias **somam a variância total explicada** pelo modelo.

Assim, "Fatia da variância (%)" é o impacto efetivo de cada driver, sem dupla
contagem; "Contribuição no modelo (%)" é essa fatia normalizada para 100%.

## Formato da planilha

- **1ª coluna:** data e hora na mesma célula (ex.: `2026-01-01 08:05`).
- **Demais colunas:** um indicador por coluna, com o valor alinhado ao instante
  da linha. O cabeçalho de cada coluna é o nome do indicador.
- Formatos aceitos: `.xlsx` ou `.csv` (vírgula decimal e separador `;` do padrão
  brasileiro são tratados automaticamente).

| DataHora         | Temperatura | Vazao | Pressao |
|------------------|-------------|-------|---------|
| 2026-01-01 08:00 | 72.4        | 11.2  | 3.1     |
| 2026-01-01 08:01 | 72.9        | 11.5  | 3.0     |

## Variáveis de laboratório (amostras compostas)

Algumas variáveis são medidas em laboratório por **composição** (ex.: a cada 4 h),
onde o valor registrado é a **média do período** `(T − N horas, T]` (T = data/hora
na planilha = fim do período). Esses valores **não podem ser interpolados**.

No app, ligue **"Há indicadores de laboratório"**, selecione quais colunas são de
laboratório e o **período composto (horas)**. O programa então:
- Ancorado em cada coleta de laboratório (instante T), calcula a **média de todos os
  indicadores** na janela `(T − N horas, T]` — colocando processo e laboratório na
  mesma base composta, sem interpolar.
- As variáveis de laboratório mantêm o valor reportado.
- As demais análises (lag, correlação, atribuição rigorosa, gatilhos) seguem normalmente
  sobre esse dataset composto (passo = N horas).

Sem variáveis de laboratório, o fluxo padrão (reamostragem + interpolação de lacunas
curtas) é usado. Gere um exemplo com `python sample_data_lab.py`.

## Como funciona

1. **Leitura e reamostragem** — o passo temporal é detectado automaticamente
   (mediana dos intervalos) e os dados são reamostrados numa grade regular;
   lacunas curtas são interpoladas. (No modo laboratório, ver seção acima.)
2. **Indicador-chave** — você escolhe qual indicador (o "efeito") quer avaliar.
3. **Detecção de lag** — para cada indicador, varre-se uma janela de defasagens
   e escolhe-se o **lag que maximiza a correlação** (correlação cruzada / TLCC).
   - **Lag positivo** = o indicador **antecede** o indicador-chave (causa→efeito).
4. **Métricas** — Correlação de **Pearson** (linear) e **Informação Mútua**
   (relações não-lineares), ambas no lag ótimo, além da correlação em lag 0.

## Saídas

- Ranking dos indicadores por correlação no lag ótimo (tabela + barras).
- Curva de correlação × lag (mostra visualmente o atraso de cada indicador).
- Séries normalizadas sobrepostas, alinhadas pelo lag ótimo.
- Dispersão no lag ótimo.
- Download do ranking em Excel.

## Instalação e execução

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validação rápida

Gere uma planilha sintética com lags conhecidos e confira se a análise os recupera:

```bash
python sample_data.py            # cria sample_data.xlsx (Vazao: lag 5 min, Pressao: lag 12 min)
streamlit run app.py             # suba sample_data.xlsx, escolha "Temperatura"
```

Espera-se que `Vazao` e `Pressao` liderem o ranking com lags ótimos próximos de
5 e 12 minutos, e que a correlação no lag ótimo seja maior que em lag 0.

## Arquivos

- `analysis.py` — núcleo de cálculo (v1 + funções da Versão 2: variáveis diárias, lag em dias, estatística, modelo, SHAP, ranking consolidado, diagnóstico). Sem dependência de UI.
- `app.py` — interface Streamlit (seletor de modo + fluxo Clássico).
- `daily_ui.py` — interface do modo **Produção Diária**.
- `sample_data.py` — gerador de dados sintéticos (modo Clássico).
- `sample_data_lab.py` — gerador com variáveis de laboratório.
- `sample_data_producao.py` — gerador do Excel 2-abas (processo + produção) para o modo Produção Diária.
