# Análise de Correlação Multivariada

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

- `analysis.py` — núcleo de cálculo (leitura, reamostragem, lag, correlação, MI). Sem dependência de UI.
- `app.py` — interface Streamlit.
- `sample_data.py` — gerador de dados sintéticos para validação.

## Guia Interativo de Data Wrangling

App Streamlit independente que apresenta um guia interativo de Data Wrangling
com pandas (menu com barra de progresso, abas, quizzes, botões de copiar código
e realce de sintaxe). O conteúdo interativo vive em `guia_data_wrangling.html`
e é renderizado por `guia_data_wrangling.py`.

```bash
streamlit run guia_data_wrangling.py
```

No Streamlit Community Cloud, aponte o "Main file path" do deploy para
`guia_data_wrangling.py`. A única dependência é `streamlit` (já em
`requirements.txt`).
