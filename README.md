# Análise de Correlação Multivariada

App em Streamlit para engenheiros **detectarem a causa** de aumento ou redução de
um **indicador-chave** a partir de uma série de indicadores de processo —
considerando o **atraso (lag)** entre causa e efeito (um distúrbio numa variável
pode só se refletir no indicador-chave alguns minutos depois).

## Abas

- **🚦 Painel de Decisão (principal):** diagnóstico com KPIs (valor atual e
  variação do indicador-chave, principal causa, **% de impacto** = variância
  explicada, antecedência/lag), medidor de impacto, **gatilhos acionáveis**
  (alerta quando um driver que antecede o indicador-chave está em distúrbio
  agora, prevendo o efeito e o tempo de antecedência), evidência causa→efeito e
  tabela de drivers com status atual.
- **🔬 Análise Detalhada:** ranking completo, curva de correlação × lag, séries
  alinhadas pelo lag e dispersão por indicador.

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

## Como funciona

1. **Leitura e reamostragem** — o passo temporal é detectado automaticamente
   (mediana dos intervalos) e os dados são reamostrados numa grade regular;
   lacunas curtas são interpoladas.
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
