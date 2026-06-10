# 👁️ Argus Analytics

**Argus Analytics** é um analisador estatístico de dados, moderno e visual,
inspirado em ferramentas como o Minitab — porém mais intuitivo e amigável.
Voltado a **engenheiros, técnicos e analistas** com conhecimento iniciante ou
intermediário em estatística, ele funciona como um **assistente estatístico
guiado**: você importa uma planilha, escolhe visualmente a análise desejada,
recebe gráficos interativos, interpretações automáticas e monta um relatório
final.

O nome faz referência a **Argus**, o gigante mitológico de muitos olhos: a ideia
é "enxergar" os dados por **diversos ângulos** — padrões, tendências,
correlações, variações e oportunidades de melhoria.

---

## ✨ Principais recursos

- Importação de **Excel** (`.xlsx`, `.xls`) e **CSV** (com suporte a decimal BR).
- Detecção automática de colunas **numéricas, categóricas, data/hora e texto**.
- Menu visual de análises em **cards**, com descrições didáticas.
- Gráficos **interativos** (Plotly) e tabelas formatadas.
- **Interpretações automáticas** em linguagem simples.
- Montagem de um **relatório final** (HTML interativo + PDF opcional).

---

## 📥 Instalação

Requisitos: **Python 3.10+**.

```bash
# (opcional) crie um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# instale as dependências
pip install -r requirements.txt
```

---

## ▶️ Como rodar

A partir da pasta `argus_analytics/`:

```bash
streamlit run app.py
```

O aplicativo abrirá no navegador (por padrão em `http://localhost:8501`).

---

## 📂 Como importar dados

1. Clique em **📋 Visão Geral dos Dados** (card ou menu lateral).
2. Faça o upload do arquivo `.xlsx`, `.xls` ou `.csv`.
3. Se for Excel com várias abas, **selecione a aba**.
4. Clique em **📥 Carregar dados**.
5. Confira a **prévia**, os **tipos de coluna** e o **resumo** da base.

> 💡 O Argus trata automaticamente o formato numérico brasileiro
> (ex.: `1.234,56`) e tenta reconhecer colunas de data/hora.

---

## 🧰 Análises disponíveis

| # | Análise | Para quê serve |
|---|---------|----------------|
| 1 | **Visão Geral dos Dados** | Importar, validar e diagnosticar a base |
| 2 | **Qualidade dos Dados** | Ausentes, duplicidades, colunas constantes, outliers |
| 3 | **Estatística Descritiva** | Média, mediana, dispersão, quartis, CV |
| 4 | **Distribuição dos Dados** | Histograma, boxplot, densidade, assimetria |
| 5 | **Teste de Normalidade** | Shapiro-Wilk e Anderson-Darling |
| 6 | **Correlação** | Pearson / Spearman / Kendall, heatmap, ranking |
| 7 | **Regressão** | Linear simples e múltipla, R², resíduos |
| 8 | **Comparação entre Grupos** | Teste t, Mann-Whitney, ANOVA, Kruskal-Wallis |
| 9 | **Controle Estatístico de Processo** | Carta de controle (individuais) |
| 10 | **Capabilidade Cp/Cpk** | Índices de capabilidade vs. especificação |
| 11 | **Análise Temporal** | Tendência, agregação, média móvel |
| 12 | **Análise com Lag** | Defasagem causa→efeito entre variáveis |
| 13 | **Outliers** | IQR, Z-score e MAD |
| 14 | **Relatório Final** | Montagem e download do relatório |

---

## 📑 Como gerar o relatório

1. Em cada análise, clique em **➕ Adicionar ao relatório**.
2. Abra **📑 Relatório Final**.
3. **Reordene** (⬆️/⬇️) ou **remova** (🗑️) análises.
4. Baixe em **HTML** (interativo, abre em qualquer navegador).
5. Se o ambiente tiver `reportlab` + `kaleido`, gere também o **PDF**.

> O PDF é opcional: se as dependências não estiverem disponíveis (ex.: alguns
> ambientes de nuvem), o botão é ocultado e o HTML continua funcionando
> normalmente.

---

## ☁️ Deploy no Streamlit Community Cloud

1. Suba a pasta `argus_analytics/` para um repositório no GitHub.
2. No Streamlit Cloud, aponte o app para `app.py`.
3. O `requirements.txt` já está pronto para o deploy.

> Observação: a geração de **PDF** depende do `kaleido`, que pode não estar
> disponível em todos os ambientes de nuvem. O relatório **HTML** funciona
> sempre.

---

## 🗂️ Estrutura do projeto

```
argus_analytics/
├── app.py                 # roteador principal + tela inicial (cards)
├── modules/               # uma tela de análise por arquivo (render(state))
├── utils/                 # helpers, plotting, validation, interpretation
├── assets/                # recursos visuais
├── reports/               # saída de relatórios
├── requirements.txt
└── README.md
```

---

## ⚖️ Aviso estatístico

O Argus apresenta interpretações automáticas para apoiar a análise, mas **não
substitui o julgamento técnico**. Em especial, lembre-se: *correlação não
implica causalidade*.

Exemplo de comando para rodar:

```bash
streamlit run app.py
```
