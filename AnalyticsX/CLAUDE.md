# Argus AnalyticsX — Contexto do projeto (handoff)

Este arquivo dá o contexto para **qualquer nova sessão do Claude Code** (inclusive
pelo celular em **claude.ai/code**) continuar o trabalho sem perder o fio. O Claude
Code carrega este arquivo automaticamente ao abrir a pasta.

## O que é
App **Streamlit** para identificar **causas de variação de KPIs** em usina de açúcar
e álcool (perdas, desqualificação, resultados analíticos). Pasta ativa: **`AnalyticsX/`**.
Deploy no **Streamlit Cloud** apontando para `AnalyticsX/app.py` (Python 3.12).

## Dois modos (seletor no topo de `app.py`)
- **📈 Análise Sazonal** (principal — `daily_ui.py`): planilha Excel **multi-abas
  classificadas pelo usuário** — Contínuas (minutos), **Periódicas** (a cada x horas),
  Alvo. Agrega o processo na janela de cada período do alvo, considera **lag em minutos
  (só das contínuas)**, **limites críticos + excursões** (Mann-Whitney), estatística
  (Pearson/Spearman/MI) + **Random Forest/XGBoost** + **SHAP**, ranking consolidado,
  diagnóstico e **relatório HTML/PDF**.
- **📊 Correlação Clássica (v1)**: análise original minuto a minuto. **NÃO alterar.**

## Decisões/convenções já firmadas (manter)
- **Terminologia "Periódico"** (não "laboratório") — cobre qualquer leitura periódica,
  não só laboratório. Sufixo interno de coluna: **`_per_`** (ex.: `Brix_per_last_lag_0min`).
- **Periódicas NÃO recebem defasagem (lag)**: entram só no período atual (lag 0). O
  filtro de lag em minutos vale **apenas para as contínuas**.
- **Relatório/diagnóstico em linguagem clara**: `humanize_variable()` converte nomes
  técnicos em frases (ex.: "instabilidade da Temperatura (Moenda), sem lag"); sem código
  bruto no texto. "**gravidade**" no lugar de "severidade".
- **Aba 📖 Legenda**: usa `st.table` (não `st.dataframe`) para não cortar o texto.
- **Identidade visual**: ícone = **olho verde** (`assets/eye_icon.png`, gerado por
  `assets/generate_icon.py`); cabeçalho/login via `branding.py`; **fonte padrão** (sem serifa).
- **Login**: `auth.py` + `st.secrets` → `[auth]` com `salt` e `[auth.users]` `usuario="hash"`.
  Senhas em hash (HMAC-SHA256). `gen_password_hash.py` gera os hashes.
  **`.streamlit/secrets.toml` é gitignored — nunca versionar credenciais.**

## Arquivos principais (em `AnalyticsX/`)
`analysis.py` (núcleo: v1 + Seção B sazonal), `daily_ui.py` (UI sazonal), `app.py`
(entrada + login), `report.py` (HTML/PDF, fuso de Brasília), `tooltips.py`, `branding.py`,
`auth.py`, `sample_data_multiabas.py` (gera `sample_multiabas.xlsx`: abas Moenda/Fermentacao/
Caldo[periódico]/Alvo).
Testes: `_test_multiabas.py`, `_test_humanize.py`, `_test_legend.py`, `_test_ui_apptest.py`,
`_test_ui_retrocompat.py`, `_test_auth.py`, `_test_header.py`, `_test_report*.py`.

## Como rodar / testar (Windows)
- App: `streamlit run app.py` (defina `PYTHONIOENCODING=utf-8`).
- Testes: `python _test_<nome>.py` — todos imprimem `TUDO OK` quando passam.
- **Deploy**: `requirements.txt` e `packages.txt` ficam na **RAIZ do repo** (o Streamlit
  Cloud lê a raiz). requirements: `plotly>=6.1`, `kaleido>=1.0`, `xhtml2pdf`, `xgboost`,
  `shap`; packages: `chromium`, `libcairo2-dev`, `pkg-config` (para o PDF).

## Pendências / próximos passos
- **Configurar os Secrets de PRODUÇÃO** no Streamlit Cloud (gerar `salt`+`hash` próprios,
  diferentes dos de teste). Sem isso, o app mostra "acesso não configurado".
- (Local, opcional) renomear a pasta `prophet_past` → `Argus` com
  `_renomear_para_Argus.ps1` (fechar editores/terminais antes).

## Entregáveis fora do app (na máquina local, em Downloads)
- `Argus_AnalyticsX_Apresentacao.pptx` — deck técnico para engenheiros.
- `Argus_AnalyticsX_Pitch_Vendas.pptx` — pitch de vendas para gestor.
