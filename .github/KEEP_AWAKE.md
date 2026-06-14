# Manter o app Streamlit acordado

O Streamlit Community Cloud coloca apps sem acesso para "dormir". Este
repositório tem um **GitHub Action** que visita o app a cada ~10 minutos com um
navegador real e, se ele estiver dormindo, clica no botão **"Yes, get this app
back up!"** para reativá-lo.

Arquivos:
- `.github/workflows/keep-awake.yml` — agendamento (cron) do GitHub Actions.
- `.github/scripts/keep_awake.py` — abre a URL e acorda o app (Playwright/Chromium).

## Configuração

A URL do app **já está fixada** no workflow:
`https://argus-analytics.streamlit.app/`. Não é preciso fazer mais nada — o
fluxo roda sozinho.

Para testar na hora: aba **Actions → "Manter app Streamlit acordado" → Run
workflow**. Cada execução guarda um screenshot (`status-screenshot`) por 3 dias,
útil para conferir se o app está no ar.

### Trocar a URL depois (opcional)

Sem editar o arquivo, crie uma **variável de repositório** que tem prioridade
sobre o valor fixo:

1. **Settings → Secrets and variables → Actions → aba "Variables"**.
2. **New repository variable** → **Name:** `STREAMLIT_APP_URL` →
   **Value:** a nova URL.

## Observações

- O repositório é **público**, então os minutos de Actions são gratuitos.
- O GitHub **desativa** workflows agendados após **60 dias sem commits** no repo.
  Se isso acontecer, basta reativá-lo na aba Actions (ou fazer qualquer commit).
- Alternativas externas, caso prefira não usar o Actions: serviços de uptime
  como **UptimeRobot** ou **cron-job.org**, apontando para a URL do app
  (intervalo de 5–15 min). São independentes do GitHub.
- O horário do cron é em **UTC**.
