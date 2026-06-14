# Manter o app Streamlit acordado

O Streamlit Community Cloud coloca apps sem acesso para "dormir". Este
repositório tem um **GitHub Action** que visita o app a cada ~10 minutos com um
navegador real e, se ele estiver dormindo, clica no botão **"Yes, get this app
back up!"** para reativá-lo.

Arquivos:
- `.github/workflows/keep-awake.yml` — agendamento (cron) do GitHub Actions.
- `.github/scripts/keep_awake.py` — abre a URL e acorda o app (Playwright/Chromium).

## Passo único de configuração (obrigatório)

Informe a URL do seu app numa **variável do repositório**:

1. No GitHub, vá em **Settings → Secrets and variables → Actions**.
2. Abra a aba **Variables** e clique em **New repository variable**.
3. Preencha:
   - **Name:** `STREAMLIT_APP_URL`
   - **Value:** a URL pública do app, por exemplo `https://seu-app.streamlit.app`
4. Salve.

Pronto. O fluxo roda sozinho. Para testar na hora: aba **Actions → "Manter app
Streamlit acordado" → Run workflow**. Cada execução guarda um screenshot
(`status-screenshot`) por 3 dias, útil para conferir se o app está no ar.

## Observações

- O repositório é **público**, então os minutos de Actions são gratuitos.
- O GitHub **desativa** workflows agendados após **60 dias sem commits** no repo.
  Se isso acontecer, basta reativá-lo na aba Actions (ou fazer qualquer commit).
- Alternativas externas, caso prefira não usar o Actions: serviços de uptime
  como **UptimeRobot** ou **cron-job.org**, apontando para a URL do app
  (intervalo de 5–15 min). São independentes do GitHub.
- O horário do cron é em **UTC**.
