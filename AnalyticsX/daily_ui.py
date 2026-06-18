"""
Modo "Análise Sazonal" — Versão 2
---------------------------------
Correlaciona o comportamento dos indicadores de processo (minuto a minuto) com um
**indicador sazonal** informado periodicamente (cadência diária OU sub-diária, ex.:
a cada poucas horas). O sistema agrega o processo na janela de cada período (definido
pelos horários do indicador sazonal), considera efeitos de **lag em minutos** e aponta
os indicadores com maior EVIDÊNCIA ESTATÍSTICA de influência (prováveis causas — sem
causalidade absoluta).

Exposto por ``render_seasonal_mode()``, chamado pelo seletor de modo em ``app.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components

import analysis
import report
from tooltips import HELP, glossario_markdown


# --------------------------------------------------------------------------- #
# Helpers de gráfico
# --------------------------------------------------------------------------- #
def _bar_rank(series: pd.Series, title: str, xlabel: str, top_n: int = 20,
              color: str = "Blues") -> go.Figure:
    """Barra horizontal de ranking a partir de uma Series (índice = variável)."""
    s = series.dropna().head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=s.values, y=s.index, orientation="h",
            text=[f"{v:.3f}" for v in s.values], textposition="auto",
            marker_color=s.values, marker_colorscale=color,
        )
    )
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title="Variável",
        height=max(320, 26 * len(s)), margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def _fig_scatter(x: pd.Series, y: pd.Series, xname: str, yname: str) -> go.Figure:
    pair = pd.concat([x, y], axis=1).dropna()
    pair.columns = [xname, yname]
    fig = go.Figure(go.Scatter(x=pair[xname], y=pair[yname], mode="markers", opacity=0.7))
    r = pair[xname].corr(pair[yname]) if len(pair) > 2 else float("nan")
    fig.update_layout(
        title=f"{xname} × {yname} (r = {r:+.3f})",
        xaxis_title=xname, yaxis_title=yname, height=420,
    )
    return fig


def _fig_series(y: pd.Series, title: str, yname: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name=yname))
    fig.update_layout(title=title, xaxis_title="Período", yaxis_title=yname, height=360)
    return fig


def _fig_real_vs_pred(y_all: pd.Series, pred_train: pd.Series,
                      pred_test: pd.Series, yname: str = "Indicador sazonal") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_all.index, y=y_all.values, mode="lines+markers",
                             name=f"{yname} (real)", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=pred_train.index, y=pred_train.values, mode="lines",
                             name="Previsto (treino)", line=dict(color="#2ca02c", dash="dot")))
    fig.add_trace(go.Scatter(x=pred_test.index, y=pred_test.values, mode="lines+markers",
                             name="Previsto (teste)", line=dict(color="#d62728")))
    if len(pred_test):
        fig.add_vline(x=pred_test.index[0], line_dash="dash", line_color="gray",
                      annotation_text="início do teste", annotation_position="top")
    fig.update_layout(title=f"{yname}: real × previsto", xaxis_title="Período",
                      yaxis_title=yname, height=420,
                      legend=dict(orientation="h", y=1.1))
    return fig


def _fig_corr_matrix(df: pd.DataFrame) -> go.Figure:
    corr = df.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True, text=corr.values,
        texttemplate="%{text:.2f}", textfont=dict(size=9), colorbar=dict(title="r"),
    ))
    n = len(corr.columns)
    fig.update_layout(title="Matriz de correlação (variáveis selecionadas + indicador sazonal)",
                      height=max(400, 32 * n), yaxis=dict(autorange="reversed"),
                      margin=dict(t=50, b=40))
    return fig


def _fmt_periodo(minutes: float) -> str:
    if minutes >= 60:
        return f"{minutes:g} min (~{minutes / 60:g} h)"
    return f"{minutes:g} min"


def _attach_button(key: str, kind: str, title: str, payload, source_tab: str) -> None:
    """Botão 📎 que adiciona um gráfico/tabela/texto ao relatório final."""
    attached = report.has_item(st.session_state, key)
    label = "✅ No relatório" if attached else "📎 Adicionar ao relatório"
    if st.button(label, key=f"rep_btn_{key}", disabled=attached,
                 help=HELP["anexar_relatorio"]):
        report.add_item(st.session_state, report.ReportItem(
            id=key, kind=kind, title=title, payload=payload, source_tab=source_tab))
        st.rerun()


def _repr_col(X, key: str, sheet_names) -> str:
    """Coluna que representa o indicador raiz (valor médio/nível, sem variações
    estatísticas). Preferência: média; senão a de maior variância; senão a 1ª."""
    cols = [c for c in X.columns if analysis.base_indicator(c, sheet_names)[0] == key]
    for cand in (f"{key}_mean", f"{key}_per_mean"):
        if cand in cols:
            return cand
    if not cols:
        return ""
    try:
        return X[cols].var(numeric_only=True).idxmax()
    except Exception:  # noqa: BLE001
        return cols[0]


def _cached_drilldown(X, target_col, sheet_names, model_type, run_shap):
    """drilldown_ranking com cache por coluna-alvo (não recomputa a cada rerun)."""
    cache = st.session_state.setdefault("v2_chain_cache", {})
    if target_col not in cache:
        cache[target_col] = analysis.drilldown_ranking(
            X, target_col, sheet_names, model_type=model_type, run_shap=run_shap)
    return cache[target_col]


# --------------------------------------------------------------------------- #
# Fluxo principal
# --------------------------------------------------------------------------- #
def render_seasonal_mode() -> None:
    st.header("📈 Análise Sazonal — evidência estatística de influência")
    st.caption(
        "Carregue **uma planilha Excel com abas** de variáveis **contínuas** (minuto "
        "a minuto), **periódicas** (informadas a cada x horas — laboratório ou "
        "qualquer leitura periódica) e do **indicador alvo** — e classifique cada "
        "aba. O sistema agrega tudo na janela de cada período do alvo, considera "
        "efeitos de **lag em minutos** (inclusive dos indicadores periódicos), "
        "avalia **limites críticos** e aponta os indicadores com maior evidência "
        "estatística de influência."
    )

    with st.expander("ℹ️ Glossário — entenda cada modelo e análise"):
        st.markdown(glossario_markdown())

    (tab_cfg, tab_base, tab_legend, tab_stat, tab_limits, tab_model, tab_rank,
     tab_day, tab_report, tab_export) = st.tabs([
        "📥 Dados & Configuração", "🧱 Base por Período", "📖 Legenda",
        "📊 Estatística", "🚦 Limites & Excursões", "🤖 Modelo & Explicabilidade",
        "🏆 Ranking Consolidado", "📅 Diagnóstico do Período",
        "📄 Relatório", "⬇️ Exportar",
    ])

    # ===================================================================== #
    # ABA — Legenda das variáveis (estática: disponível antes de processar)
    # ===================================================================== #
    with tab_legend:
        st.subheader("📖 Legenda dos nomes de variáveis",
                     help="Cada coluna da base por período é gerada automaticamente "
                          "a partir dos indicadores brutos. Esta aba explica como "
                          "ler o nome e o que cada sufixo significa.")
        st.markdown(
            "Os nomes seguem o padrão **`[Aba_]Indicador_sufixo[_lag_Xmin]`**:\n\n"
            "- **Aba** — nome da aba de origem (aparece só quando há mais de uma "
            "aba de variáveis contínuas, ex.: `Moenda_`);\n"
            "- **Indicador** — o nome da coluna na sua planilha (ex.: `Temperatura`);\n"
            "- **Sufixo** — a estatística calculada na janela do período (tabela abaixo);\n"
            "- **`_lag_Xmin`** — defasagem, **só aparece quando há atraso**: "
            "`_lag_480min` é o valor da variável **480 minutos antes** (efeito "
            "defasado). **Sem esse sufixo = período atual** (sem defasagem). O "
            "lag vale apenas para as variáveis contínuas.\n\n"
            "**Exemplo:** `Moenda_Temperatura_std` = *instabilidade da Temperatura "
            "(aba Moenda), no período atual* — mede o quanto o sinal oscila; se "
            "aparece no topo do ranking com correlação negativa, a instabilidade "
            "da temperatura está derrubando o alvo."
        )
        legend_df = analysis.suffix_legend_table()
        for cat in legend_df["Categoria"].unique():
            sub = (legend_df[legend_df["Categoria"] == cat]
                   .drop(columns="Categoria").reset_index(drop=True))
            st.markdown(f"**{cat}**")
            # st.table (em vez de st.dataframe) quebra o texto em várias linhas
            # e não corta as descrições longas com reticências.
            st.table(sub.style.hide(axis="index"))
        _attach_button("legenda_sufixos", "table", "Legenda dos sufixos das variáveis",
                       legend_df, "Legenda")

    # ===================================================================== #
    # ABA 1 — Dados & Configuração
    # ===================================================================== #
    with tab_cfg:
        uploaded = st.file_uploader(
            "Arquivo único (Excel com abas) ou CSV de processo",
            type=["xlsx", "xls", "csv"], key="v2_file",
            help="Excel: cada aba contém variáveis contínuas (minutos), "
                 "periódicas (a cada x horas) ou o indicador alvo — você "
                 "classifica cada aba abaixo. CSV: carregue o processo aqui e o "
                 "alvo em outro CSV.",
        )
        if uploaded is None:
            st.info(
                "Aguardando o arquivo. Em Excel, use **abas distintas** para "
                "variáveis contínuas (DataHora + indicadores), variáveis "
                "periódicas e o indicador alvo — você informa **qual aba é "
                "qual** logo abaixo. Em CSV, carregue o processo aqui e o "
                "indicador alvo logo abaixo."
            )
            return

        try:
            uploaded.seek(0)
            sheet_names = analysis.list_sheets(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erro ao abrir o arquivo: {exc}")
            return

        # rótulos PT-BR <-> papéis internos
        ROLE_PT = {analysis.ROLE_CONTINUA: "Contínua",
                   analysis.ROLE_PERIODICO: "Periódico",
                   analysis.ROLE_ALVO: "Alvo",
                   analysis.ROLE_IGNORAR: "Ignorar"}
        PT_ROLE = {v: k for k, v in ROLE_PT.items()}

        cont_sheets: list[dict] = []   # {"name", "df", "dt_col"}
        lab_sheets: list[dict] = []    # periódicas: {"name","df","dt_col","value_cols","period_min"}
        prod_raw = None
        prod_sheet_name = "Alvo"

        if sheet_names:
            # --- Leitura única de todas as abas (cache por arquivo) ------ #
            cache_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("v2_sheets_key") != cache_key:
                try:
                    uploaded.seek(0)
                    st.session_state["v2_sheets_raw"] = pd.read_excel(
                        uploaded, sheet_name=None, engine="openpyxl")
                    st.session_state["v2_sheets_key"] = cache_key
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro ao ler as abas: {exc}")
                    return
            raw_by_sheet: dict[str, pd.DataFrame] = st.session_state["v2_sheets_raw"]

            # --- Classificação das abas ---------------------------------- #
            st.subheader("🗂️ Classificação das abas",
                         help="Informe o que cada aba contém antes de iniciar a "
                              "análise. O tipo sugerido vem de uma heurística "
                              "(nome da aba e espaçamento entre registros) — confira.")
            class_template = pd.DataFrame({
                "Aba": sheet_names,
                "Tipo": [ROLE_PT[analysis.guess_sheet_role(n, raw_by_sheet[n])]
                         for n in sheet_names],
            })
            roles_edit = st.data_editor(
                class_template, hide_index=True, use_container_width=True,
                disabled=["Aba"], key=f"v2_roles_{cache_key}",
                column_config={
                    "Tipo": st.column_config.SelectboxColumn(
                        "Tipo", options=list(PT_ROLE), required=True,
                        help="Contínua: períodos curtos (minutos). Periódico: "
                             "leituras a cada x horas (laboratório ou outras) — o "
                             "valor informado em T representa a média da janela "
                             "anterior. Alvo: o indicador a explicar (1 aba). "
                             "Ignorar: não usar.",
                    ),
                },
            )
            roles = {r["Aba"]: PT_ROLE[r["Tipo"]] for _, r in roles_edit.iterrows()}

            alvo_names = [n for n, r in roles.items() if r == analysis.ROLE_ALVO]
            cont_names = [n for n, r in roles.items() if r == analysis.ROLE_CONTINUA]
            lab_names = [n for n, r in roles.items() if r == analysis.ROLE_PERIODICO]
            if len(alvo_names) != 1:
                st.warning("Classifique **exatamente 1 aba** como Alvo para continuar.")
                return
            if not cont_names and not lab_names:
                st.warning("Classifique ao menos **1 aba** como Contínua ou Periódico.")
                return
            prod_sheet_name = alvo_names[0]
            prod_raw = raw_by_sheet[prod_sheet_name]

            # --- Configuração por aba ------------------------------------ #
            st.subheader("⚙️ Colunas e parâmetros")
            for name in cont_names:
                df_sheet = raw_by_sheet[name]
                with st.expander(f"📈 Contínua — **{name}**", expanded=False):
                    cols = list(df_sheet.columns)
                    guess = analysis.guess_datetime_column(df_sheet)
                    dt_col_s = st.selectbox(
                        "Coluna de data/hora", cols,
                        index=cols.index(guess) if guess in cols else 0,
                        key=f"v2_dt_{name}",
                    )
                    st.dataframe(df_sheet.head(5), use_container_width=True)
                    cont_sheets.append({"name": name, "df": df_sheet, "dt_col": dt_col_s})

            for name in lab_names:
                df_sheet = raw_by_sheet[name]
                with st.expander(f"🔁 Periódico — **{name}**", expanded=False):
                    cols = list(df_sheet.columns)
                    guess = analysis.guess_datetime_column(df_sheet)
                    dt_col_s = st.selectbox(
                        "Coluna de data/hora da leitura", cols,
                        index=cols.index(guess) if guess in cols else 0,
                        key=f"v2_labdt_{name}", help=HELP["lab_janela"],
                    )
                    value_opts = [c for c in cols if c != dt_col_s]
                    value_cols = st.multiselect(
                        "Colunas de indicadores (valores)", value_opts,
                        default=value_opts, key=f"v2_labvals_{name}",
                    )
                    lab_period_det = None
                    try:
                        _lv, lab_period_det = analysis.parse_lab_sheet(
                            df_sheet, dt_col_s, value_cols or None)
                        st.metric("Intervalo entre leituras (detectado)",
                                  _fmt_periodo(lab_period_det), help=HELP["lab_janela"])
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Não foi possível ler esta aba ainda: {exc}")
                    st.dataframe(df_sheet.head(5), use_container_width=True)
                    if value_cols:
                        lab_sheets.append({
                            "name": name, "df": df_sheet, "dt_col": dt_col_s,
                            "value_cols": value_cols, "period_min": lab_period_det,
                        })

            ac1, ac2 = st.columns(2)
            with ac1:
                prod_date_col = st.selectbox(
                    f"Coluna de data/hora do alvo ({prod_sheet_name})",
                    list(prod_raw.columns), key="v2_prod_date",
                )
            with ac2:
                prod_num_cols = [c for c in prod_raw.columns if c != prod_date_col]
                target_col = st.selectbox(
                    "Coluna do indicador alvo",
                    prod_num_cols or list(prod_raw.columns), key="v2_target",
                )
        else:
            # --- Fluxo CSV (compatibilidade): processo + alvo ------------ #
            uploaded.seek(0)
            proc_raw = analysis.read_sheet(uploaded)
            st.info("CSV não possui abas — carregue a base do **indicador alvo** (CSV) abaixo.")
            prod_file = st.file_uploader("Base do indicador alvo (CSV)",
                                         type=["csv"], key="v2_prod_file")
            if prod_file is None:
                return
            prod_file.seek(0)
            prod_raw = analysis.read_sheet(prod_file)

            st.subheader("⚙️ Colunas e parâmetros")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                dt_col = st.selectbox("Coluna de data/hora (processo)",
                                      list(proc_raw.columns), key="v2_dt_col")
            with cc2:
                prod_date_col = st.selectbox("Coluna de data/hora (indicador alvo)",
                                             list(prod_raw.columns), key="v2_prod_date")
            with cc3:
                prod_num_cols = [c for c in prod_raw.columns if c != prod_date_col]
                target_col = st.selectbox("Coluna do indicador alvo",
                                          prod_num_cols or list(prod_raw.columns),
                                          key="v2_target")
            cont_sheets.append({"name": "Processo", "df": proc_raw, "dt_col": dt_col})

        # --- Detecção do período do alvo --------------------------------- #
        period_detected = None
        try:
            _ts, period_detected = analysis.parse_target_series(
                prod_raw, prod_date_col, target_col)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Não foi possível ler o indicador alvo ainda: {exc}")

        pc0, pc1 = st.columns([1, 1])
        with pc0:
            if period_detected:
                st.metric("Período do alvo detectado", _fmt_periodo(period_detected),
                          help=HELP["periodo_sazonal"])
            else:
                st.metric("Período do alvo detectado", "—", help=HELP["periodo_sazonal"])
        with pc1:
            period_min = st.number_input(
                "Período de agregação (minutos)", min_value=1.0,
                value=float(round(period_detected)) if period_detected else 1440.0,
                step=float(max(1, int(period_detected or 60))),
                help=HELP["periodo_sazonal"], key="v2_period",
            )

        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            default_lag = int(period_min * 3)
            max_lag_min = st.number_input(
                "Máx. de lag (minutos)", min_value=0, value=default_lag,
                step=int(max(1, period_min)), help=HELP["lag"], key="v2_maxlag",
            )
        with cp2:
            tipo = st.selectbox(
                "Tipo de análise",
                ["Completa (com SHAP)", "Estatística + Modelo", "Só estatística"],
                help=HELP["tipo_analise"], key="v2_tipo",
            )
        with cp3:
            model_options = ["RandomForest"] + (["XGBoost"] if analysis._HAS_XGB else [])
            model_type = st.radio("Modelo preditivo", model_options,
                                  horizontal=True, help=HELP["modelo"], key="v2_model")

        n_blocks = int(np.floor(max_lag_min / period_min)) if period_min else 0
        caption = (
            f"O lag se aplica às variáveis **contínuas**: serão criados "
            f"**{n_blocks + 1} bloco(s)** de defasagem (lag 0 + {n_blocks} "
            f"período(s) anterior(es)), rotulados em minutos."
        )
        if lab_sheets:
            caption += (" As variáveis **periódicas** entram apenas no **período "
                        "atual** (sem defasagem).")
        st.caption(caption)
        use_turnos = st.toggle(
            "Criar variáveis por turno (00–08, 08–16, 16–24)", value=False,
            help="Útil quando o período é diário; pouco relevante para períodos curtos.",
            key="v2_turnos",
        )

        # --- Limites críticos por indicador (todas as abas) -------------- #
        limit_rows: list[dict] = []
        for sh in cont_sheets:
            for c in sh["df"].columns:
                if c != sh["dt_col"]:
                    limit_rows.append({"Origem": sh["name"], "Indicador": c})
        for sh in lab_sheets:
            for c in sh["value_cols"]:
                limit_rows.append({"Origem": sh["name"], "Indicador": c})

        with st.expander("🎯 Limites críticos por indicador — opcional"):
            st.caption(
                "Preencha Mínimo e/ou Máximo apenas dos indicadores que deseja "
                "monitorar. Além das variáveis de tempo/área fora de faixa, o "
                "sistema gera a análise de **excursões** (aba 🚦 Limites & "
                "Excursões).", help=HELP["limites"],
            )
            limits_template = pd.DataFrame(limit_rows)
            limits_template["Mínimo"] = np.nan
            limits_template["Máximo"] = np.nan
            comp_key = "|".join(f"{r['Origem']}.{r['Indicador']}" for r in limit_rows)
            edited = st.data_editor(
                limits_template, hide_index=True, use_container_width=True,
                key=f"v2_limits_{abs(hash(comp_key))}",
                disabled=["Origem", "Indicador"],
            )

        limits_by_sheet: dict[str, dict] = {}
        for _, r in edited.iterrows():
            lo, hi = r["Mínimo"], r["Máximo"]
            if pd.notna(lo) or pd.notna(hi):
                limits_by_sheet.setdefault(r["Origem"], {})[r["Indicador"]] = (
                    float(lo) if pd.notna(lo) else None,
                    float(hi) if pd.notna(hi) else None,
                )

        run_model = tipo in ("Completa (com SHAP)", "Estatística + Modelo")
        run_shap = tipo == "Completa (com SHAP)"

        if st.button("🚀 Processar", type="primary", key="v2_run"):
            try:
                with st.spinner("Construindo a base por período e analisando..."):
                    target_series, period_det = analysis.parse_target_series(
                        prod_raw, prod_date_col, target_col)

                    # 1) Contínuas: agrega por janela do alvo + lags em minutos
                    cont_inputs = [
                        {"name": sh["name"], "df": sh["df"], "dt_col": sh["dt_col"],
                         "limits": limits_by_sheet.get(sh["name"], {})}
                        for sh in cont_sheets
                    ]
                    period_feat, steps_by_sheet = analysis.build_period_features_multi(
                        cont_inputs, target_series.index, float(period_min),
                        use_turnos=use_turnos,
                    ) if cont_inputs else (pd.DataFrame(index=pd.DatetimeIndex(
                        target_series.index, name="periodo")), {})
                    lagged = analysis.add_lag_features_min(
                        period_feat, float(max_lag_min), float(period_min)
                    ) if not period_feat.empty else period_feat

                    # 2) Periódicos: alinha às janelas do alvo, SEM defasagem
                    #    (apenas o período atual — não se aplica filtro de lag).
                    lab_period_by_sheet: dict[str, float] = {}
                    lab_frames: list[pd.DataFrame] = []
                    prefix_lab = len(lab_sheets) > 1
                    for sh in lab_sheets:
                        lab_vals, lab_period = analysis.parse_lab_sheet(
                            sh["df"], sh["dt_col"], sh["value_cols"])
                        lab_limits = limits_by_sheet.get(sh["name"], {})
                        if prefix_lab:
                            tok = analysis._clean_token(sh["name"])
                            ren = {c: f"{tok}_{c}" for c in lab_vals.columns}
                            lab_vals = lab_vals.rename(columns=ren)
                            lab_limits = {ren.get(k, k): v for k, v in lab_limits.items()}
                        lab_period_by_sheet[sh["name"]] = lab_period
                        lab_frames.append(analysis.add_lab_lags(
                            lab_vals, lab_period, target_series.index,
                            float(period_min), max_lag_min=0.0,
                            limits=lab_limits,
                        ))
                    all_feat = lagged
                    for lf in lab_frames:
                        all_feat = all_feat.join(lf) if not all_feat.empty else lf

                    merged = analysis.merge_with_target(all_feat, target_series, "Sazonal")
                    if merged.empty:
                        st.error(
                            "Nenhum período em comum entre processo e indicador alvo. "
                            "Verifique as colunas de data/hora e a sobreposição dos períodos."
                        )
                        return

                    # 3) Excursões de limites (sobre as séries brutas)
                    excursion_events: dict[str, pd.DataFrame] = {}
                    excursion_total: dict[str, float] = {}
                    raw_series: dict[str, dict] = {}
                    for sh in cont_sheets:
                        lim = limits_by_sheet.get(sh["name"], {})
                        if not lim:
                            continue
                        step = steps_by_sheet.get(sh["name"]) or 1.0
                        for col, (lo, hi) in lim.items():
                            label = f"{sh['name']} · {col}"
                            excursion_events[label] = analysis.detect_excursion_events(
                                sh["df"], sh["dt_col"], col, lo, hi, step)
                            n_valid = int(pd.to_numeric(
                                sh["df"][col], errors="coerce").notna().sum())
                            excursion_total[label] = n_valid * step
                            raw_series[label] = {
                                "df": sh["df"], "dt_col": sh["dt_col"], "col": col,
                                "lo": lo, "hi": hi,
                            }
                    for sh in lab_sheets:
                        lim = limits_by_sheet.get(sh["name"], {})
                        if not lim:
                            continue
                        step = lab_period_by_sheet.get(sh["name"]) or 60.0
                        for col, (lo, hi) in lim.items():
                            label = f"{sh['name']} · {col}"
                            excursion_events[label] = analysis.detect_excursion_events(
                                sh["df"], sh["dt_col"], col, lo, hi, step)
                            n_valid = int(pd.to_numeric(
                                sh["df"][col], errors="coerce").notna().sum())
                            excursion_total[label] = n_valid * step
                            raw_series[label] = {
                                "df": sh["df"], "dt_col": sh["dt_col"], "col": col,
                                "lo": lo, "hi": hi,
                            }
                    exc_summary = analysis.excursion_summary(
                        excursion_events, excursion_total)
                    exc_target = analysis.excursion_vs_target(merged)

                    # 4) Limpeza + estatística + modelo + explicabilidade
                    clean, rep_limpeza = analysis.clean_modeling_table(
                        merged, target="Sazonal")
                    X, y = analysis.split_features_target(clean, "Sazonal")

                    if X.shape[1] == 0:
                        st.error("Nenhuma variável explicativa restou após a limpeza.")
                        return

                    stat = analysis.statistical_ranking(X, y)

                    model_result = None
                    shap_imp = None
                    if run_model:
                        try:
                            model_result = analysis.train_model(
                                X, y, model_type=model_type)
                        except Exception as exc:  # noqa: BLE001
                            st.warning(f"Modelo não treinado: {exc}")
                    if run_shap and model_result is not None:
                        shap_imp = analysis.shap_importance(model_result, X)
                        if shap_imp is None:
                            st.info("SHAP indisponível — usando importância do modelo.")

                    importances = model_result.importances if model_result else None
                    consolidated = analysis.consolidated_ranking(stat, importances, shap_imp)
                    all_sheet_names = ([sh["name"] for sh in cont_sheets]
                                       + [sh["name"] for sh in lab_sheets])
                    # Shapley/LMG por indicador — "SHAP estatístico", funciona em
                    # todos os modos (não precisa de modelo).
                    ind_shapley, ind_shapley_r2 = analysis.indicator_shapley(
                        X, y, all_sheet_names, stat_df=stat)
                    diag = analysis.generate_diagnosis(consolidated, model_result,
                                                       sheet_names=all_sheet_names)
                    report_text = analysis.generate_report_text(
                        consolidated, model_result, stat,
                        excursion_tbl=exc_target, period_min=float(period_min),
                        n_periodos=len(clean), sheet_names=all_sheet_names,
                    )

                st.session_state["v2_results"] = {
                    "period_feat": period_feat,
                    "step_min": (min(steps_by_sheet.values())
                                 if steps_by_sheet else float(period_min)),
                    "steps_by_sheet": steps_by_sheet,
                    "lab_period_by_sheet": lab_period_by_sheet,
                    "period_min": float(period_min), "report": rep_limpeza,
                    "X": X, "y": y, "stat": stat, "model": model_result,
                    "shap": shap_imp, "consolidated": consolidated, "diag": diag,
                    "report_text": report_text,
                    "excursion_events": excursion_events,
                    "excursion_summary": exc_summary,
                    "excursion_vs_target": exc_target,
                    "raw_series": raw_series,
                    "all_sheet_names": all_sheet_names,
                    "model_type": (model_type if run_model else None),
                    "run_shap": run_shap,
                    "ind_shapley": ind_shapley, "ind_shapley_r2": ind_shapley_r2,
                    "n_periodos": len(clean), "n_features": X.shape[1],
                }
                # reinicia a investigação em cadeia ao reprocessar
                st.session_state.pop("v2_chain", None)
                st.session_state.pop("v2_chain_cache", None)
                # texto editável do relatório recomeça do texto automático
                st.session_state["v2_report_text_edit"] = report_text
                st.success(
                    f"Base por período pronta: **{len(clean)} período(s)** × "
                    f"**{X.shape[1]} variáveis explicativas**. Veja as demais abas."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro no processamento: {exc}")
                return

    res = st.session_state.get("v2_results")
    if not res:
        with tab_base:
            st.info("Configure as colunas na aba **Dados & Configuração** e clique em 🚀 Processar.")
        return

    period_feat = res["period_feat"]
    X, y = res["X"], res["y"]
    stat = res["stat"]
    model_result = res["model"]
    shap_imp = res["shap"]
    consolidated = res["consolidated"]
    sheet_names = res.get("all_sheet_names")

    # ===================================================================== #
    # ABA 2 — Base por Período
    # ===================================================================== #
    with tab_base:
        st.subheader("🧱 Base de indicadores por período",
                     help="Cada linha é um período do indicador sazonal; as colunas são as "
                          "variáveis do processo resumidas naquela janela (antes da expansão de lag).")
        st.caption(
            f"Período sazonal: **{_fmt_periodo(res['period_min'])}** · passo do processo: "
            f"**{res['step_min']:g} min**. Total modelado: **{res['n_periodos']} período(s)** × "
            f"**{res['n_features']} variáveis** (com lag)."
        )
        st.dataframe(period_feat.round(3), use_container_width=True)
        _attach_button("base_periodo", "table", "Base de indicadores por período (amostra)",
                       period_feat.round(3).reset_index().head(30), "Base por Período")

        rep = res["report"]
        with st.expander("🧹 Relatório de limpeza"):
            st.write(f"- Colunas não numéricas convertidas: **{len(rep['nao_numericas'])}**")
            st.write(f"- Colunas removidas por excesso de nulos: **{len(rep['muitos_nulos'])}**")
            if rep["muitos_nulos"]:
                st.caption(", ".join(rep["muitos_nulos"][:50]))
            st.write(f"- Colunas constantes removidas: **{len(rep['constantes'])}**")
            if rep["constantes"]:
                st.caption(", ".join(rep["constantes"][:50]))

    # ===================================================================== #
    # ABA 3 — Estatística
    # ===================================================================== #
    with tab_stat:
        if stat.empty:
            st.warning("Sem variáveis com associação calculável.")
        else:
            st.subheader("📊 Ranking estatístico de associação", help=HELP["estatistica"])
            st.dataframe(stat, use_container_width=True, hide_index=True)
            _attach_button("stat_ranking", "table", "Ranking estatístico de associação",
                           stat, "Estatística")
            g1, g2 = st.columns(2)
            sp = stat.set_index("Variável")["Spearman r"]
            fig_sp = _bar_rank(sp.abs().sort_values(ascending=False),
                               "Ranking |Spearman|", "|Spearman r|")
            with g1:
                st.plotly_chart(fig_sp, use_container_width=True, key="v2_bar_spearman")
                _attach_button("fig_spearman", "figure", "Ranking |Spearman|",
                               fig_sp, "Estatística")
            mi = stat.set_index("Variável")["Mutual Information"].sort_values(ascending=False)
            fig_mi = _bar_rank(mi, "Ranking Mutual Information", "Mutual Information",
                               color="Greens")
            with g2:
                st.plotly_chart(fig_mi, use_container_width=True, key="v2_bar_mi")
                _attach_button("fig_mi", "figure", "Ranking Mutual Information",
                               fig_mi, "Estatística")

            st.subheader("🔎 Dispersão: variável × indicador alvo")
            sel = st.selectbox("Variável", list(X.columns), key="v2_scatter_var")
            fig_scatter = _fig_scatter(X[sel], y, sel, "Indicador alvo")
            st.plotly_chart(fig_scatter, use_container_width=True, key="v2_scatter")
            _attach_button(f"fig_scatter_{analysis._clean_token(sel)}", "figure",
                           f"Dispersão: {sel} × indicador alvo", fig_scatter, "Estatística")

            if st.checkbox("Mostrar matriz de correlação (top variáveis)",
                           key="v2_show_corr"):
                top_vars = stat["Variável"].head(12).tolist()
                sub = X[top_vars].join(y.rename("Sazonal"))
                fig_corr = _fig_corr_matrix(sub)
                st.plotly_chart(fig_corr, use_container_width=True,
                                key="v2_corr_matrix")
                _attach_button("fig_corr_matrix", "figure",
                               "Matriz de correlação (top variáveis)",
                               fig_corr, "Estatística")

    # ===================================================================== #
    # ABA — Limites & Excursões
    # ===================================================================== #
    with tab_limits:
        exc_summary = res.get("excursion_summary")
        exc_events = res.get("excursion_events") or {}
        exc_target = res.get("excursion_vs_target")
        raw_series = res.get("raw_series") or {}
        if exc_summary is None or exc_summary.empty:
            st.info(
                "Nenhum limite crítico foi configurado. Defina limites na aba "
                "**📥 Dados & Configuração** (expander 🎯) e processe novamente "
                "para habilitar a análise de excursões."
            )
        else:
            st.subheader("🚦 Resumo de excursões por indicador", help=HELP["excursao"])
            st.dataframe(exc_summary, use_container_width=True, hide_index=True)
            _attach_button("exc_summary", "table", "Resumo de excursões por indicador",
                           exc_summary, "Limites & Excursões")

            st.subheader("📉 Linha do tempo das excursões")
            sel_ind = st.selectbox("Indicador", list(exc_events.keys()),
                                   key="v2_exc_ind")
            info = raw_series.get(sel_ind)
            if info is not None:
                dfr = info["df"]
                dts = pd.to_datetime(dfr[info["dt_col"]], dayfirst=True, errors="coerce")
                vals = pd.to_numeric(dfr[info["col"]], errors="coerce")
                serie = pd.Series(vals.values, index=dts).dropna().sort_index()
                # decima séries longas só para visualização
                if len(serie) > 20000:
                    serie = serie.iloc[:: len(serie) // 20000 + 1]
                fig_tl = go.Figure(go.Scatter(x=serie.index, y=serie.values,
                                              mode="lines", name=info["col"]))
                if info["lo"] is not None:
                    fig_tl.add_hline(y=info["lo"], line_dash="dash", line_color="red",
                                     annotation_text="mínimo")
                if info["hi"] is not None:
                    fig_tl.add_hline(y=info["hi"], line_dash="dash", line_color="red",
                                     annotation_text="máximo")
                ev = exc_events.get(sel_ind)
                if ev is not None and not ev.empty:
                    for _, r in ev.head(80).iterrows():
                        fig_tl.add_vrect(x0=r["inicio"], x1=r["fim"],
                                         fillcolor="red", opacity=0.15, line_width=0)
                fig_tl.update_layout(title=f"{sel_ind} — limites e excursões",
                                     xaxis_title="Tempo", yaxis_title=info["col"],
                                     height=420)
                st.plotly_chart(fig_tl, use_container_width=True, key="v2_exc_timeline")
                _attach_button(f"fig_exc_{analysis._clean_token(sel_ind)}", "figure",
                               f"Excursões: {sel_ind}", fig_tl, "Limites & Excursões")
                ev_show = exc_events.get(sel_ind)
                if ev_show is not None and not ev_show.empty:
                    with st.expander(f"📋 Eventos de excursão de {sel_ind}"):
                        st.dataframe(ev_show, use_container_width=True, hide_index=True)

            if exc_target is not None and not exc_target.empty:
                st.subheader("🧪 Violações de limite × indicador alvo",
                             help=HELP["mann_whitney"])
                st.caption(
                    "Compara o alvo nos períodos COM e SEM violação de cada limite. "
                    "p-value < 0,05 no teste de Mann-Whitney sugere impacto real."
                )
                st.dataframe(exc_target, use_container_width=True, hide_index=True)
                _attach_button("exc_vs_target", "table",
                               "Violações de limite × indicador alvo",
                               exc_target, "Limites & Excursões")

                # Boxplot do alvo: períodos com vs. sem violação (período atual)
                pct_cols = [c for c in X.columns
                            if "_pct_fora" in c and "_lag_" not in c]
                if pct_cols:
                    sel_box = st.selectbox(
                        "Variável de violação (boxplot do alvo)", pct_cols,
                        key="v2_exc_box", help=HELP["boxplot"],
                    )
                    grp = X[sel_box] > 0
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(y=y[~grp], name="Sem violação",
                                             marker_color="#2ca02c"))
                    fig_box.add_trace(go.Box(y=y[grp], name="Com violação",
                                             marker_color="#d62728"))
                    fig_box.update_layout(
                        title=f"Indicador alvo × violação ({sel_box})",
                        yaxis_title="Indicador alvo", height=420,
                    )
                    st.plotly_chart(fig_box, use_container_width=True, key="v2_exc_boxplot")
                    _attach_button(f"fig_box_{analysis._clean_token(sel_box)}", "figure",
                                   f"Boxplot do alvo: {sel_box}", fig_box,
                                   "Limites & Excursões")

    # ===================================================================== #
    # ABA 4 — Modelo & Explicabilidade
    # ===================================================================== #
    with tab_model:
        fig_serie = _fig_series(y, "Série temporal do indicador alvo", "Indicador alvo")
        st.plotly_chart(fig_serie, use_container_width=True, key="v2_prod_series")
        _attach_button("fig_serie_alvo", "figure", "Série temporal do indicador alvo",
                       fig_serie, "Modelo & Explicabilidade")
        if model_result is None:
            st.info("Modelo não foi treinado (escolha um tipo de análise com modelo).")
        else:
            m = model_result.metrics
            st.subheader(f"🤖 Modelo {model_result.model_name} — desempenho (teste)",
                         help=HELP["split_temporal"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE", f"{m['MAE']:.2f}", help=HELP["mae"])
            k2.metric("RMSE", f"{m['RMSE']:.2f}", help=HELP["rmse"])
            k3.metric("MAPE", "—" if np.isnan(m["MAPE"]) else f"{m['MAPE']:.1f}%", help=HELP["mape"])
            k4.metric("R²", "—" if np.isnan(m["R2"]) else f"{m['R2']:.3f}", help=HELP["r2"])
            _attach_button("model_metrics", "metrics",
                           f"Desempenho do modelo {model_result.model_name} (teste)",
                           {"MAE": f"{m['MAE']:.2f}", "RMSE": f"{m['RMSE']:.2f}",
                            "MAPE": "—" if np.isnan(m["MAPE"]) else f"{m['MAPE']:.1f}%",
                            "R²": "—" if np.isnan(m["R2"]) else f"{m['R2']:.3f}"},
                           "Modelo & Explicabilidade")

            pred_train = pd.Series(model_result.y_pred_train, index=model_result.y_train.index)
            pred_test = pd.Series(model_result.y_pred, index=model_result.y_test.index)
            fig_rp = _fig_real_vs_pred(y, pred_train, pred_test, "Indicador alvo")
            st.plotly_chart(fig_rp, use_container_width=True, key="v2_real_pred")
            _attach_button("fig_real_pred", "figure", "Indicador alvo: real × previsto",
                           fig_rp, "Modelo & Explicabilidade")

            cimp, cshap = st.columns(2)
            with cimp:
                st.subheader("Importância do modelo", help=HELP["importancia_modelo"])
                fig_imp = _bar_rank(model_result.importances,
                                    "Importância do modelo", "Importância")
                st.plotly_chart(fig_imp, use_container_width=True, key="v2_bar_imp")
                _attach_button("fig_importancia", "figure", "Importância do modelo",
                               fig_imp, "Modelo & Explicabilidade")
            with cshap:
                st.subheader("Ranking SHAP", help=HELP["shap"])
                if shap_imp is not None:
                    fig_shap = _bar_rank(shap_imp, "Ranking SHAP (|impacto| médio)",
                                         "|SHAP| médio", color="Oranges")
                    st.plotly_chart(fig_shap, use_container_width=True, key="v2_bar_shap")
                    _attach_button("fig_shap", "figure", "Ranking SHAP",
                                   fig_shap, "Modelo & Explicabilidade")
                else:
                    st.info("SHAP não disponível neste processamento (escolha 'Completa (com SHAP)').")

    # ===================================================================== #
    # ABA 5 — Ranking Consolidado + Diagnóstico
    # ===================================================================== #
    with tab_rank:
        if consolidated.empty:
            st.warning("Sem ranking consolidado disponível.")
        else:
            # --- Ranking LIMPO por indicador (primário) --------------------- #
            ind_rank = analysis.indicator_ranking(consolidated, sheet_names)
            st.subheader("🏆 Indicadores que mais impactaram o alvo",
                         help="Impacto consolidado por indicador (a evidência mais "
                              "forte entre as métricas e defasagens dele), sem "
                              "sufixos nem tipo de métrica.")
            st.dataframe(ind_rank, use_container_width=True, hide_index=True)
            _attach_button("rank_indicador", "table",
                           "Indicadores que mais impactaram o alvo",
                           ind_rank, "Ranking Consolidado")
            if not ind_rank.empty:
                fig_ind = _bar_rank(ind_rank.set_index("Indicador")["Impacto"],
                                    "Impacto por indicador", "Impacto (0–1)",
                                    color="Purples")
                st.plotly_chart(fig_ind, use_container_width=True, key="v2_bar_indicador")
                _attach_button("fig_indicador", "figure", "Impacto por indicador",
                               fig_ind, "Ranking Consolidado")

            # --- Contribuição de cada indicador (Shapley estatístico) ------- #
            ind_shap = res.get("ind_shapley")
            if ind_shap is not None and not ind_shap.empty:
                r2tot = float(res.get("ind_shapley_r2", 0.0) or 0.0)
                st.subheader("🧩 Contribuição de cada indicador (Shapley)",
                             help=HELP["shapley_indicador"])
                st.caption(
                    "Reparte de forma **justa** quanto cada indicador explica da "
                    "variação do alvo (valores de Shapley — a base do SHAP — mas "
                    "**sem treinar modelo**, então vale também no modo *Só "
                    f"estatística*). As parcelas somam o total explicado ≈ "
                    f"**{r2tot * 100:.0f}%**."
                )
                st.dataframe(ind_shap, use_container_width=True, hide_index=True)
                _attach_button("rank_shapley_ind", "table",
                               "Contribuição de cada indicador (Shapley)",
                               ind_shap, "Ranking Consolidado")
                fig_shap_ind = _bar_rank(
                    ind_shap.set_index("Indicador")["Contribuição (%)"],
                    "Contribuição por indicador (Shapley)", "Contribuição (%)",
                    color="Oranges")
                st.plotly_chart(fig_shap_ind, use_container_width=True,
                                key="v2_bar_shapley_ind")
                _attach_button("fig_shapley_ind", "figure",
                               "Contribuição por indicador (Shapley)",
                               fig_shap_ind, "Ranking Consolidado")

            with st.expander("🔬 Detalhe por métrica (variáveis derivadas)"):
                st.caption(
                    "Combina Spearman, Mutual Information, importância do modelo e "
                    "SHAP (quando disponíveis), normalizados no **Score consolidado**."
                )
                st.dataframe(consolidated.round(4), use_container_width=True,
                             hide_index=True)
                _attach_button("rank_consolidado", "table", "Ranking consolidado",
                               consolidated.round(4), "Ranking Consolidado")
                fig_cons = _bar_rank(
                    consolidated.set_index("Variável")["Score consolidado"],
                    "Score consolidado", "Score", color="Purples")
                st.plotly_chart(fig_cons, use_container_width=True,
                                key="v2_bar_consolidado")
                _attach_button("fig_consolidado", "figure", "Score consolidado",
                               fig_cons, "Ranking Consolidado")

            st.subheader("🧭 Diagnóstico gerencial")
            st.info(res["diag"])
            _attach_button("diag_texto", "text", "Diagnóstico gerencial",
                           res["diag"], "Ranking Consolidado")

            # --- Investigação causa-efeito em cadeia ------------------------ #
            st.divider()
            st.subheader("🔗 Investigação em cadeia (causa e efeito)",
                         help="Escolha um indicador de alto impacto e refaça a "
                              "análise usando-o como novo alvo (o valor do "
                              "indicador, sem variações estatísticas), para "
                              "descobrir o que impactou nele. Encadeie quantas "
                              "vezes quiser.")
            st.caption("A cada passo, o indicador escolhido vira o novo alvo e a "
                       "análise é refeita com as demais variáveis.")

            chain = st.session_state.setdefault("v2_chain", [
                {"label": "Alvo principal", "col": None, "ranking": ind_rank}])
            chain[0]["ranking"] = ind_rank   # objeto novo a cada rerun

            st.markdown("**Cadeia:** " + "  →  ".join(
                f"`{s['label']}`" for s in chain))

            cur = chain[-1]
            depth = len(chain)
            cur_rank = cur["ranking"]
            if cur_rank is None or cur_rank.empty:
                st.info("Sem drivers com evidência suficiente neste nível.")
            else:
                tgt_y_label = ("o indicador alvo" if cur["col"] is None
                               else analysis.humanize_variable(cur["col"], sheet_names))
                sel = st.selectbox(
                    f"Indicador para investigar (o que impactou em **{tgt_y_label}**):",
                    cur_rank["Indicador"].tolist(), key=f"v2_chain_sel_{depth}")
                if st.button("🔎 Investigar este indicador", type="primary",
                             key=f"v2_chain_go_{depth}"):
                    key_by_disp = {}
                    for c in X.columns:
                        k, d = analysis.base_indicator(c, sheet_names)
                        key_by_disp.setdefault(d, k)
                    tgt_key = key_by_disp.get(sel)
                    target_col = _repr_col(X, tgt_key, sheet_names) if tgt_key else ""
                    if not target_col:
                        st.warning("Não foi possível localizar a série deste indicador.")
                    else:
                        with st.spinner("Refazendo a análise com o novo alvo..."):
                            ranking = _cached_drilldown(
                                X, target_col, sheet_names,
                                res.get("model_type"), res.get("run_shap", False))
                        st.session_state["v2_chain"].append(
                            {"label": sel, "col": target_col, "ranking": ranking})
                        st.rerun()

            if depth > 1:
                cb1, cb2 = st.columns(2)
                if cb1.button("↩️ Voltar um nível", key="v2_chain_back"):
                    st.session_state["v2_chain"].pop()
                    st.rerun()
                if cb2.button("🔄 Reiniciar investigação", key="v2_chain_reset"):
                    st.session_state["v2_chain"] = st.session_state["v2_chain"][:1]
                    st.rerun()

            # dispersão: indicador escolhido no passo atual × alvo do passo
            if depth > 1 and cur["col"] in X.columns:
                prev = chain[-2]
                y_ser = y if prev["col"] is None else X[prev["col"]]
                yname = ("Indicador alvo" if prev["col"] is None
                         else analysis.humanize_variable(prev["col"], sheet_names))
                xname = analysis.humanize_variable(cur["col"], sheet_names)
                fig_chain = _fig_scatter(X[cur["col"]], y_ser, xname, yname)
                st.plotly_chart(fig_chain, use_container_width=True,
                                key=f"v2_chain_scatter_{depth}")
                _attach_button(f"fig_chain_{analysis._clean_token(cur['label'])}",
                               "figure", f"Cadeia: {xname} × {yname}",
                               fig_chain, "Ranking Consolidado")

    # ===================================================================== #
    # ABA 6 — Diagnóstico do Período
    # ===================================================================== #
    with tab_day:
        st.subheader("📅 Diagnóstico de um período específico",
                     help="Mostra o valor real do indicador sazonal no período e as variáveis "
                          "que mais contribuíram para a previsão (SHAP quando disponível).")
        if model_result is None:
            st.info("Treine um modelo (tipo de análise com modelo) para o diagnóstico do período.")
        else:
            periodos = list(X.index)
            labels = [pd.Timestamp(d).strftime("%d/%m/%Y %H:%M") for d in periodos]
            idx = st.selectbox("Período", range(len(periodos)),
                               format_func=lambda i: labels[i], key="v2_day_sel")
            date = periodos[idx]

            real = float(y.loc[date])
            hist_mean = float(y.mean())
            d1, d2 = st.columns(2)
            d1.metric("Indicador sazonal no período", f"{real:.2f}",
                      delta=f"{real - hist_mean:+.2f} vs. média")
            d2.metric("Média histórica", f"{hist_mean:.2f}")

            contrib = analysis.day_contributions(model_result, X, date)
            st.caption(
                "Principais variáveis que mais contribuíram para a previsão deste período "
                "(SHAP quando disponível), comparando o valor do período com a média histórica."
            )
            st.dataframe(contrib, use_container_width=True, hide_index=True)

            if not contrib.empty:
                cc = contrib.set_index("Variável")["Contribuição"].iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=cc.values, y=cc.index, orientation="h",
                    marker_color=["#d62728" if v < 0 else "#2ca02c" for v in cc.values],
                ))
                fig.update_layout(title="Contribuição por variável (período selecionado)",
                                  xaxis_title="Contribuição", height=max(320, 30 * len(cc)),
                                  margin=dict(l=10, r=10, t=50, b=40))
                st.plotly_chart(fig, use_container_width=True, key="v2_day_contrib")
                _attach_button(f"fig_contrib_{idx}", "figure",
                               f"Contribuições do período {labels[idx]}",
                               fig, "Diagnóstico do Período")

    # ===================================================================== #
    # ABA — Relatório (montagem, prévia HTML e exportação PDF)
    # ===================================================================== #
    with tab_report:
        st.subheader("📄 Relatório final",
                     help="Reúna os itens anexados (📎) nas abas de análise, ajuste o "
                          "texto e exporte em HTML (interativo) ou PDF.")

        meta = st.session_state.setdefault(report.META_KEY, {})
        mc1, mc2 = st.columns(2)
        with mc1:
            meta["titulo"] = st.text_input("Título do relatório",
                                           value=meta.get("titulo", "Relatório de Análise de Causas"),
                                           key="v2_rep_titulo")
        with mc2:
            meta["autor"] = st.text_input("Autor / área",
                                          value=meta.get("autor", ""), key="v2_rep_autor")
        meta["observacoes"] = st.text_input("Observações (opcional)",
                                            value=meta.get("observacoes", ""),
                                            key="v2_rep_obs")

        st.markdown("**🧭 Texto da análise** — gerado automaticamente a partir dos "
                    "rankings, lags e limites; edite à vontade:")
        diagnosis_md = st.text_area(
            "Texto do relatório", height=320, key="v2_report_text_edit",
            label_visibility="collapsed",
        )

        items = report.get_items(st.session_state)
        st.markdown(f"**📎 Itens anexados ({len(items)})**")
        if not items:
            st.info("Nenhum item anexado ainda — use os botões "
                    "**📎 Adicionar ao relatório** nas abas de análise.")
        for i, it in enumerate(items):
            c1, c2, c3, c4 = st.columns([6, 1, 1, 1])
            with c1:
                st.markdown(f"**{i + 1}. {it.title}**  \n"
                            f"<span style='color:#888;font-size:0.8em'>"
                            f"{it.kind} · origem: {it.source_tab}</span>",
                            unsafe_allow_html=True)
                it.comment = st.text_input(
                    "Comentário (opcional)", value=it.comment,
                    key=f"v2_rep_com_{it.id}", placeholder="Comentário no relatório...",
                    label_visibility="collapsed",
                )
            with c2:
                if st.button("▲", key=f"v2_rep_up_{it.id}", disabled=i == 0):
                    report.move_item(st.session_state, i, -1)
                    st.rerun()
            with c3:
                if st.button("▼", key=f"v2_rep_dn_{it.id}", disabled=i == len(items) - 1):
                    report.move_item(st.session_state, i, +1)
                    st.rerun()
            with c4:
                if st.button("🗑️", key=f"v2_rep_rm_{it.id}"):
                    report.remove_item(st.session_state, it.id)
                    st.rerun()

        st.divider()
        show_preview = st.toggle("👁️ Mostrar prévia do relatório", value=True,
                                 key="v2_rep_preview")
        if show_preview:
            try:
                html_preview = report.build_html(items, meta, diagnosis_md or "",
                                                 interactive=True)
                st_components.html(html_preview, height=900, scrolling=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro ao montar a prévia: {exc}")

        dl1, dl2 = st.columns(2)
        with dl1:
            try:
                html_doc = report.build_html(items, meta, diagnosis_md or "",
                                             interactive=True)
                st.download_button(
                    "⬇️ Baixar relatório (HTML interativo)", data=html_doc,
                    file_name="relatorio_analise_causas.html", mime="text/html",
                    key="v2_rep_dl_html",
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro ao gerar o HTML: {exc}")
        with dl2:
            fingerprint = report.items_fingerprint(items, meta, diagnosis_md or "")
            cache = st.session_state.get("v2_rep_pdf_cache")
            if st.button("🖨️ Gerar PDF", key="v2_rep_make_pdf"):
                try:
                    with st.spinner("Convertendo gráficos e montando o PDF..."):
                        pdf_bytes = report.build_pdf(items, meta, diagnosis_md or "")
                    st.session_state["v2_rep_pdf_cache"] = (fingerprint, pdf_bytes)
                    cache = st.session_state["v2_rep_pdf_cache"]
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Erro ao gerar o PDF: {exc}")
            if cache and cache[0] == fingerprint:
                st.download_button(
                    "⬇️ Baixar relatório (PDF)", data=cache[1],
                    file_name="relatorio_analise_causas.pdf", mime="application/pdf",
                    key="v2_rep_dl_pdf",
                )
            elif cache:
                st.caption("O relatório mudou — clique em **🖨️ Gerar PDF** novamente.")

    # ===================================================================== #
    # ABA — Exportar
    # ===================================================================== #
    with tab_export:
        st.subheader("⬇️ Exportar resultados (Excel)")
        sheets = {
            "Base por periodo": period_feat.round(4).reset_index(),
            "Ranking correlacoes": stat,
            "Ranking consolidado": consolidated.round(4),
        }
        if not stat.empty:
            sheets["Ranking MI"] = stat[["Variável", "Mutual Information"]].sort_values(
                "Mutual Information", ascending=False
            )
        if model_result is not None:
            sheets["Importancia modelo"] = (
                model_result.importances.rename("Importância")
                .reset_index().rename(columns={"index": "Variável"})
            )
        if shap_imp is not None:
            sheets["Ranking SHAP"] = (
                shap_imp.rename("SHAP médio").reset_index()
                .rename(columns={"index": "Variável"})
            )
        exc_summary_x = res.get("excursion_summary")
        if exc_summary_x is not None and not exc_summary_x.empty:
            sheets["Excursoes resumo"] = exc_summary_x
        exc_target_x = res.get("excursion_vs_target")
        if exc_target_x is not None and not exc_target_x.empty:
            sheets["Excursoes vs alvo"] = exc_target_x
        sheets["Diagnostico"] = res["diag"]
        sheets["Relatorio"] = res.get("report_text", "")

        st.download_button(
            "⬇️ Baixar resultados (Excel)",
            data=analysis.to_excel_multi(sheets),
            file_name="analise_sazonal_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="v2_download",
        )
        st.caption("O arquivo contém: base por período, rankings (correlações, MI, "
                   "importância, SHAP), ranking consolidado, excursões e diagnóstico.")
