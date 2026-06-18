"""Teste de UI ponta a ponta com streamlit.testing.AppTest.

Injeta sample_multiabas.xlsx no file_uploader (monkeypatch), deixa a
classificação automática das abas (heurística acerta as 4), seleciona
"Só estatística" p/ rapidez e clica em 🚀 Processar. Verifica ausência de
exceções e presença dos resultados nas abas.

Uso: python _test_ui_apptest.py
"""

from streamlit.testing.v1 import AppTest

SCRIPT = r"""
import io
import streamlit as st


class FakeUpload(io.BytesIO):
    name = "sample_multiabas.xlsx"

    @property
    def size(self) -> int:
        return len(self.getvalue())


with open("sample_multiabas.xlsx", "rb") as f:
    _DATA = f.read()

_real_uploader = st.file_uploader


def _fake_uploader(label, *args, **kwargs):
    if kwargs.get("key") == "v2_file":
        return FakeUpload(_DATA)
    return _real_uploader(label, *args, **kwargs)


st.file_uploader = _fake_uploader

import daily_ui
daily_ui.render_seasonal_mode()
"""

at = AppTest.from_string(SCRIPT, default_timeout=600)
at.run()
assert not at.exception, f"excecao na renderizacao inicial: {at.exception}"
print("Render inicial OK -", len(at.tabs), "abas")

# tipo de analise rapida
at.selectbox(key="v2_tipo").select("Só estatística")
at.run()
assert not at.exception, f"excecao ao trocar tipo: {at.exception}"

# clicar Processar
at.button(key="v2_run").click()
at.run()
assert not at.exception, f"excecao no processamento: {at.exception}"

errors = [e.value for e in at.error]
assert not errors, f"st.error exibido: {errors}"
ok = [s.value for s in at.success]
assert ok, "mensagem de sucesso nao exibida"
print("Processamento OK:", ok[0][:100])

res = at.session_state["v2_results"]
assert res["n_periodos"] > 100, res["n_periodos"]
assert not res["stat"].empty
assert res["report_text"]
assert res["excursion_summary"] is not None
print(f"Resultados: {res['n_periodos']} periodos x {res['n_features']} variaveis")
print("Texto do relatorio presente:", len(res["report_text"]), "chars")

# Periodicas (_per_) entram no periodo atual: SEM sufixo de lag nenhum.
cols = list(res["X"].columns)
per_cols = [c for c in cols if "_per_" in c]
assert per_cols, "esperava colunas periodicas (_per_) na base"
assert not any("_lag_" in c for c in per_cols), \
    f"periodicas nao deveriam ter sufixo de lag: {[c for c in per_cols if '_lag_' in c][:5]}"
# _last desconsiderada e period atual sem sufixo _lag_0min em TODA a base
assert not any("_last" in c for c in cols), "a variavel _last deveria estar fora"
assert not any("_lag_0min" in c for c in cols), \
    "periodo atual nao deve ter sufixo _lag_0min"
print(f"Periodicas sem defasagem/_last OK ({len(per_cols)} colunas periodicas)")

# rerun pos-processamento: abas de resultado renderizam sem excecao
at.run()
assert not at.exception, f"excecao ao renderizar resultados: {at.exception}"
print("Render das abas de resultado OK")

# ranking por indicador + estado da cadeia
assert res.get("all_sheet_names"), "all_sheet_names deveria estar em v2_results"

# Shapley por indicador deve existir MESMO no modo "Só estatística" (sem modelo)
assert res["model"] is None, "este teste roda em modo Só estatística"
ish = res.get("ind_shapley")
assert ish is not None and not ish.empty, "Shapley por indicador ausente no modo só estatistico"
assert list(ish.columns) == ["Indicador", "Contribuição (%)", "Parcela (R²)"]
print(f"Shapley por indicador OK sem modelo ({len(ish)} indicadores, "
      f"R2~{res.get('ind_shapley_r2')})")
chain = at.session_state["v2_chain"]
assert len(chain) == 1 and chain[0]["col"] is None, "cadeia comeca no alvo principal"
ind0 = chain[0]["ranking"]
assert list(ind0.columns) == ["Indicador", "Impacto", "Nº métricas"]
assert not ind0["Indicador"].str.contains(r"_lag_|_mean|_std|_per_", regex=True).any()
print(f"Ranking por indicador OK ({len(ind0)} indicadores limpos)")

# push: investigar o 1o indicador -> cadeia cresce para 2
at.selectbox(key="v2_chain_sel_1").select(ind0["Indicador"].iloc[0])
at.run()
at.button(key="v2_chain_go_1").click()
at.run()
assert not at.exception, f"excecao no drill-down: {at.exception}"
assert len(at.session_state["v2_chain"]) == 2, "cadeia deveria ter 2 passos"
print(f"Drill-down OK (passo 2: investigando '{at.session_state['v2_chain'][1]['label']}')")

# pop: voltar um nivel -> cadeia volta para 1
at.button(key="v2_chain_back").click()
at.run()
assert not at.exception, f"excecao ao voltar: {at.exception}"
assert len(at.session_state["v2_chain"]) == 1, "cadeia deveria voltar a 1 passo"
print("Voltar um nivel OK")
print("TUDO OK")
