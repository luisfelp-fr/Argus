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

# Periodicas (Brix/Pol = sufixo _per_) NAO devem ter defasagem: so lag_0min
per_cols = [c for c in res["X"].columns if "_per_" in c]
assert per_cols, "esperava colunas periodicas (_per_) na base"
per_com_lag = [c for c in per_cols if "_lag_" in c and not c.endswith("_lag_0min")]
assert not per_com_lag, f"periodicas nao deveriam ter defasagem: {per_com_lag[:5]}"
print(f"Periodicas sem defasagem OK ({len(per_cols)} colunas, todas em lag 0)")

# rerun pos-processamento: abas de resultado renderizam sem excecao
at.run()
assert not at.exception, f"excecao ao renderizar resultados: {at.exception}"
print("Render das abas de resultado OK")
print("TUDO OK")
