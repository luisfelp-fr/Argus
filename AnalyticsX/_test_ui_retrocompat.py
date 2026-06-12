"""Retrocompatibilidade: sample_producao.xlsx (2 abas) no novo fluxo + modo v1.

Uso: python _test_ui_retrocompat.py
"""

import os

from streamlit.testing.v1 import AppTest

if not os.path.exists("sample_producao.xlsx"):
    import sample_data_producao as gen
    import pandas as pd
    processo, producao = gen.generate()
    with pd.ExcelWriter("sample_producao.xlsx", engine="openpyxl") as writer:
        processo.to_excel(writer, sheet_name="Processo", index=False)
        producao.to_excel(writer, sheet_name="Producao", index=False)
    print("sample_producao.xlsx gerado")

SCRIPT = r"""
import io
import streamlit as st


class FakeUpload(io.BytesIO):
    name = "sample_producao.xlsx"

    @property
    def size(self) -> int:
        return len(self.getvalue())


with open("sample_producao.xlsx", "rb") as f:
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
assert not at.exception, f"excecao: {at.exception}"
print("Render 2-abas OK (classificacao: Processo=continua, Producao=alvo via heuristica)")

at.selectbox(key="v2_tipo").select("Só estatística")
at.run()
at.button(key="v2_run").click()
at.run()
assert not at.exception, f"excecao no processamento: {at.exception}"
errors = [e.value for e in at.error]
assert not errors, f"st.error: {errors}"
ok = [s.value for s in at.success]
assert ok, "sem mensagem de sucesso"
print("Processamento 2-abas OK:", ok[0][:90])

res = at.session_state["v2_results"]
stat_idx = res["stat"].set_index("Variável")
# sem prefixo de aba (1 so aba continua) -> nomes iguais ao fluxo antigo
assert not any(v.startswith("Processo_") for v in stat_idx.index)
print("Nomes de colunas retrocompativeis (sem prefixo de aba) OK")

# ---- equivalencia numerica: pipeline antigo (build_period_features) vs ---- #
# ---- novo (build_period_features_multi com 1 aba) ------------------------- #
import pandas as pd
import analysis

proc_raw = pd.read_excel("sample_producao.xlsx", sheet_name="Processo")
prod_raw = pd.read_excel("sample_producao.xlsx", sheet_name="Producao")
ts, pm = analysis.parse_target_series(prod_raw, "Data", "Producao")
old_feat, old_step = analysis.build_period_features(
    proc_raw, "DataHora", ts.index, pm)
new_feat, steps = analysis.build_period_features_multi(
    [{"name": "Processo", "df": proc_raw, "dt_col": "DataHora", "limits": {}}],
    ts.index, pm)
pd.testing.assert_frame_equal(old_feat, new_feat)
assert abs(old_step - steps["Processo"]) < 1e-9
print(f"Equivalencia numerica antigo x novo OK "
      f"({new_feat.shape[0]} periodos x {new_feat.shape[1]} colunas identicas)")

# ---- modo classico v1 (app.py) continua funcionando -------------------- #
at1 = AppTest.from_file("app.py", default_timeout=300)
at1.run()
assert not at1.exception, f"excecao no app.py: {at1.exception}"
print("app.py (seletor de modo + v1) renderiza sem excecao")
print("TUDO OK")
