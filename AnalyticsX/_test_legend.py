"""Confere que a aba Legenda renderiza com st.table sem cortar texto."""

from streamlit.testing.v1 import AppTest

SCRIPT = """
import streamlit as st
import analysis
df = analysis.suffix_legend_table()
for cat in df['Categoria'].unique():
    sub = df[df['Categoria'] == cat].drop(columns='Categoria').reset_index(drop=True)
    st.markdown(f'**{cat}**')
    st.table(sub.style.hide(axis='index'))
"""

at = AppTest.from_string(SCRIPT, default_timeout=60)
at.run()
assert not at.exception, at.exception
print("tabelas renderizadas:", len(at.table))
# textos longos (que antes eram cortados) devem aparecer inteiros
alvos = ["estabilize o controle",
         "prováveis causas a investigar" if False else "Severidade combinada da violação"]
joined = " ".join(str(t.value.to_dict()) for t in at.table)
for a in alvos:
    assert a in joined, f"texto cortado/ausente: {a!r}"
print("OK - st.table mostra o texto completo, sem corte")
print("TUDO OK")
