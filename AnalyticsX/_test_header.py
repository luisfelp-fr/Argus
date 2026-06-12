"""Confirma que o app.py renderiza o cabecalho com o olho verde (logado)."""

from streamlit.testing.v1 import AppTest

import auth

SALT = "salt-teste"
HASH = auth.hash_password("x", SALT)

at = AppTest.from_file("app.py", default_timeout=120)
at.secrets["auth"] = {"salt": SALT, "users": {"u": HASH}}
at.session_state[auth.SESSION_KEY] = "u"   # ja logado
at.run()
assert not at.exception, f"excecao: {at.exception}"
mds = " ".join(m.value for m in at.markdown)
assert "Argus AnalyticsX" in mds, "titulo ausente no cabecalho"
assert "data:image/png;base64," in mds, "olho verde (imagem) ausente no cabecalho"
print("[OK] cabecalho do app com olho verde renderizado, sem excecao")
print("TUDO OK")
