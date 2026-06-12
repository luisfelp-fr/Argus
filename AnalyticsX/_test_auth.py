"""Testa o fluxo de login (auth.py) via streamlit AppTest.

Cria segredos em memória, verifica que o app bloqueia sem login, rejeita senha
errada e libera com a correta.
"""

from streamlit.testing.v1 import AppTest

import auth

SALT = "salt-de-teste-1234567890"
USER = "engenheiro"
PWD = "senha-correta"
HASH = auth.hash_password(PWD, SALT)

SECRETS = {"auth": {"salt": SALT, "users": {USER: HASH}}}

SCRIPT = """
import streamlit as st
import auth
user = auth.require_login()
auth.logout_button()
st.title("APP LIBERADO")
st.write(f"logado: {user}")
"""


def run(extra=None):
    at = AppTest.from_string(SCRIPT, default_timeout=30)
    at.secrets["auth"] = SECRETS["auth"]
    if extra:
        extra(at)
    at.run()
    return at


# 1) Sem login: app bloqueado (titulo do app nao aparece; form de login presente)
at = run()
assert "APP LIBERADO" not in [t.value for t in at.title], \
    "app deveria estar bloqueado sem login"
assert any(ti.key == "login_pwd" for ti in at.text_input), \
    "esperava o formulario de login"
print("[OK] bloqueia sem login e mostra formulario")

# 2) Senha errada
at = run()
at.text_input(key="login_user").set_value(USER)
at.text_input(key="login_pwd").set_value("errada")
at.button[0].click()
at.run()
assert "APP LIBERADO" not in [t.value for t in at.title]
assert any("inválidos" in e.value for e in at.error), "esperava erro de senha"
print("[OK] rejeita senha errada")

# 3) Senha correta
at = run()
at.text_input(key="login_user").set_value(USER)
at.text_input(key="login_pwd").set_value(PWD)
at.button[0].click()
at.run()
assert "APP LIBERADO" in [t.value for t in at.title], "deveria liberar"
assert at.session_state[auth.SESSION_KEY] == USER
print("[OK] libera com senha correta")

# 4) App sem segredos configurados: mostra ajuda de setup, nao libera.
#    (pulado quando ha um .streamlit/secrets.toml local — o AppTest o carrega)
import os
if os.path.exists(os.path.join(".streamlit", "secrets.toml")):
    print("[SKIP] cenario 'sem segredos' (existe .streamlit/secrets.toml local)")
else:
    at2 = AppTest.from_string(SCRIPT, default_timeout=30)
    at2.run()
    assert "APP LIBERADO" not in [t.value for t in at2.title]
    assert any("não configurado" in e.value or "configurado" in e.value
               for e in at2.error), "esperava aviso de setup"
    print("[OK] sem segredos -> mostra instrucoes de configuracao")

print("TUDO OK")
