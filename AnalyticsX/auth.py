"""
Controle de acesso (login + senha) para o Argus AnalyticsX
----------------------------------------------------------
Tela de login simples, sem dependências externas. As credenciais ficam no
cofre de segredos do Streamlit (``st.secrets``) — nunca no código nem no
repositório. As senhas são guardadas como **hash** (HMAC-SHA256 com um salt
por implantação), de modo que o segredo não revela a senha em texto puro.

Configuração (Streamlit Cloud → Settings → Secrets, ou .streamlit/secrets.toml
local):

    [auth]
    salt = "uma-frase-aleatoria-bem-longa-trocada-por-voce"

    [auth.users]
    "joao.silva" = "<hash>"
    "maria"      = "<hash>"

Gere cada ``<hash>`` com:

    python gen_password_hash.py

Uso no app:

    import auth
    user = auth.require_login()   # bloqueia o app até autenticar
    ...
    auth.logout_button()          # botão de sair (ex.: na sidebar)
"""

from __future__ import annotations

import hashlib
import hmac

import streamlit as st

import branding

SESSION_KEY = "auth_user"


def hash_password(password: str, salt: str) -> str:
    """HMAC-SHA256 da senha com o salt da implantação (hex)."""
    return hmac.new(salt.encode("utf-8"), password.encode("utf-8"),
                    hashlib.sha256).hexdigest()


def _auth_config():
    """Lê a seção [auth] dos segredos; retorna None se ausente."""
    try:
        cfg = st.secrets["auth"]
    except Exception:  # noqa: BLE001 - secrets ausentes
        return None
    users = cfg.get("users", {})
    if not users:
        return None
    return {"salt": cfg.get("salt", ""), "users": dict(users)}


def _verify(username: str, password: str, cfg: dict) -> bool:
    stored = cfg["users"].get(username)
    if not stored:
        return False
    calc = hash_password(password, cfg["salt"])
    return hmac.compare_digest(str(stored), calc)


def _render_setup_help() -> None:
    st.title("🔒 Argus AnalyticsX")
    st.error("Controle de acesso ainda não configurado.")
    st.markdown(
        "Para habilitar o login, adicione as credenciais no **cofre de "
        "segredos** do Streamlit (Settings → Secrets na nuvem, ou o arquivo "
        "`.streamlit/secrets.toml` localmente):"
    )
    st.code(
        '[auth]\n'
        'salt = "uma-frase-aleatoria-bem-longa"\n\n'
        '[auth.users]\n'
        '"joao.silva" = "<hash>"\n',
        language="toml",
    )
    st.caption("Gere cada `<hash>` rodando `python gen_password_hash.py`. "
               "As senhas nunca ficam em texto puro.")
    st.stop()


def _render_login() -> None:
    # centraliza o formulário
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        st.markdown(branding.login_header_html(), unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            st.subheader("Entrar")
            username = st.text_input("Usuário", key="login_user")
            password = st.text_input("Senha", type="password", key="login_pwd")
            ok = st.form_submit_button("Acessar", type="primary",
                                       use_container_width=True)
        if ok:
            cfg = _auth_config()
            if cfg and _verify(username.strip(), password, cfg):
                st.session_state[SESSION_KEY] = username.strip()
                st.rerun()
            else:
                st.error("Usuário ou senha inválidos.")
        st.caption("Acesso restrito. Em caso de dúvida, contate o "
                   "administrador do sistema.")
    st.stop()


def require_login() -> str:
    """Bloqueia o app até a autenticação. Retorna o usuário logado."""
    if st.session_state.get(SESSION_KEY):
        return st.session_state[SESSION_KEY]
    cfg = _auth_config()
    if cfg is None:
        _render_setup_help()
    _render_login()
    return ""  # inalcançável (st.stop), mas mantém o tipo de retorno


def logout_button(location=None) -> None:
    """Mostra o usuário logado e um botão de sair (padrão: barra lateral)."""
    container = location or st.sidebar
    user = st.session_state.get(SESSION_KEY)
    if not user:
        return
    container.markdown(f"👤 **{user}**")
    if container.button("Sair", key="logout_btn"):
        st.session_state.pop(SESSION_KEY, None)
        st.rerun()
