"""
Gera o hash de uma senha para colar no cofre de segredos do Streamlit.

Uso interativo:
    python gen_password_hash.py
    # informe o salt e a senha quando solicitado

Uso direto (cuidado: a senha fica no histórico do shell):
    python gen_password_hash.py "<salt>" "<senha>"

O salt deve ser o MESMO usado em [auth].salt nos segredos, e uma frase
aleatória longa (ex.: gere com  python -c "import secrets;print(secrets.token_hex(24))").
"""

import getpass
import sys

from auth import hash_password


def main() -> None:
    if len(sys.argv) == 3:
        salt, pwd = sys.argv[1], sys.argv[2]
    else:
        salt = input("Salt (o mesmo de [auth].salt): ").strip()
        pwd = getpass.getpass("Senha: ")
        pwd2 = getpass.getpass("Repita a senha: ")
        if pwd != pwd2:
            print("As senhas não conferem.")
            sys.exit(1)
    if not salt:
        print("Salt vazio — defina um salt antes de gerar o hash.")
        sys.exit(1)
    print("\nCole no cofre de segredos, em [auth.users]:")
    print(f'    "usuario" = "{hash_password(pwd, salt)}"')


if __name__ == "__main__":
    main()
