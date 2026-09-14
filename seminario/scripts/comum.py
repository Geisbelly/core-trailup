"""Configuracao compartilhada. Ajuste DADOS se baixou em outro lugar."""
import os
from pathlib import Path

DADOS = Path(os.environ.get('DADOS', Path(__file__).resolve().parents[1] / 'dados'))
EDNET = DADOS / 'EdNet-KT3' / 'KT3'
OULAD = DADOS / 'oulad'
SAIDA = Path(__file__).resolve().parents[1] / 'saida'
# parents[1] = seminario/ ; parents[2] = raiz do repositorio
SAIDA.mkdir(exist_ok=True)

# A janela do experimento. Os eixos saem dos dias 0-29 de cada aluno; o desfecho
# (voltou ou nao) sai dos dias 30-59. Nada da segunda janela entra nos eixos.
JANELA = 30
HORIZONTE = 60
GAP_SESSAO_SEG = 1800   # 30 min sem evento = sessao nova
MIN_EVENTOS = 30        # aluno com menos que isso na janela nao entra
JANELA_RECENTE = 10     # "recencia" = dias ativos nos ultimos 10 da janela

SEMENTES = [7, 11, 23, 42, 101, 202, 303, 404, 505, 606, 707, 2026]


def exige(caminho, comoobter):
    if not Path(caminho).exists():
        raise SystemExit(
            f'\nNao encontrei: {caminho}\n\n{comoobter}\n\n'
            f'Ou aponte a variavel DADOS para onde voce baixou:\n'
            f'    DADOS=/caminho/para/dados python3 scripts/01_preparar.py\n')
    return Path(caminho)
