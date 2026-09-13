"""Deteccao de chute: resposta rapida demais PARA AQUELA QUESTAO, e errada.

Validado no EdNet por consequencia, nao por definicao. No reencontro da mesma
questao pelo mesmo aluno:

    errou CHUTANDO      acerta depois em 65,1%
    errou TRABALHANDO   acerta depois em 72,5%
    tinha acertado      acerta depois em 86,1%

Quem chutou aprendeu menos - que e o que o conceito preve.

E e TRACO do aluno: correlacao +0,922 entre duas metades independentes das
respostas dele. Mediana 0,2%, p90 2,0%, p99 7,9% - uma cauda pequena chuta muito.

Uso: separar "nao sabe" de "nao tentou". Sao situacoes que pedem respostas
opostas, e hoje o sistema ve as duas como "erro".

Sem dependencia externa.
"""
from __future__ import annotations

FRACAO_RAPIDA = 0.30       # abaixo de 30% da mediana da questao
MIN_RESPOSTAS_QUESTAO = 50 # a mediana precisa ser estavel para o limiar valer


def foi_chute(segundos: float, acertou: bool, latencia_mediana_questao: float,
              respostas_questao: int, fracao: float = FRACAO_RAPIDA) -> bool:
    """O limiar e RELATIVO: questao de 60s e de 15s nao tem o mesmo 'rapido'."""
    if acertou or respostas_questao < MIN_RESPOSTAS_QUESTAO:
        return False
    if latencia_mediana_questao <= 0:
        return False
    return segundos < fracao * latencia_mediana_questao


def taxa_chute(chutes: int, respostas: int) -> float:
    return chutes / respostas if respostas else 0.0


def perfil_chute(taxa: float) -> str:
    """Faixas dos percentis medidos no EdNet."""
    if taxa >= 0.079: return 'muito acima do normal'   # p99
    if taxa >= 0.020: return 'acima do normal'         # p90
    return 'dentro do normal'
