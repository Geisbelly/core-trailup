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

# Compromisso do limiar, medido em 127.554 casos (errou na 1a vez, reencontrou).
# "acerta ao reencontrar" e a validacao: chute deve aprender MENOS.
#
#   limiar   marca    acerta ao reencontrar   nao-marcados   diferenca
#    0,15     0,6%           58,7%                72,3%       -13,6 pts
#    0,20     0,8%           60,8%                72,3%       -11,5 pts
#    0,30     1,9%           64,7%                72,3%        -7,6 pts   <- padrao
#    0,40     3,7%           66,4%                72,4%        -6,0 pts
#    0,60    10,8%           69,1%                72,6%        -3,5 pts
#
# A diferenca encolhe monotonicamente conforme o limiar afrouxa - o construto
# e real e graduado. 0,30 e o meio-termo; para sinalizar so o caso gritante,
# passe fracao=0,15 e ganhe o dobro de separacao marcando um terco.
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
