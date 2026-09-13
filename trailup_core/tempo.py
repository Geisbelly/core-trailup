"""Tempo esperado de resposta a uma questao.

Medido no EdNet (6,5 M respostas, split por aluno): R2 entre 0,56 e 0,57 sobre
o log da latencia, usando APENAS a mediana da questao (0,571 e 0,561 em duas
particoes por aluno). O modelo de boosting com 14 features chega a 0,651.

O aluno nao prediz o proprio tempo: sob split por aluno a mediana dele sequer
existe (aluno de teste nunca visto no treino), e o preditor cai para o global -
R2 -0,037, pior que a media.

O achado que define esta implementacao: ajustar pelo ritmo do aluno PIORA.

    so a mediana da questao          R2 0,571
    x ritmo_do_aluno^0,3             R2 0,568
    x ritmo_do_aluno^0,5             R2 0,530
    x ritmo_do_aluno^1,0             R2 0,303

O aluno quase nao prediz o proprio tempo (R2 0,007 sozinho): o que manda e a
questao. Entao a formula e a mediana, sem ajuste.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p, expm1

MIN_RESPOSTAS = 5          # abaixo disso a mediana ainda oscila demais
LIMITE_LENTO = 3.0         # x acima do esperado: 2,1% das respostas, acerto cai de 67% p/ 54%


@dataclass(frozen=True)
class Esperado:
    segundos: float
    respostas: int
    confiavel: bool
    def __str__(self):
        return (f'{self.segundos:.0f}s esperados (n={self.respostas}'
                f'{"" if self.confiavel else ", pouca evidencia"})')


def esperado(latencia_mediana: float, respostas: int) -> Esperado:
    """`respostas` = quantos ALUNOS responderam a questao."""
    if latencia_mediana <= 0:
        raise ValueError('latencia mediana deve ser positiva')
    return Esperado(float(latencia_mediana), respostas, respostas >= MIN_RESPOSTAS)


def quao_lento(segundos: float, esp: Esperado) -> float:
    """Razao entre o observado e o esperado. 1,0 = no ritmo da questao."""
    return segundos / esp.segundos if esp.segundos > 0 else 1.0


def demorando(segundos: float, esp: Esperado, limite: float = LIMITE_LENTO) -> bool:
    """Esta muito acima do esperado PARA ESTA QUESTAO?

    Nao e inferencia de estado: e comparacao com a distribuicao da propria
    questao. No EdNet, acima de 3x o acerto cai de 67% para 54%.
    """
    return esp.confiavel and quao_lento(segundos, esp) >= limite


def duracao_prevista(latencias_medianas: list[float]) -> float:
    """Duracao esperada de uma atividade inteira, em segundos."""
    return float(sum(latencias_medianas))
