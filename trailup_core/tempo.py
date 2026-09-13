"""Tempo esperado de resposta a uma questao.

Medido no EdNet (6,5 M respostas, split por aluno): R2 entre 0,56 e 0,58 sobre
o log da latencia, usando APENAS a estatistica da questao.

    media global                                  R2 -0,045
    mediana bruta da questao                      R2  0,574
    MEDIA DO LOG da questao (esperado_de_amostra) R2  0,578
    mediana encolhida para a global (k=5)         R2  0,575
    boosting com 14 features                      R2  0,651

Encolher para a media global nao ajuda: a mediana da questao ja e estavel com
poucas respostas. O teto do boosting nao e alcancavel por formula fechada.

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


def esperado_de_amostra(latencias, respostas: int | None = None) -> Esperado:
    """Estimador melhor que a mediana bruta: MEDIA DO LOG das latencias.

    O alvo previsto e log(tempo), entao a media no log e o estimador casado.
    Medido no EdNet, split por aluno: R2 0,578 contra 0,574 da mediana bruta.
    Ganho pequeno, mas de graca.

    RESSALVA: e menos robusto que a mediana a outlier extremo. Com poucas
    respostas e uma latencia absurda (aluno que deixou a tela aberta), a
    mediana e mais segura. O ganho medido vale no agregado, com o corte de
    600 s que a extracao aplica - aplique um corte parecido antes de chamar.
    """
    v = [float(x) for x in latencias if x is not None and float(x) > 0]
    if not v:
        raise ValueError('nenhuma latencia positiva na amostra')
    from math import log, expm1, log1p
    media_log = sum(log1p(x) for x in v) / len(v)
    n = respostas if respostas is not None else len(v)
    return Esperado(expm1(media_log), n, n >= MIN_RESPOSTAS)


def quao_lento(segundos: float, esp: Esperado) -> float:
    """Razao entre o observado e o esperado. 1,0 = no ritmo da questao."""
    return segundos / esp.segundos if esp.segundos > 0 else 1.0


def demorando(segundos: float, esp: Esperado, limite: float = LIMITE_LENTO) -> bool:
    """Esta muito acima do esperado PARA ESTA QUESTAO?

    Nao e inferencia de estado: e comparacao com a distribuicao da propria
    questao.

    Curva completa, medida em 300.000 respostas do EdNet (split por aluno):

        lentidao      acerto     respostas
        0 a 0,5x      73,6%        30.056
        0,5 a 1x      72,7%       119.999
        1 a 2x        62,7%       125.431     <- a queda acontece AQUI
        2 a 3x        54,0%        16.667
        3 a 5x        53,8%         5.957
        5 a 10x       53,2%         1.573

    Acima de 3x: 53,8% contra 67,4% abaixo - confere o que o modulo afirmava.

    MAS 3x E CONSERVADOR. A queda se da entre 1x e 2x (72,7% -> 62,7%) e o
    acerto ja esta em 54% na faixa de 2 a 3x; acima disso e plano. Com o
    padrao de 3x, so 2,6% das respostas sao marcadas. Baixar para 2,0 marca
    ~8% com praticamente a mesma separacao. E escolha de operacao: quantos
    casos voce quer olhar.
    """
    return esp.confiavel and quao_lento(segundos, esp) >= limite


def duracao_prevista(latencias_medianas: list[float]) -> float:
    """Duracao esperada de uma atividade inteira, em segundos."""
    return float(sum(latencias_medianas))
