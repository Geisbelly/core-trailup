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

# Quao estavel e a mediana da questao, por numero de respostas (medido em
# 11.421 questoes, contra a mediana calculada com 50+):
#
#    n     correlacao   erro relativo medio
#    3       0,854            25,9%
#    5       0,886            20,2%      <- MIN_RESPOSTAS
#    8       0,917            15,2%
#   15       0,942            11,5%
#   30       0,969             7,7%
#
# ATENCAO - 5 NAO E "ESTAVEL". O comentario anterior dizia "abaixo disso a
# mediana ainda oscila demais", sugerindo que a partir de 5 ela nao oscila.
# Com 5 respostas o erro tipico e de 20%.
#
# O limite serve para o RITMO (classificar rapida/lenta acerta 97,3% com 5,
# porque o corte de 40 s fica longe da mediana da maioria das questoes) e NAO
# serve para cravar duracao. Para +-10% no tempo esperado, exija ~15; para
# +-8%, ~30.
MIN_RESPOSTAS = 5
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
    """`respostas` = quantos ALUNOS responderam a questao.

    `Esperado.confiavel` fica True a partir de MIN_RESPOSTAS, mas isso quer
    dizer "serve para classificar ritmo", nao "o numero esta preciso". Use
    `precisao()` para saber o erro tipico daquela estimativa.
    """
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


def precisao(respostas: int) -> float:
    """Erro relativo tipico da mediana da questao, com `respostas` observacoes.

    Interpola a curva medida no EdNet. Devolve 0,20 para n=5 e 0,08 para n=30.
    """
    tab = [(3, .259), (5, .202), (8, .152), (15, .115), (30, .077), (50, .060)]
    if respostas <= tab[0][0]:
        return tab[0][1]
    if respostas >= tab[-1][0]:
        return tab[-1][1]
    for (a, va), (b, vb) in zip(tab, tab[1:]):
        if respostas <= b:
            return va + (vb - va) * (respostas - a) / (b - a)
    return tab[-1][1]


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


# Soma de medianas NAO e a mediana da soma. Medido no EdNet, a soma crua das
# medianas subestima o tempo real da atividade em 13%, de forma estavel:
#
#   questoes   soma das medianas   mediana real   media real   erro
#      5             102 s            117 s         140 s      -13%
#     10             207 s            239 s         291 s      -13%
#     20             404 s            466 s         594 s      -13%
#
# O fator abaixo corrige o vies: 207 x 1,155 = 239, a mediana real.
CORRECAO_SOMA = 1.155


def duracao_prevista(latencias_medianas: list[float],
                     correcao: float = CORRECAO_SOMA) -> float:
    """Duracao esperada de uma atividade inteira, em segundos.

    A soma crua das medianas subestima em 13% - latencia e assimetrica a
    direita e a soma se concentra na media, nao na mediana. `correcao` desfaz
    o vies (medido; ver a tabela acima).

    PRECISAO: e estimativa grosseira. Mesmo corrigida, so ~62% das atividades
    de 10 questoes caem dentro de +-25% do tempo real, e o erro absoluto medio
    e de 24%. Serve para dizer "uns 4 minutos", nao para cronometrar.

    Somar MEDIAS em vez de medianas acerta a mediana total sem correcao
    (243 s contra 239 s reais), mas tem erro absoluto pior (26%). Por isso a
    escolha foi mediana + fator.
    """
    return float(sum(latencias_medianas)) * correcao
