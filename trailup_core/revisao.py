"""Quando revisar: curva de esquecimento medida.

585.663 reencontros da mesma questao pelo mesmo aluno no EdNet. Retencao em
funcao do intervalo:

    intervalo        acertou antes   errou antes
    < 1 hora             90,1%          83,6%
    1 - 3 dias           87,3%          71,5%
    1 - 3 semanas        83,5%          61,4%
    2 - 6 meses          83,1%          57,5%

Quem acertou perde 7 pontos em seis meses; quem errou perde 26. Aprendizado
raso evapora, consolidado resiste.

Controle: a dificuldade media das questoes e constante entre as faixas
(0,715 a 0,734), entao a queda e do TEMPO, nao da composicao.

Sem dependencia externa.

VALIDACAO COMO PREDITOR (2026-09-13, 223.358 reencontros, split por aluno):

    curva de esquecimento (este modulo)     AUC 0,665   ECE 0,021
    so "acertou antes" (sim/nao)            AUC 0,586
    so a dificuldade da questao             AUC 0,618
    curva + dificuldade da questao          AUC 0,661   ECE 0,077

A curva bate os dois componentes isolados, e somar a dificuldade da questao
PIORA - tanto a ordenacao quanto a calibracao. Ela ja carrega o que precisa.

Ate esta medida a curva era so descritiva: o formato tinha sido medido, mas
nunca testado como preditor.
"""
from __future__ import annotations

from bisect import bisect_right

# (dias, retencao) medidos
_ACERTOU = [(0.04, .901), (1, .907), (3, .873), (7, .859), (21, .835), (60, .819), (180, .831)]
_ERROU   = [(0.04, .836), (1, .793), (3, .715), (7, .661), (21, .614), (60, .581), (180, .575)]


def retencao(dias: float, acertou_antes: bool) -> float:
    """Probabilidade estimada de acertar ao reencontrar, apos `dias`."""
    t = _ACERTOU if acertou_antes else _ERROU
    if dias <= t[0][0]: return t[0][1]
    if dias >= t[-1][0]: return t[-1][1]
    i = bisect_right([x[0] for x in t], dias) - 1
    (a, va), (b, vb) = t[i], t[i + 1]
    return va + (vb - va) * (dias - a) / (b - a)


def dias_ate_revisar(acertou_antes: bool, retencao_alvo: float = 0.75) -> float:
    """Em quantos dias a retencao cai ate o alvo? Devolve o prazo de revisao.

    Com alvo de 75%: quem ERROU precisa voltar em ~2 dias; quem ACERTOU nao
    chega a cair ate 75% no horizonte medido (180 dias) - revisar por outra
    razao que nao o esquecimento.
    """
    t = _ACERTOU if acertou_antes else _ERROU
    if t[0][1] < retencao_alvo:
        return 0.0
    for (a, va), (b, vb) in zip(t, t[1:]):
        if vb <= retencao_alvo:
            if va == vb: return float(b)
            return a + (b - a) * (va - retencao_alvo) / (va - vb)
    return float('inf')


def prioridade_revisao(dias_desde: float, acertou_antes: bool) -> float:
    """0 a 1: quanto esta questao precisa ser revisada agora."""
    return max(0.0, min(1.0, 1.0 - retencao(dias_desde, acertou_antes)))
