"""Dominio do aluno num topico: probabilidade de acertar a proxima questao.

Substitui `DeepKnowledgeTracingAnalyzer` (base +-0,08/-0,07, confianca fixa 0,66),
que no EdNet da AUC 0,586 e SATURA: 27,7% das estimativas ficam no teto de 0,98,
porque o ponto de equilibrio da regra e 46,7% de acerto - acima disso todo aluno
sobe ate o teto e fica.

Formula medida (EdNet, split por aluno):

    so a regra atual                              AUC 0,586
    so a dificuldade da questao                   AUC 0,703
    0,7 questao + 0,3 topico                      AUC 0,722
    0,60 questao + 0,15 topico + 0,25 GLOBAL      AUC 0,727   <- esta
    modelo de boosting com 25 features            AUC 0,754

Uma linha de aritmetica captura 95% do ganho do modelo sobre a regra.

O TERCEIRO TERMO, E POR QUE ELE PESA MAIS QUE O TOPICO: o acerto do aluno em
TODAS as questoes e mais estavel que o acerto dele naquele topico, que quase
sempre tem poucas observacoes. Medido, o peso otimo do global (0,25) e maior
que o do topico (0,15). Testadas 14 combinacoes; o ganho e de +0,005 e some
se o global entrar sozinho (0,721).

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

# Dois conjuntos de pesos, porque os otimos sao diferentes:
PESO_QUESTAO_2 = 0.70      # sem historico global: 0,70 questao + 0,30 topico  -> 0,722
PESO_QUESTAO = 0.60        # com historico global                              -> 0,727
PESO_TOPICO = 0.15         # historico do aluno NAQUELE topico
PESO_GLOBAL = 0.25         # historico do aluno em tudo - mais estavel
PRIOR_ALUNO = 3            # encolhimento do acerto no topico
PRIOR_GLOBAL = 8           # encolhimento do acerto global (mais dados, encolhe menos)


@dataclass(frozen=True)
class Dominio:
    p: float               # probabilidade de acertar a proxima
    confianca: float       # cresce com observacao, satura
    tendencia: str         # 'subindo' | 'estavel' | 'caindo'
    respostas_topico: int
    def __str__(self):
        return (f'{self.p:.0%} de chance de acertar | confianca {self.confianca:.0%} '
                f'| {self.tendencia} (n={self.respostas_topico})')


def confianca(respostas: int) -> float:
    """Substitui o 0,66 fixo: cresce com observacao e satura em 0,92."""
    return round(min(0.92, 0.35 + 0.57 * (1 - exp(-respostas / 8))), 2)


def dominio(dificuldade_questao: float, acertos_no_topico: int, respostas_no_topico: int,
            media_global: float = 0.67, p_anteriores: list[float] | None = None,
            acertos_totais: int | None = None, respostas_totais: int | None = None) -> Dominio:
    """`dificuldade_questao` = taxa de acerto da questao (use dificuldade.estimar).

    `acertos_totais` / `respostas_totais` = historico do aluno em TODOS os
    topicos. Quando informados, entram como terceiro termo e levam a AUC de
    0,722 para 0,727. Sem eles, o peso do global vai para o topico e a formula
    volta a ser a de dois termos.
    """
    n = respostas_no_topico
    aluno = (acertos_no_topico + media_global * PRIOR_ALUNO) / (n + PRIOR_ALUNO)
    if respostas_totais is None or acertos_totais is None:
        p = PESO_QUESTAO_2 * dificuldade_questao + (1 - PESO_QUESTAO_2) * aluno
    else:
        if not 0 <= acertos_totais <= respostas_totais:
            raise ValueError('acertos_totais incompativel com respostas_totais')
        glob = (acertos_totais + media_global * PRIOR_GLOBAL) / (respostas_totais + PRIOR_GLOBAL)
        p = (PESO_QUESTAO * dificuldade_questao + PESO_TOPICO * aluno
             + PESO_GLOBAL * glob)
    tend = 'estavel'
    if p_anteriores and len(p_anteriores) >= 3:
        d = p - p_anteriores[-3]
        tend = 'subindo' if d > 0.03 else ('caindo' if d < -0.03 else 'estavel')
    return Dominio(round(p, 3), confianca(n), tend, n)


def precisa_reforco(d: Dominio, limiar: float = 0.45, conf_minima: float = 0.70) -> bool:
    """So afirma quando ha evidencia. A regra atual dispara sem olhar confianca."""
    return d.p < limiar and d.confianca >= conf_minima
