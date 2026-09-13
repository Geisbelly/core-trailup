"""Dominio do aluno num topico: probabilidade de acertar a proxima questao.

Substitui `DeepKnowledgeTracingAnalyzer` (base +-0,08/-0,07, confianca fixa 0,66),
que no EdNet da AUC 0,586 e SATURA: 27,7% das estimativas ficam no teto de 0,98,
porque o ponto de equilibrio da regra e 46,7% de acerto - acima disso todo aluno
sobe ate o teto e fica.

Formula medida (EdNet, split por aluno):

    so a regra atual                              AUC 0,586
    so a dificuldade da questao                   AUC 0,703
    0,7 x questao + 0,3 x aluno no topico         AUC 0,721   <- esta
    modelo de boosting com 25 features            AUC 0,754

Uma linha de aritmetica captura 95% do ganho do modelo sobre a regra.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

PESO_QUESTAO = 0.70        # medido: 0,7/0,3 foi melhor que 0,5/0,5 e 0,3/0,7
PRIOR_ALUNO = 3            # encolhimento do acerto do aluno no topico


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
            media_global: float = 0.67, p_anteriores: list[float] | None = None) -> Dominio:
    """`dificuldade_questao` = taxa de acerto da questao (use dificuldade.estimar)."""
    n = respostas_no_topico
    aluno = (acertos_no_topico + media_global * PRIOR_ALUNO) / (n + PRIOR_ALUNO)
    p = PESO_QUESTAO * dificuldade_questao + (1 - PESO_QUESTAO) * aluno
    tend = 'estavel'
    if p_anteriores and len(p_anteriores) >= 3:
        d = p - p_anteriores[-3]
        tend = 'subindo' if d > 0.03 else ('caindo' if d < -0.03 else 'estavel')
    return Dominio(round(p, 3), confianca(n), tend, n)


def precisa_reforco(d: Dominio, limiar: float = 0.45, conf_minima: float = 0.70) -> bool:
    """So afirma quando ha evidencia. A regra atual dispara sem olhar confianca."""
    return d.p < limiar and d.confianca >= conf_minima
