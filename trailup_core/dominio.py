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

CALIBRACAO (auditoria de 2026-09-13): o `p` devolvido E probabilidade, com
ECE 0,0083 apos a recalibracao. Antes dela o ECE era 0,0564 e o erro por decil
chegava a 14 pontos - o numero afirmava ser probabilidade sem ser.

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
# Os dois priors foram escolhidos sem medida. Varridos em 2026-09-13 sobre
# 120.000 previsoes, com IC95 por bootstrap - e o resultado e INSENSIVEL:
#
#   PRIOR_GLOBAL:  2 -> 0,7238 | 8 -> 0,7241 | 32 -> 0,7236 | 64 -> 0,7229
#   PRIOR_ALUNO :  1 -> 0,7239 | 3 -> 0,7241 | 8 -> 0,7240 | 16 -> 0,7237
#
# Todos os IC se sobrepoem. A AUC varia na quarta casa de 2 a 32 (global) e de
# 1 a 16 (aluno). Sao arbitrarios, e medimos que nao importam nessa faixa -
# o que e diferente de "nao tem base" e diferente de "tem base".
PRIOR_ALUNO = 3            # encolhimento do acerto no topico
PRIOR_GLOBAL = 8           # encolhimento do acerto global

# RECALIBRACAO. A media linear de duas probabilidades COMPRIME para o meio:
# sem correcao, o decil mais baixo previa 0,474 e observava 0,337 (14 pontos de
# erro), e o mais alto previa 0,825 contra 0,937. ECE 0,0564.
# Correcao de Platt em dois parametros, ajustada no treino e medida no teste:
#   ECE 0,0564 -> 0,0083   |   AUC inalterado (0,725)   |   faixa 0,21-0,95 -> 0,04-1,00
# b ~ 2 diz o tamanho da compressao: a media linear encolhe o logito pela metade.
#
# PONTO FIXO = 0,6412, resolvendo a/(1-b) no logito. NAO e a taxa base do
# corpus (0,6677) - uma correcao de Platt nao preserva a media. A expansao
# acontece em torno de 0,6412: acima dele as previsoes SOBEM, abaixo DESCEM.
# Como a taxa base (0,6677) fica acima do ponto fixo, a previsao media sobe um
# pouco - o oposto de "a recalibracao centra na media do corpus".
RECAL_A = -0.578273
RECAL_B = +1.995671


@dataclass(frozen=True)
class Dominio:
    p: float               # probabilidade de acertar a proxima
    confianca: float       # cresce com observacao, satura
    tendencia: str         # 'subindo' | 'estavel' | 'caindo'
    respostas_topico: int
    def __str__(self):
        return (f'{self.p:.0%} de chance de acertar | confianca {self.confianca:.0%} '
                f'| {self.tendencia} (n={self.respostas_topico})')


def incerteza(respostas: int, acertos: int = 0, media_global: float = 0.67) -> float:
    """Desvio-padrao posterior da taxa do aluno no topico. E o que ENCOLHE com n.

    Medido no EdNet, por faixa de n no topico:

        n         erro |p-y|   desvio da taxa do aluno
        4-7         0,385              0,169
        9-19        0,377              0,121
        21-49       0,378              0,080
        51-199      0,378              0,046
        201+        0,382              0,022

    A versao anterior de `confianca` era a formula 0,35 + 0,57*(1-exp(-n/8)),
    que ia de 0,48 a 0,92 - mas o erro real |p - y| e PLANO em 0,41. Ela
    afirmava que a estimativa fica muito mais confiavel com n, e nao fica: o
    |p-y| e dominado pelo ruido de Bernoulli do resultado, nao pelo erro da
    estimativa. O que de fato encolhe e a incerteza sobre a TAXA do aluno, e e
    isso que estas funcoes reportam agora.

    Usa o posterior Beta para nao devolver zero quando o aluno acertou tudo ou
    errou tudo (com n=2 e taxa 0 ou 1, o desvio binomial daria 0).
    """
    a = acertos + PRIOR_ALUNO * media_global
    b = (respostas - acertos) + PRIOR_ALUNO * (1 - media_global)
    if b < 0:
        raise ValueError(f'acertos={acertos} incompativel com respostas={respostas}')
    return sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))


def confianca(respostas: int, acertos: int = 0, media_global: float = 0.67) -> float:
    """1 menos duas vezes a incerteza da taxa do aluno, limitado a [0, 0,95].

    Nunca chega a 1: com 201+ respostas a incerteza ainda e 0,022, e o teto
    honesto fica em 0,95.
    """
    return round(min(0.95, max(0.0, 1 - 2 * incerteza(respostas, acertos, media_global))), 3)


def _recalibrar(p: float) -> float:
    from math import exp, log
    p = min(max(p, 1e-4), 1 - 1e-4)
    z = RECAL_A + RECAL_B * log(p / (1 - p))
    return 1.0 / (1.0 + exp(-max(-30.0, min(30.0, z))))


def dominio(dificuldade_questao: float, acertos_no_topico: int, respostas_no_topico: int,
            media_global: float = 0.67, p_anteriores: list[float] | None = None,
            acertos_totais: int | None = None, respostas_totais: int | None = None,
            recalibrar: bool = True) -> Dominio:
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
    if recalibrar:
        p = _recalibrar(p)
    tend = 'estavel'
    if p_anteriores and len(p_anteriores) >= 3:
        d = p - p_anteriores[-3]
        tend = 'subindo' if d > 0.03 else ('caindo' if d < -0.03 else 'estavel')
    return Dominio(round(p, 3), confianca(n, acertos_no_topico, media_global), tend, n)


def precisa_reforco(d: Dominio, limiar: float = 0.45, conf_minima: float = 0.70) -> bool:
    """So afirma quando ha evidencia. A regra atual dispara sem olhar confianca.

    ATENCAO - A RECALIBRACAO MUDOU QUANTO ISTO DISPARA. Com o `p` comprimido,
    `limiar=0,45` disparava em 2,3% das decisoes; com o `p` calibrado, dispara
    em 12,2%. CINCO VEZES MAIS chamadas de reforco no mesmo limiar.

    Nao e regressao: e o limiar passando a significar o que diz. Medido, o
    acerto de quem dispara fica abaixo do limiar em toda a faixa:

        limiar   dispara em   acerto observado
         0,30       3,3%           25,9%
         0,35       5,5%           29,7%
         0,40       8,3%           33,2%
         0,45      12,2%           36,8%     <- padrao
         0,50      17,2%           40,5%

    Antes, "p < 0,45" selecionava so os 2,3% mais extremos, porque quase tudo
    estava espremido em torno de 0,65.

    PARA MANTER O VOLUME ANTIGO de disparos, use `limiar=0,28`. Para manter o
    SIGNIFICADO ("reforcar quem tende a acertar menos de 45%"), mantenha 0,45 e
    dimensione a operacao para 12%.
    """
    return d.p < limiar and d.confianca >= conf_minima
