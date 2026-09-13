"""GATE: vale abrir a LLM para este aluno neste topico?

E a decisao mais cara do TrailUp - cada disparo e uma passagem por
`agente_conteudo` e, quando `gerar_materiais` abre, por `agente_geracao_midia`
(13 a 29 chamadas de texto + TTS + 10 a 18 imagens por perfil x topico).

ALVO MEDIDO: a taxa de acerto do aluno nas PROXIMAS 10 respostas do mesmo
topico ficar abaixo de 50%. Nao e "acerta a proxima" - uma questao dificil
derruba aquela probabilidade sem que o aluno esteja travado. Treinar no alvo
errado da 0,698, PIOR que a regra que ia substituir.

Medido no EdNet, split por aluno, 1.721.188 pontos de decisao, taxa base 12,2%:

    criterio                          AUC    prec@5%  prec@10%  lift@10%
    regra do TrailUp                 0,687     40,8%     32,9%     2,7x
    acerto acumulado no topico       0,720     36,3%     31,0%     2,5x
    forma fechada (3 termos)         0,725     37,0%     33,0%     2,7x
    ESTA (fechada + regra)           0,732     40,0%     33,9%     2,8x

Auditado em 2026-09-13 chamando risco() sobre 1.627.217 pontos de decisao:
AUC 0,738 e precisao 33,7% no ponto de 10% (lift 2,9x). Confere.
    boosting com 30 features         0,779         -     41,7%     3,4x

POR QUE COMBINAR COM A REGRA, QUE E PIOR. A regra satura - 27,7% das
estimativas ficam no teto. Como ESTIMATIVA isso e defeito. Mas no extremo
inferior a saturacao vira informacao: os 0,9% de alunos presos no piso tem
57,3% de taxa de alvo, contra 12,2% da base. A forma suave nao marca esses
casos, e e por isso que sozinha ela perde no ponto de 5%.

ONDE ELA NAO CHEGA: o boosting continua 0,047 de AUC acima, e no ponto de
operacao de 10% a diferenca e de 41,7% contra 33,9% de precisao. Se a API
puder carregar sklearn, use o modelo. Isto aqui e o melhor sem dependencia.

E O GAP NAO E DE NAO-LINEARIDADE. Testado (2026-09-13): exportar a interacao
entre as quatro features como TABELA de consulta - 1.646 celulas sobre
acerto no topico x acerto global x dificuldade x n de respostas:

    so o acumulado, linear                    AUC 0,746
    tabela de 1.646 celulas                   AUC 0,746
    tabela + linear                           AUC 0,748
    boosting com 30 features                  AUC 0,779

A tabela captura toda interacao possivel entre essas quatro variaveis e nao
ganha nada. O que separa 0,748 de 0,779 sao as OUTRAS 26 features, nao a forma
da funcao. Fechar o gap exige instrumentacao nova, nao aritmetica mais
esperta.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass

PASSO_ACERTO = 0.08        # a regra do TrailUp, reproduzida como componente
PASSO_ERRO = 0.07
PISO, TETO = 0.05, 0.98
PESO_ACUMULADO = 0.50      # medido: 0,5/0,5 foi o melhor no ponto de operacao
PESO_SEQUENCIAL = 0.50
PESO_TOPICO, PESO_GLOBAL, PESO_DIF = 0.50, 0.35, 0.15
# CONFERIDO (2026-09-13): 5 e o cotovelo. AUC do acerto acumulado no topico
# para prever travamento, por numero de respostas ja dadas:
#
#     n no topico     casos       AUC
#        1-2        127.273      0,534    <- quase acaso
#        3-4        119.650      0,610
#        5-9        265.976      0,660    <- o minimo
#       10-19       424.188      0,695
#       20-49       865.299      0,714
#       50+       3.801.801      0,738
#
# Abaixo de 5 o sinal e quase acaso (0,534 com 1-2 respostas). O corte esta
# no lugar certo, e nao era escolha justificada ate esta medida.
MIN_RESPOSTAS = 5          # abaixo disso o gate nao opina


@dataclass(frozen=True)
class Risco:
    score: float           # 0 a 1; so tem sentido COMPARADO dentro da coorte
    confiavel: bool
    respostas_topico: int

    def __str__(self):
        return (f'risco {self.score:.2f}' if self.confiavel
                else f'risco {self.score:.2f} (pouca evidencia, n={self.respostas_topico})')


def sequencial(acertos_em_ordem, inicial: float = 0.5) -> float:
    """Componente que reproduz a regra do TrailUp, inclusive a saturacao.

    A saturacao e mantida DE PROPOSITO: e ela que identifica o aluno preso no
    piso, que tem 57,3% de taxa de alvo contra 12,2% da base.
    """
    v = inicial
    for acertou in acertos_em_ordem:
        v = v + (PASSO_ACERTO if acertou else -PASSO_ERRO)
        v = min(TETO, max(PISO, v))
    return v


def risco(acertos_em_ordem, acerto_global: float, acerto_medio_topico: float,
          media_global: float = 0.67) -> Risco:
    """Risco de o aluno travar neste topico nas proximas respostas.

    acertos_em_ordem         - lista de bool das respostas do aluno NESTE topico,
                               em ordem cronologica
    acerto_global            - taxa de acerto do aluno em TODOS os topicos
    acerto_medio_topico      - taxa media de ACERTO das questoes ja vistas aqui
                               (facilidade: 0,9 = topico facil). O nome anterior
                               era `dificuldade_media_topico`, que invertia o
                               sentido para quem lesse so a assinatura: passar
                               0,9 como "muito dificil" REDUZIA o risco.

    O score nao e probabilidade calibrada: use-o para ORDENAR e corte no
    percentil da sua coorte, como em `engajamento.risco`.
    """
    seq = list(acertos_em_ordem)
    n = len(seq)
    ac_top = sum(seq) / n if n else media_global
    acumulado = (PESO_TOPICO * ac_top + PESO_GLOBAL * acerto_global
                 + PESO_DIF * acerto_medio_topico)
    s = sequencial(seq)
    # os dois componentes medem ACERTO; o risco e o complemento
    score = PESO_ACUMULADO * (1 - acumulado) + PESO_SEQUENCIAL * (1 - s)
    return Risco(round(min(1.0, max(0.0, score)), 3), n >= MIN_RESPOSTAS, n)


def deve_disparar(r: Risco, limiar_da_coorte: float) -> bool:
    """`limiar_da_coorte` = o score no percentil que voce escolheu disparar.

    Pontos de operacao medidos (taxa base 12,2%):

        dispara em    precisao    lift    cobertura
            5%          40,0%     3,3x       16%
           10%          33,9%     2,8x       28%
           20%          27,8%     2,3x       46%

    Nao usar limiar fixo: o score depende da distribuicao da coorte.
    """
    return r.confiavel and r.score >= limiar_da_coorte
