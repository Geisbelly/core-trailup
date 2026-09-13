"""Engajamento: eixos separados, validados em DUAS realidades diferentes.

O QUE RESPONDE
    "Este aluno ainda vai estar estudando daqui a um mes?"
    Nao responde "vai aprender mais" - ver LIMITE, no fim.

ONDE FOI MEDIDO (as duas bases sao deliberadamente distantes)
    EdNet KT3  adulto coreano, autoestudo de TOEIC, sem turma, sem professor,
               continuar e 100% voluntario.   18.119 alunos, retencao 30,5%
    OULAD      universitario britanico, EAD, com turma, matricula e abandono
               formal registrado.             24.731 alunos, retencao 92,9%

    Mesmo desenho nas duas: eixos nos dias 0-29, desfecho nos dias 30-59.
    Nada da segunda janela entra no calculo dos eixos.

OS EIXOS, E QUAIS SOBREVIVERAM AS DUAS BASES
    (AUC = chance de ordenar certo um par voltou/nao-voltou; 0,50 = cara ou coroa)

    eixo            EdNet              OULAD              replica?
    recencia        0,810 [0,804-0,818]  0,863 [0,854-0,870]  SIM
    frequencia      0,799 [0,792-0,806]  0,811 [0,800-0,820]  SIM
    profundidade    0,624 [0,617-0,634]  0,564 [0,549-0,581]  SIM, fraco
    volume          0,400 [0,393-0,411]  0,522 [0,507-0,536]  NAO - REMOVIDO

    O volume (respostas por sessao) tinha sinal NEGATIVO forte no EdNet e virou
    ruido no OULAD. Uma inversao que aparece numa base e some na outra nao e
    achado - e propriedade daquela base. Saiu do modulo.

RECENCIA E O EIXO NOVO, E O MELHOR DOS DOIS
    Dias ativos nos ULTIMOS 10 dias da janela, nao nos 30. Bate frequencia nas
    duas bases, e as duas juntas ganham ~4 pontos de AUC sobre o desenho antigo:

        3 eixos antigos (freq, volume, prof)   EdNet 0,802 | OULAD 0,815
        recencia + frequencia                  EdNet 0,843 | OULAD 0,861

    Leitura: "apareceu ultimamente" vale mais que "aparece bastante".
    Consequencia de produto: recalcular perto da decisao, nao uma vez por mes.

CORTES ABSOLUTOS NAO TRANSFEREM - ESTE E O PONTO DA CALIBRACAO
    Aplicando os cortes do EdNet ao OULAD, com a MESMA unidade:
        frequencia:     EdNet 42% baixo / 24% medio / 33% alto
                        OULAD  6% baixo / 17% medio / 77% alto
    Por isso as faixas saem de tabela por coorte, e `calibrar()` existe.

UM MODELO SO, TREINADO NAS DUAS BASES JUNTAS
    Juntar as bases cruas NAO funciona: com os valores absolutos, prever DE QUAL
    BASE a linha veio da AUC 0,994. O modelo aprende a base e aplica a taxa dela
    - nao mede engajamento. Treinar assim PIORA o EdNet (0,849 -> 0,788).

    A correcao e padronizar DENTRO de cada coorte (z) ANTES de juntar:

        prever a base, valor absoluto        AUC 0,994
        prever a base, z dentro da coorte    AUC 0,508   <- indistinguivel

    Com isso, um modelo unico treinado nas duas empata com o modelo especifico
    de cada uma:

        base     modelo unico   so aquela base
        EdNet       0,856           0,854
        OULAD       0,862           0,862

    E ele ordena bem ate numa base que nunca viu:
        treinado no EdNet, ordenando o OULAD    AUC 0,865
        treinado no OULAD, ordenando o EdNet    AUC 0,842

    E POR ISSO QUE `ordenar()` EXISTE E `calibrar()` E OBRIGATORIO:
    a ORDEM transfere entre plataformas; o NIVEL nao transfere de jeito nenhum
    (ECE 0,36 a 0,57 ao exportar o modelo de uma base para a outra). O mesmo
    ordenador precisa de limiares completamente diferentes:

        base    saem    limiar p/ alertar os 10% piores   precisao
        EdNet   69,8%              0,440                    94,9%
        OULAD    7,1%              0,582                    38,7%

ACURACIA SOZINHA NAO DIZ NADA AQUI
    No OULAD, com limiar 0,5 o modelo alerta NINGUEM e acerta 92,9% - a mesma
    acuracia de "nunca alertar". Acuracia balanceada nesse ponto: 50,0%.
    Use ponto de operacao (alertar os X% de maior risco), nao limiar fixo.

LIMITE MEDIDO - LEIA ANTES DE EXIBIR
    Nenhum eixo prediz GANHO de acerto entre os dois meses (Spearman -0,04 a
    +0,03, n=5.523 no EdNet). Engajamento prediz PERMANENCIA, nao APRENDIZADO.

ONDE NAO FUNCIONOU
    ARES (200 universitarios, leitura em L2 obrigatoria, 6 semanas): retencao
    de 87,7% e AUC 0,49-0,57 em todos os eixos. Quando a participacao e
    compulsoria e quase todo mundo volta, nao ha variacao a prever. Este modulo
    pressupoe continuacao VOLUNTARIA.

TRAJETORIA: "caiu / estavel / subiu"
    A tendencia da atividade nao acrescenta ao modelo - recencia + frequencia
    ja a contem (saber quanto o aluno fez no total e quanto fez ultimamente
    determina a forma). Somar tendencia rende +0,003 de AUC nas duas bases.

    Mas como EXPLICACAO ela e forte, desde que comparada do jeito certo.
    Fixando o TOTAL de dias ativos na janela:

        total de dias    caiu    estavel   subiu     (retencao, EdNet)
             3 a 6      18,6%     43,5%    60,0%
             7 a 12     38,5%     71,9%    81,6%
            13 a 20     59,4%     83,8%    89,1%

        total de dias    caiu    estavel   subiu     (retencao, OULAD)
             3 a 6      74,3%     87,2%    91,6%
             7 a 12     89,9%     96,4%    97,7%
            13 a 20     96,9%     99,0%    99,4%

    Monotonico nos seis estratos, nas duas bases, com ate 41 pontos de
    diferenca. Mesmo esforco total, formas diferentes, destinos diferentes.

    O CONTROLE ERRADO INVERTE O SINAL - e este e o alerta. Fixando a RECENCIA
    em vez do total, "caiu" aparece com retencao MAIOR que "estavel" (55,4%
    contra 46,8% no EdNet), porque entre alunos igualmente ativos agora, quem
    "caiu" e quem era mais ativo antes (5,1 contra 1,6 dias na 1a janela).
    Alertar sobre queda sem fixar o total marca justamente os mais engajados.

A PROFUNDIDADE FICA FORA DO ORDENADOR
    Ela ajuda dentro de uma base, mas atrapalha no modelo unico (EdNet cai de
    0,856 para 0,848), porque significa coisas diferentes em cada plataforma -
    segundos lendo explicacao no EdNet, fracao de cliques em conteudo no OULAD.
    Fica como eixo de DIAGNOSTICO (o que esta acontecendo), nao de predicao.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log

JANELA_DIAS = 30        # janela de observacao
# JANELA_RECENTE = 10 foi escolhido sem medida. Varrido em 2026-09-13, e a
# conclusao e um aviso: OTIMIZAR O EIXO SOZINHO PIORA O SISTEMA.
#
# O eixo isolado (AUC contra retencao, IC95 por bootstrap):
#
#     janela    EdNet                OULAD
#        5      0,767 [0,760-0,773]  0,834 [0,825-0,842]
#        7      0,789 [0,782-0,795]  0,857 [0,847-0,864]
#       10      0,810 [0,805-0,818]  0,863 [0,854-0,870]   <- padrao
#       14      0,824 [0,818-0,830]  0,862 [0,853-0,870]
#       20      0,827 [0,821-0,833]  0,847 [0,837-0,856]
#       30      0,799 [0,791-0,806]  0,811 [0,800-0,820]
#
# Por ai, 14 parece melhor: ganha 0,014 no EdNet e empata no OULAD. MAS o que
# o modulo entrega e `ordenar()`, o modelo compartilhado pelas duas bases - e
# com janela 14 ele DESABA:
#
#     janela    modelo unico          so aquela base
#       10      EdNet 0,856           0,854
#       14      EdNet 0,812           0,853
#       14      OULAD 0,855           0,855
#
# Com 14 dias a recencia ocupa metade da janela de 30 e fica colinear com a
# frequencia - o peso da frequencia chega a ficar NEGATIVO (-0,025), sinal de
# que o ajuste conjunto perdeu o pe. As duas bases divergem mais, e os pesos
# compartilhados servem pior as duas.
#
# 30 e pior que tudo de 5 para cima nas duas bases: a janela cheia dilui, que
# e o que sustenta a recencia existir como eixo separado da frequencia.
JANELA_RECENTE = 10     # ultimos dias da janela que formam a recencia
GAP_SESSAO_SEG = 1800   # inatividade que separa duas sessoes
# MIN_EVENTOS = 30 foi escolhido sem medida, e "abaixo disso os eixos sao
# ruido" sugeria que a partir de 30 eles funcionam. Medido no EdNet, o AUC da
# recencia por faixa de eventos na janela:
#
#    eventos      alunos    AUC
#     30-49        4.708    0,644
#     50-99        5.032    0,718
#    100-299       5.440    0,801
#    300-999       2.491    0,864
#    1000+           448    0,853
#
# No minimo de 30 o eixo entrega 0,644, nao os 0,810 do agregado. O numero de
# capa vem dos alunos com muita atividade. 30 e o piso para NAO devolver lixo;
# para o desempenho anunciado, exija ~100. `Engajamento.confiavel` diz apenas
# que passou do piso.
MIN_EVENTOS = 30        # piso; ver a curva acima antes de confiar no numero

FAVORAVEL = {'recencia': True, 'frequencia': True, 'profundidade': True}

# Retencao observada por dias ativos nos ultimos 10, nas duas bases de
# referencia. Nao sao para usar como probabilidade no seu produto - estao aqui
# para mostrar a FORMA (monotonica nas duas) e o quanto o nivel muda de base
# para base. Use `calibrar()` com dado proprio.
REFERENCIA = {
    'EdNet': {0: .124, 1: .473, 2: .555, 3: .681, 4: .759, 5: .808,
              6: .887, 7: .908, 8: .933, 9: .961, 10: .982},
    'OULAD': {0: .613, 1: .886, 2: .935, 3: .963, 4: .982, 5: .986,
              6: .992, 7: .998, 8: .999, 9: .999, 10: 1.00},
}

# Cortes estruturais da recencia: "sumiu" / "apareceu pouco" / "esta presente".
# Sao os unicos cortes que valem nas duas bases, porque sao contagem, nao escala.
CORTE_RECENCIA = (0, 3)   # 0 dias = baixo | 1 a 3 = medio | 4+ = alto

# Ordenador comum: logistica treinada nas DUAS bases juntas, sobre features
# padronizadas DENTRO de cada coorte. Preve p(continuar).
# So vale sobre z da propria coorte - passar valor cru aqui nao significa nada.
PESOS = {'intercepto': 0.8462, 'recencia': 0.8285, 'frequencia': 0.1358}


@dataclass(frozen=True)
class Eixo:
    nome: str
    valor: float
    faixa: str            # 'baixo' | 'medio' | 'alto'

    @property
    def bom(self) -> bool:
        return self.faixa == ('alto' if FAVORAVEL[self.nome] else 'baixo')


@dataclass(frozen=True)
class Calibracao:
    """Tudo que e propriedade da SUA coorte, e nao transfere de outra."""
    cortes: dict = field(default_factory=dict)      # eixo -> (lim_baixo, lim_medio)
    retencao: dict = field(default_factory=dict)    # dias_recentes -> taxa observada
    media: dict = field(default_factory=dict)       # eixo -> media da coorte
    desvio: dict = field(default_factory=dict)      # eixo -> desvio da coorte
    limiar: dict = field(default_factory=dict)      # taxa de alerta -> limiar de risco
    taxa_efetiva: dict = field(default_factory=dict)  # taxa pedida -> taxa que de fato dispara
    n: int = 0

    @property
    def confiavel(self) -> bool:
        return self.n >= 300

    def z(self, eixo: str, valor: float) -> float:
        """Padroniza dentro da coorte. E o passo que impede o modelo de
        aprender 'de qual plataforma veio' em vez de engajamento."""
        sd = self.desvio.get(eixo, 0.0)
        return 0.0 if sd <= 0 else (valor - self.media.get(eixo, 0.0)) / sd


@dataclass(frozen=True)
class Engajamento:
    recencia: Eixo
    frequencia: Eixo | None
    profundidade: Eixo | None
    eventos: int
    confiavel: bool
    retencao_esperada: float | None = None

    def eixos(self) -> tuple:
        return tuple(e for e in (self.recencia, self.frequencia, self.profundidade) if e)

    def alertas(self) -> list:
        """Eixos em faixa desfavoravel. Lista vazia = nada a fazer."""
        if not self.confiavel:
            return []
        return [e.nome for e in self.eixos() if not e.bom and e.faixa == 'baixo']


def sessoes(instantes_seg, gap: int = GAP_SESSAO_SEG) -> int:
    """Conta sessoes numa lista de timestamps em segundos."""
    t = sorted(instantes_seg)
    if not t:
        return 0
    return 1 + sum(1 for a, b in zip(t, t[1:]) if b - a > gap)


def trajetoria(dias_por_janela, limiar: float = 0.25) -> str:
    """Forma da atividade: 'caiu', 'estavel' ou 'subiu'.

    `dias_por_janela` = dias ativos em cada sub-janela, da mais antiga para a
    mais recente (ex.: [dias 0-9, dias 10-19, dias 20-29]).

    E DESCRICAO, NAO PREDICAO. Nao somar ao score: recencia + frequencia ja
    contem esta informacao (+0,003 de AUC). Serve para dizer ao professor
    POR QUE o aluno esta em risco, numa forma que ele possa agir.

    So faz sentido comparada entre alunos de MESMO TOTAL de atividade. Entre
    alunos de mesma recencia ela se inverte - ver o cabecalho do modulo.

    O LIMIAR PRESSUPOE JANELAS DE ~10 DIAS. `incl` e a variacao media POR
    JANELA, entao o mesmo padrao muda de rotulo conforme quantas janelas voce
    passa: [8, 1] da inclinacao -7,0 e [8, 6, 4, 1] da -2,33, para a mesma
    queda de 8 para 1. Os numeros do cabecalho foram medidos com TRES janelas
    de 10 dias; com outra divisao, remeca o limiar.

    O limiar de 0,25 dia por janela e ESTRITO: [5, 5, 4] ja e "caiu". E a regra
    exata sob a qual os numeros do cabecalho foram medidos, e por isso ela e o
    padrao - mas "estavel" fica sendo uma faixa estreita (no EdNet, 633 alunos
    contra 2.640 em "caiu" dentro da mesma recencia). Se a sua janela for
    menor, afrouxe `limiar` e remeca.
    """
    j = [float(x) for x in dias_por_janela]
    if len(j) < 2:
        raise ValueError('precisa de ao menos duas sub-janelas')
    if len(j) > 6:
        raise ValueError(f'{len(j)} sub-janelas e demais: o limiar foi medido '
                         'com tres janelas de 10 dias e nao transfere para '
                         'janelas muito curtas')
    incl = (j[-1] - j[0]) / (len(j) - 1)
    return 'caiu' if incl < -limiar else ('subiu' if incl > limiar else 'estavel')


def _faixa_por_corte(valor: float, cortes) -> str:
    lo, hi = cortes
    return 'baixo' if valor <= lo else ('medio' if valor <= hi else 'alto')


def calibrar(coorte, janela_dias: int = JANELA_DIAS,
             janela_recente: int = JANELA_RECENTE) -> Calibracao:
    """Constroi faixas e tabela de retencao a partir da sua propria coorte.

    `coorte` e uma sequencia de dicts, um por aluno, com pelo menos:
        dias_recentes  - dias ativos nos ultimos `janela_recente` dias
        dias_ativos    - dias ativos na janela inteira
        voltou         - 1 se continuou depois da janela, 0 se nao
    e opcionalmente `profundidade_seg`.

    Precisa de >=300 alunos com desfecho observado para ser confiavel.
    """
    linhas = [dict(c) for c in coorte]
    if not linhas:
        raise ValueError('coorte vazia')
    faltando = {'dias_recentes', 'dias_ativos'} - set(linhas[0])
    if faltando:
        raise ValueError(f'cada aluno da coorte precisa de {sorted(faltando)}; '
                         f'recebi {sorted(linhas[0])}')
    if any(l['dias_recentes'] != l['dias_recentes'] for l in linhas):
        raise ValueError('dias_recentes com NaN na coorte')
    cortes = {}
    for nome, chave, transf in (('frequencia', 'dias_ativos', lambda v: v / janela_dias),
                                ('profundidade', 'profundidade_seg', lambda v: log(1 + max(v, 0.0)))):
        vals = sorted(transf(l[chave]) for l in linhas if l.get(chave) is not None)
        if len(vals) >= 100:
            cortes[nome] = (vals[len(vals) // 3], vals[2 * len(vals) // 3])
    ret = {}
    com_desfecho = [l for l in linhas if l.get('voltou') is not None]
    for k in range(janela_recente + 1):
        sel = [l['voltou'] for l in com_desfecho if int(l['dias_recentes']) == k]
        if len(sel) >= 30:
            ret[k] = sum(sel) / len(sel)

    # media e desvio da coorte - sem isto o ordenador comum nao se aplica
    media, desvio = {}, {}
    for eixo, chave, transf in (('recencia', 'dias_recentes', float),
                                ('frequencia', 'dias_ativos', lambda v: v / janela_dias)):
        vals = [transf(l[chave]) for l in linhas if l.get(chave) is not None]
        if len(vals) >= 30:
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            media[eixo], desvio[eixo] = mu, var ** 0.5

    cal = Calibracao(cortes=cortes, retencao=ret, media=media, desvio=desvio,
                     n=len(com_desfecho))

    # limiares de risco para alertar os X% piores DESTA coorte
    limiar, efetiva = {}, {}
    if media and desvio and com_desfecho:
        rs = [1.0 - ordenar(int(l['dias_recentes']), l.get('dias_ativos', 0), cal, janela_dias)
              for l in linhas]
        ordenados = sorted(rs)
        for taxa in (0.05, 0.10, 0.20, 0.30):
            i = int((1 - taxa) * (len(ordenados) - 1))
            lim = ordenados[i]
            limiar[taxa] = lim
            # quantos DE FATO passam nesse limiar - empates fazem estourar
            efetiva[taxa] = sum(1 for r in rs if r >= lim) / len(rs)
    return Calibracao(cortes=cortes, retencao=ret, media=media, desvio=desvio,
                      limiar=limiar, taxa_efetiva=efetiva, n=len(com_desfecho))


def ordenar(dias_recentes: int, dias_ativos: int, calibracao: Calibracao,
            janela_dias: int = JANELA_DIAS) -> float:
    """Score de 0 a 1 - chance relativa de continuar, pelo ordenador comum.

    Treinado nas duas bases de referencia JUNTAS, sobre features padronizadas
    dentro de cada coorte. Empata com o modelo especifico de cada base
    (EdNet 0,856 vs 0,854; OULAD 0,862 vs 0,862) e ordena bem numa base que
    nunca viu (0,842 a 0,865).

    NAO e probabilidade calibrada. Serve para ORDENAR e para cortar nos
    limiares da propria coorte (`Calibracao.limiar`). O nivel absoluto nao
    transfere entre plataformas - exportar o limiar de uma base para outra da
    erro de calibracao de 0,36 a 0,57.
    """
    if not calibracao.desvio:
        raise ValueError('ordenar() exige calibracao com media/desvio da coorte; '
                         'rode calibrar() primeiro')
    from math import exp
    zr = calibracao.z('recencia', float(dias_recentes))
    zf = calibracao.z('frequencia', dias_ativos / max(janela_dias, 1))
    logit = (PESOS['intercepto'] + PESOS['recencia'] * zr
             + PESOS['frequencia'] * zf)
    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + exp(-logit))


def risco(dias_recentes: int, dias_ativos: int, calibracao: Calibracao,
          taxa_alerta: float = 0.10, janela_dias: int = JANELA_DIAS) -> bool:
    """Este aluno esta entre os `taxa_alerta` de maior risco DESTA coorte?

    Usar isto, e nao um limiar fixo de 0,5. Medido no OULAD: com limiar 0,5 o
    modelo alerta ninguem e acerta 92,9% - a mesma coisa que nao ter modelo.

    ATENCAO: a taxa pedida NAO e a taxa que dispara. `dias_recentes` e contagem
    discreta e no EdNet 67% dos alunos tem zero - muitos empatam no limiar e o
    alerta estoura. Medido, pedindo 20%:

        EdNet   dispara em 27,5%
        OULAD   dispara em 20,8%

    Consulte `calibracao.taxa_efetiva[taxa]` ANTES de dimensionar a operacao;
    ela traz quantos de fato passam, medido na sua propria coorte.
    """
    if taxa_alerta not in calibracao.limiar:
        raise ValueError(f'limiar para {taxa_alerta:.0%} nao calibrado; '
                         f'disponiveis: {sorted(calibracao.limiar)}')
    r = 1.0 - ordenar(dias_recentes, dias_ativos, calibracao, janela_dias)
    return r >= calibracao.limiar[taxa_alerta]


def medir(dias_recentes: int, dias_ativos: int, eventos: int,
          segundos_profundidade: float | None = None,
          calibracao: Calibracao | None = None,
          janela_dias: int = JANELA_DIAS) -> Engajamento:
    """Eixos a partir do que a plataforma ja registra.

    dias_recentes  - dias distintos com atividade nos ultimos JANELA_RECENTE dias
    dias_ativos    - dias distintos com atividade na janela inteira
    eventos        - total de eventos (respostas, cliques) na janela
    segundos_profundidade - tempo mediano em material de conteudo apos erro;
                     None quando a plataforma nao instrumenta isso
    calibracao     - faixas da sua coorte; sem ela, so a recencia tem faixa
    """
    rec = Eixo('recencia', float(dias_recentes),
               _faixa_por_corte(dias_recentes, CORTE_RECENCIA))

    freq = prof = None
    if calibracao and 'frequencia' in calibracao.cortes:
        v = dias_ativos / max(janela_dias, 1)
        freq = Eixo('frequencia', v, _faixa_por_corte(v, calibracao.cortes['frequencia']))
    if calibracao and segundos_profundidade is not None and 'profundidade' in calibracao.cortes:
        v = log(1 + max(segundos_profundidade, 0.0))
        prof = Eixo('profundidade', v, _faixa_por_corte(v, calibracao.cortes['profundidade']))

    esperada = None
    if calibracao and calibracao.confiavel:
        esperada = calibracao.retencao.get(int(dias_recentes))

    return Engajamento(recencia=rec, frequencia=freq, profundidade=prof,
                       eventos=eventos, confiavel=eventos >= MIN_EVENTOS,
                       retencao_esperada=esperada)


if __name__ == '__main__':
    print('forma da curva de recencia nas duas bases de referencia:')
    print(f'  {"dias ativos nos ultimos 10":<28} {"EdNet":>8} {"OULAD":>8}')
    for k in range(11):
        print(f'  {k:>26}   {REFERENCIA["EdNet"][k]:>7.1%} {REFERENCIA["OULAD"][k]:>8.1%}')
    print('\n  monotonica nas duas, com niveis muito diferentes.')
    print('  e por isso que o nivel tem de sair da sua coorte, nao daqui.\n')

    casos = [('sumiu ha duas semanas', dict(dias_recentes=0, dias_ativos=9, eventos=300)),
             ('presente e constante',  dict(dias_recentes=8, dias_ativos=20, eventos=300)),
             ('voltou agora',          dict(dias_recentes=3, dias_ativos=4, eventos=60)),
             ('poucos dados',          dict(dias_recentes=1, dias_ativos=2, eventos=12))]
    for nome, kw in casos:
        e = medir(**kw)
        print(f'{nome:>24}: recencia {e.recencia.faixa:>5} | alertas={e.alertas()} '
              f'| confiavel={e.confiavel}')
