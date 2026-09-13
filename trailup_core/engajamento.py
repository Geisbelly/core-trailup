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

LIMITE MEDIDO - LEIA ANTES DE EXIBIR
    Nenhum eixo prediz GANHO de acerto entre os dois meses (Spearman -0,04 a
    +0,03, n=5.523 no EdNet). Engajamento prediz PERMANENCIA, nao APRENDIZADO.

ONDE NAO FUNCIONOU
    ARES (200 universitarios, leitura em L2 obrigatoria, 6 semanas): retencao
    de 87,7% e AUC 0,49-0,57 em todos os eixos. Quando a participacao e
    compulsoria e quase todo mundo volta, nao ha variacao a prever. Este modulo
    pressupoe continuacao VOLUNTARIA.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log

JANELA_DIAS = 30        # janela de observacao
JANELA_RECENTE = 10     # ultimos dias da janela que formam a recencia
GAP_SESSAO_SEG = 1800   # inatividade que separa duas sessoes
MIN_EVENTOS = 30        # abaixo disso os eixos sao ruido

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
    """Faixas de uma coorte propria. Sem isto, so a recencia tem faixa."""
    cortes: dict = field(default_factory=dict)      # eixo -> (lim_baixo, lim_medio)
    retencao: dict = field(default_factory=dict)    # dias_recentes -> taxa observada
    n: int = 0

    @property
    def confiavel(self) -> bool:
        return self.n >= 300


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
    return Calibracao(cortes=cortes, retencao=ret, n=len(com_desfecho))


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
