"""Fuzz: toda funcao publica com entrada hostil. O contrato e simples -
ou levanta erro claro, ou devolve valor SANO. Nunca NaN, inf, nem fora de faixa.

Roda SEM dataset. Primeira execucao achou 21 problemas; sobram 3, todos a
sentinela NaN intencional de `discriminacao` (veredito 'indeterminado'), que
agora e barrada em `confirmar()`.
"""
import os, sys, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
from trailup_core import dificuldade as DIF, ritmo as RIT, tempo as TMP, dominio as DOM
from trailup_core import chute as CHU, revisao as REV, discriminacao as DSC
from trailup_core import engajamento as ENG, gate as GAT, pre_avaliacao as PA

NAN, INF = float('nan'), float('inf')
PROBLEMAS = []
def checa(nome, fn, *a, **k):
    try:
        r = fn(*a, **k)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return  # recusar e comportamento correto
    except Exception as e:
        PROBLEMAS.append(f'{nome}{a}: excecao inesperada {type(e).__name__}: {e}'); return
    vals = []
    if isinstance(r, float): vals = [r]
    elif hasattr(r, '__dataclass_fields__'):
        vals = [getattr(r, f) for f in r.__dataclass_fields__
                if isinstance(getattr(r, f), float)]
    for v in vals:
        if math.isnan(v): PROBLEMAS.append(f'{nome}{a}: devolveu NaN')
        elif math.isinf(v): PROBLEMAS.append(f'{nome}{a}: devolveu infinito')

print('=== dificuldade ===')
for k, n in [(-1,10),(5,-10),(0,0),(10,5),(NAN,10),(10,NAN),(INF,INF),(10**9,10**9)]:
    checa('estimar', DIF.estimar, k, n)
    checa('prever_turma', DIF.prever_turma, k, n, 30)
for m in (-5, 0, NAN, INF, 10**9):
    checa('prever_turma(turma)', DIF.prever_turma, 20, 30, m)
checa('derivar_prior vazio', DIF.derivar_prior, [], [])
checa('derivar_prior NaN', DIF.derivar_prior, [NAN, 0.5], [10, 10])
checa('derivar_prior n=0', DIF.derivar_prior, [0.3, 0.7], [0, 0])

print('=== ritmo / tempo ===')
for lat, n in [(-1,10),(0,10),(NAN,10),(INF,10),(10,-1),(10,0)]:
    checa('ritmo', RIT.ritmo, lat, n)
    checa('esperado', TMP.esperado, lat, n)
checa('esperado_de_amostra vazio', TMP.esperado_de_amostra, [])
checa('esperado_de_amostra NaN', TMP.esperado_de_amostra, [NAN, 10.0])
checa('esperado_de_amostra inf', TMP.esperado_de_amostra, [INF, 10.0])
checa('duracao vazia', TMP.duracao_prevista, [])
checa('duracao NaN', TMP.duracao_prevista, [NAN, 10.0])
checa('duracao negativa', TMP.duracao_prevista, [-10.0, 10.0])
for n in (-1, 0, NAN, INF): checa('precisao', TMP.precisao, n)
e = TMP.esperado(20.0, 50)
for s in (-1.0, 0.0, NAN, INF): checa('quao_lento', TMP.quao_lento, s, e)

print('=== dominio / gate / chute ===')
for q, a, n in [(-1,5,10),(2,5,10),(NAN,5,10),(0.5,-5,10),(0.5,50,10),(0.5,5,-10),(0.5,5,0)]:
    checa('dominio', DOM.dominio, q, a, n)
for n, a in [(-1,0),(0,0),(NAN,0),(10,-5),(10,50)]:
    checa('incerteza', DOM.incerteza, n, a)
    checa('confianca', DOM.confianca, n, a)
for mg in (-1.0, 0.0, 1.0, 2.0, NAN): checa('dominio(media_global)', DOM.dominio, 0.5, 5, 10, mg)
checa('gate vazio', GAT.risco, [], 0.5, 0.5)
for ag in (-1.0, 2.0, NAN, INF): checa('gate(acerto_global)', GAT.risco, [True]*6, ag, 0.5)
checa('gate seq gigante', GAT.risco, [True]*100000, 0.5, 0.5)
for s, m in [(-1.0,10.0),(NAN,10.0),(10.0,0.0),(10.0,NAN),(10.0,-5.0)]:
    checa('foi_chute', CHU.foi_chute, s, False, m, 60)
for c, r in [(-1,10),(20,10),(0,0),(5,0)]: checa('taxa_chute', CHU.taxa_chute, c, r)

print('=== revisao / discriminacao / engajamento ===')
for d in (-10.0, 0.0, NAN, INF, 10**9):
    checa('retencao', REV.retencao, d, True)
    checa('esquecimento', REV.esquecimento, d, True)
for alvo in (-1.0, 0.0, 1.0, 2.0, NAN): checa('dias_ate_revisar', REV.dias_ate_revisar, True, alvo)
checa('discriminacao vazia', DSC.discriminacao, [])
checa('discriminacao NaN', DSC.discriminacao, [(True, NAN)] * 50)
checa('discriminacao constante', DSC.discriminacao, [(True, 0.5)] * 50)
checa('confirmar NaN', DSC.confirmar, NAN, 0.1)
checa('sessoes vazio', ENG.sessoes, [])
checa('sessoes NaN', ENG.sessoes, [0, NAN, 10])
for j in ([], [5], [NAN, 1], [INF, 1]): checa('trajetoria', ENG.trajetoria, j)
for dr, da, ev in [(-1,10,100),(100,10,100),(3,-5,100),(3,10,-1),(NAN,10,100)]:
    checa('medir', ENG.medir, dr, da, ev)
checa('calibrar vazia', ENG.calibrar, [])
checa('calibrar sem campos', ENG.calibrar, [{'voltou': 1}])
checa('calibrar NaN', ENG.calibrar, [{'dias_recentes': NAN, 'dias_ativos': 5, 'voltou': 1}] * 400)

print('=== pre_avaliacao ===')
checa('stopwords vazio', PA.stopwords, [])
checa('preparar vazio', PA.preparar, '')
checa('preparar None-ish', PA.preparar, '   ')
ref = PA.preparar('inflacao e o aumento dos precos na economia')
for r in ('', '   ', 'a', 'x' * 100000): checa('avaliar', PA.avaliar, r, ref)
checa('triar vazio', PA.triar, [], ref)
checa('triar com vazios', PA.triar, ['', '', ''], ref)

ESPERADOS = 3   # a sentinela NaN de discriminacao, documentada
print(f'\n{len(PROBLEMAS)} problemas (esperados: {ESPERADOS})')
for p in PROBLEMAS: print('  ' + p[:120])
sys.exit(1 if len(PROBLEMAS) > ESPERADOS else 0)
