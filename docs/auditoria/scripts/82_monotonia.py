"""Monotonicidade ao longo de TODO o dominio, nao em pontos isolados.

Cada funcao tem direcoes que precisam valer sempre. Testes de ponto passam por
acidente; varrer o dominio denso pega inversao local, plato inesperado e sinal
trocado numa faixa so.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
from trailup_core import dificuldade as DIF, ritmo as RIT, tempo as TMP, dominio as DOM
from trailup_core import chute as CHU, revisao as REV, engajamento as ENG, gate as GAT

FALHAS = []
def cresce(nome, fn, xs, estrito=False):
    v = [fn(x) for x in xs]
    for (a, va), (b, vb) in zip(zip(xs, v), zip(xs[1:], v[1:])):
        if (vb < va) or (estrito and vb == va):
            FALHAS.append(f'{nome}: {a}->{b} deu {va:.4f}->{vb:.4f}'); return
    print(f'  ok  {nome}')
def decresce(nome, fn, xs, estrito=False):
    cresce(nome, lambda x: -fn(x), xs, estrito)

print('=== dificuldade ===')
cresce('estimar: taxa cresce com acertos', lambda k: DIF.estimar(k, 100).taxa, range(0, 101), estrito=True)
decresce('estimar: largura cai com n', lambda n: DIF.estimar(int(.7*n), n).largura, range(5, 500, 5))
cresce('prever_turma: largura cai com a turma',
       lambda m: -DIF.prever_turma(20, 30, m).largura, range(5, 200, 5))
for n in (10, 30, 100, 400):
    decresce(f'prever_turma >= estimar (n={n})',
             lambda m, n=n: DIF.prever_turma(int(.7*n), n, m).largura - DIF.estimar(int(.7*n), n).largura,
             range(5, 200, 5))

print('\n=== tempo / ritmo ===')
decresce('tempo.precisao melhora com n', TMP.precisao, range(1, 200))
cresce('quao_lento cresce com o tempo', lambda s: TMP.quao_lento(float(s), TMP.esperado(20.0, 50)), range(1, 300), estrito=True)
cresce('ritmo.confianca cresce com n', lambda n: RIT.ritmo(15.0, n).confianca, range(5, 300))

print('\n=== dominio ===')
cresce('p cresce com a facilidade da questao', lambda q: DOM.dominio(q/100, 5, 10).p, range(1, 100), estrito=False)
cresce('p cresce com os acertos do aluno', lambda a: DOM.dominio(0.5, a, 40).p, range(0, 41))
# incerteza depende de n E da taxa; com acertos=int(0,7n) a taxa oscila e a
# monotonia se quebra por discretizacao, nao por defeito. Varre com taxa fixa.
decresce('incerteza cai com n (taxa 1,0)', lambda n: DOM.incerteza(n, n), range(1, 300))
cresce('confianca cresce com n (taxa 1,0)', lambda n: DOM.confianca(n, n), range(1, 300))
cresce('recalibrar preserva a ordem', lambda i: DOM._recalibrar(0.01 + i*0.0098), range(0, 100), estrito=True)

print('\n=== revisao ===')
# _ACERTOU nao e monotonica (sobe em 0,04->1 dia e em 60->180): esta medido e
# documentado no modulo, entao so _ERROU entra na varredura estrita.
decresce('retencao cai com os dias (errou antes)', lambda d: REV.retencao(d, False), [x/2 for x in range(1, 400)])
cresce('esquecimento cresce com os dias (errou antes)', lambda d: REV.esquecimento(d, False), [x/2 for x in range(1, 400)])
cresce('retencao cresce com a proporcao', lambda i: REV.retencao(7, True, i/100), range(1, 100))
decresce('dias_ate_revisar cai com alvo maior', lambda i: REV.dias_ate_revisar(False, 0.30 + i*0.006), range(0, 100))

print('\n=== engajamento / gate / chute ===')
cal = ENG.calibrar([{'dias_recentes': i % 11, 'dias_ativos': (i % 30),
                     'profundidade_seg': float(i % 60), 'voltou': 1 if i % 11 > 4 else 0}
                    for i in range(1200)])
cresce('ordenar cresce com a recencia', lambda k: ENG.ordenar(k, 15, cal), range(0, 11), estrito=True)
cresce('ordenar cresce com dias ativos', lambda d: ENG.ordenar(5, d, cal), range(0, 31))
cresce('gate.sequencial cresce com acertos', lambda k: GAT.sequencial([True]*k + [False]*(30-k)), range(0, 31))
decresce('gate.risco cai com o acerto no topico',
         lambda i: GAT.risco([True]*i + [False]*(20-i), 0.5, 0.5).score, range(0, 21))
decresce('gate.risco cai com o acerto global', lambda i: GAT.risco([False]*10, i/100, 0.5).score, range(1, 100))
cresce('taxa_chute cresce com chutes', lambda c: CHU.taxa_chute(c, 100), range(0, 101), estrito=True)

print(f'\n{len(FALHAS)} violacoes de monotonia')
for f in FALHAS: print('  ' + f)
sys.exit(1 if FALHAS else 0)
