"""Tres afirmacoes que faltavam conferir:
 1. dificuldade: "aproximacao normal x Beta exato, diferenca < 1 ponto"
 2. gate.MIN_RESPOSTAS = 5
 3. engajamento.MIN_EVENTOS = 30
"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import beta as B
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
from trailup_core import dificuldade as DIF
M2 = HERE.replace('/engaj', '/m2')

print('=== 1. aproximacao normal vs Beta exato ===')
A0, B0 = DIF.Prior().a, DIF.Prior().b
print(f'  {"n":<8} {"acertos":<9} {"normal":>16} {"Beta exato":>16} {"dif. max":>10}')
pior = 0.0
for n in (5, 10, 30, 100, 400):
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        k = int(round(n*frac))
        f = DIF.estimar(k, n)
        a, b = k + A0, (n-k) + B0
        lo_e, hi_e = B.ppf(0.05, a, b), B.ppf(0.95, a, b)
        dif = max(abs(f.minimo - lo_e), abs(f.maximo - hi_e))
        pior = max(pior, dif)
        if frac in (0.1, 0.5):
            print(f'  {n:<8} {k:<9} [{f.minimo:.3f}, {f.maximo:.3f}] [{lo_e:.3f}, {hi_e:.3f}] {dif:>10.4f}')
print(f'\n  maior diferenca em toda a grade: {pior:.4f} ({pior*100:.2f} pontos percentuais)')
print(f'  afirmado no modulo: "menor que 1 ponto"  -> {"CONFERE" if pior < 0.01 else "NAO CONFERE"}')

# cobertura real dos dois, com dados
rng = np.random.RandomState(5)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct'])
us = d.user.unique(); rng.shuffle(us); tr = set(us[:len(us)//2])
d['h'] = d.user.isin(tr)
g = d.groupby(['qidx','h']).correct.agg(['sum','size']).unstack().dropna()
g.columns = ['s0','s1','n0','n1']
G = g[(g.n0 >= 20) & (g.n1 >= 200)]
alvo = (G.s1/G.n1).values
print(f'\n  cobertura real ({len(G):,} questoes, alvo = metade com n>=200):')
for lab, usar_beta in [('normal (modulo)', False), ('Beta exato', True)]:
    ok = []
    for s0, n0, al in zip(G.s0.values, G.n0.values, alvo):
        if usar_beta:
            a, b = s0+A0, (n0-s0)+B0
            lo, hi = B.ppf(0.05, a, b), B.ppf(0.95, a, b)
        else:
            f = DIF.estimar(int(s0), int(n0)); lo, hi = f.minimo, f.maximo
        ok.append(lo <= al <= hi)
    print(f'    {lab:<18} {np.mean(ok):.1%}')

print('\n=== 2 e 3. os minimos: MIN_RESPOSTAS do gate e MIN_EVENTOS do engajamento ===')
E = pd.read_parquet(f'{HERE}/eixos.parquet')
print(f'  engajamento: AUC de ordenar() por faixa de eventos na janela')
print(f'  {"eventos":<14} {"alunos":>8} {"AUC da recencia":>17}')
for lo, hi in [(30,50),(50,100),(100,300),(300,1000),(1000,10**9)]:
    m = (E.n >= lo) & (E.n < hi)
    if m.sum() < 300: continue
    print(f'  {f"{lo}-{hi-1}":<14} {int(m.sum()):>8,} {roc_auc_score(E.ret[m].values, E.rec[m].values):>17.3f}')
m = E.n < 30
print(f'  (abaixo de 30 o modulo se recusa a opinar; a base tem {int(m.sum()):,} desses)')
