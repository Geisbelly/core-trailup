"""Cobertura por faixa de n, com o ruido do alvo contabilizado, e o efeito
de somar a variacao entre coortes reais (sigma=0,0153)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M2 = HERE.replace('/engaj', '/m2')
A, B, Z = 5.07, 2.08, 1.645
SIG2 = 0.0153 ** 2
rng = np.random.RandomState(20260913)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct'])
us = d.user.unique(); rng.shuffle(us)
d['lado'] = d.user.map(pd.Series(np.arange(len(us)) % 2, index=us))
g = d.groupby(['qidx','lado']).correct.agg(['sum','size']).unstack()
g.columns = ['s0','s1','n0','n1']; G = g.dropna()
G = G[(G.n0 >= 20) & (G.n1 >= 50)]
alvo = (G.s1 / G.n1)
a = G.s0 + A; b = (G.n0 - G.s0) + B
mu = a / (a + b); vpost = a * b / ((a + b) ** 2 * (a + b + 1))
valvo = alvo * (1 - alvo) / G.n1                       # ruido do proprio alvo

print('=== cobertura por faixa de n (alvo = taxa numa coorte de validacao) ===')
print(f'  {"n":<12} {"questoes":>9} {"posterior":>11} {"+ruido alvo":>12} {"+ruido+sigma":>13}')
for lo, hi in [(20,30),(30,50),(50,100),(100,200),(200,400),(400,10**9)]:
    m = ((G.n0 >= lo) & (G.n0 < hi)).values
    if m.sum() < 100: continue
    row = []
    for v in [vpost[m], vpost[m] + valvo[m], vpost[m] + valvo[m] + SIG2]:
        sd = np.sqrt(v)
        row.append(((alvo[m] >= mu[m] - Z*sd) & (alvo[m] <= mu[m] + Z*sd)).mean())
    print(f'  {f"{lo}-{hi-1}":<12} {int(m.sum()):>9,} {row[0]:>11.1%} {row[1]:>12.1%} {row[2]:>13.1%}')
print('\n  -> a queda em n alto era em boa parte RUIDO DO ALVO, nao defeito do estimador')

print('\n=== o que o professor pergunta: "quanto minha turma de m vai acertar?" ===')
print(f'  {"turma":<8} {"so posterior":>14} {"largura":>9} {"preditivo":>11} {"largura":>9}')
for mm in (15, 30, 60, 120):
    sim = rng.binomial(mm, np.clip(alvo.values, 0, 1)) / mm
    for lab, v in [('post', vpost.values), ('pred', vpost.values + mu.values*(1-mu.values)/mm + SIG2)]:
        sd = np.sqrt(v)
        c = ((sim >= mu.values - Z*sd) & (sim <= mu.values + Z*sd)).mean()
        l = (2*Z*sd).mean()
        if lab == 'post': c0, l0 = c, l
        else: c1, l1 = c, l
    print(f'  {mm:<8} {c0:>14.1%} {l0:>9.3f} {c1:>11.1%} {l1:>9.3f}')
