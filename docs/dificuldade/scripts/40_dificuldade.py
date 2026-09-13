"""MELHORIA 1: cobertura do intervalo de dificuldade em n alto.

Defeito medido: o intervalo de 90% cobre 89,3% em n=30, mas so 77,6% em n=200.
Causa: o modelo binomial supoe TAXA FIXA. A dificuldade real de uma questao
varia entre coortes, e essa variacao nao encolhe com n.

Correcao: estimar a variancia ENTRE COORTES (sigma^2) do proprio corpus e
soma-la ao desvio do posterior. O intervalo passa a cobrir a taxa que a
questao tera numa coorte NOVA, que e o que o consumidor quer saber.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(20260913)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct'])
print(f'{len(d):,} respostas | {d.qidx.nunique():,} questoes', flush=True)

# --- particiona ALUNOS em duas coortes (nao respostas: coorte = conjunto de alunos)
us = d.user.unique(); rng.shuffle(us)
lado = pd.Series(np.arange(len(us)) % 2, index=us)
d['lado'] = d.user.map(lado)
g = d.groupby(['qidx','lado']).correct.agg(['sum','size']).unstack()
g.columns = ['s0','s1','n0','n1']
G = g.dropna()
G = G[(G.n0 >= 30) & (G.n1 >= 30)]
p0 = G.s0 / G.n0; p1 = G.s1 / G.n1
print(f'{len(G):,} questoes com >=30 respostas nas duas coortes\n')

# --- decomposicao da variancia: total = binomial + entre-coortes
dif = (p0 - p1).values
var_obs = dif.var()
var_bin = (p0 * (1 - p0) / G.n0 + p1 * (1 - p1) / G.n1).mean()
sigma2 = max((var_obs - var_bin) / 2.0, 0.0)
print('=== quanto da diferenca entre coortes NAO e ruido de amostragem? ===')
print(f'  variancia observada de (p0-p1) : {var_obs:.5f}')
print(f'  esperada so por binomial       : {var_bin:.5f}')
print(f'  sobra (2*sigma^2)              : {var_obs-var_bin:.5f}')
print(f'  SIGMA entre coortes            : {np.sqrt(sigma2):.4f}  ({np.sqrt(sigma2)*100:.1f} pontos percentuais)')
print(f'  -> mesmo com n infinito, a taxa de uma questao varia +-{1.645*np.sqrt(sigma2)*100:.1f} pts entre coortes (90%)')

# --- cobertura: intervalo construido na coorte 0, alvo = taxa na coorte 1
A, B = 5.07, 2.08
Z = 1.645
def cobertura(sig2, ns):
    out = []
    for lo, hi in ns:
        m = (G.n0 >= lo) & (G.n0 < hi)
        if m.sum() < 100: continue
        a = G.s0[m] + A; b = (G.n0[m] - G.s0[m]) + B
        mu = a / (a + b)
        var_post = a * b / ((a + b) ** 2 * (a + b + 1))
        sd = np.sqrt(var_post + sig2)
        alvo = p1[m]
        cob = ((alvo >= mu - Z * sd) & (alvo <= mu + Z * sd)).mean()
        larg = (2 * Z * sd).mean()
        out.append((f'{lo}-{hi-1}', int(m.sum()), cob, larg))
    return out

NS = [(30,50),(50,100),(100,200),(200,400),(400,100000)]
print('\n=== cobertura do intervalo de 90% (alvo: taxa numa coorte NOVA) ===')
print(f'  {"n":<12} {"questoes":>9} {"ANTES":>8} {"largura":>9} {"DEPOIS":>8} {"largura":>9}')
a0 = cobertura(0.0, NS); a1 = cobertura(sigma2, NS)
for (lab, k, c0, l0), (_, _, c1, l1) in zip(a0, a1):
    print(f'  {lab:<12} {k:>9,} {c0:>8.1%} {l0:>9.3f} {c1:>8.1%} {l1:>9.3f}')
print(f'\n  SIGMA2 = {sigma2:.6f}   (constante a por no modulo)')
