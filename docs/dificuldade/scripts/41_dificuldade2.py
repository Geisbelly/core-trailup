"""Cobertura do intervalo de dificuldade, medida sobre o objeto certo.

Duas perguntas diferentes, dois intervalos diferentes:
  (a) POSTERIOR  cobre a taxa LATENTE da questao. Alvo tem de ser quase sem
      ruido -> usar so questoes cuja metade de validacao e grande.
  (b) PREDITIVO  cobre a taxa OBSERVADA numa proxima turma de tamanho m.
      Precisa somar o ruido binomial dessa turma.

E a variacao entre coortes REAIS (nao divisao aleatoria): alunos antigos vs
recentes, que e a unica forma de coorte que o EdNet permite construir.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M2 = HERE.replace('/engaj', '/m2')
A, B, Z = 5.07, 2.08, 1.645
rng = np.random.RandomState(20260913)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts'])

us = d.user.unique(); rng.shuffle(us)
d['lado'] = d.user.map(pd.Series(np.arange(len(us)) % 2, index=us))
g = d.groupby(['qidx','lado']).correct.agg(['sum','size']).unstack()
g.columns = ['s0','s1','n0','n1']; G = g.dropna()

print('=' * 74)
print('(a) POSTERIOR cobre a taxa LATENTE? (alvo: metade com n>=400)')
print('=' * 74)
V = G[(G.n1 >= 400) & (G.n0 >= 20)]
alvo = (V.s1 / V.n1)
print(f'  {len(V):,} questoes | ruido do alvo: sd medio {np.sqrt((alvo*(1-alvo)/V.n1)).mean():.4f}')
print(f'\n  {"n da estimativa":<18} {"questoes":>9} {"cobertura":>10} {"largura":>9}')
for lo, hi in [(20,30),(30,50),(50,100),(100,200),(200,400),(400,10**9)]:
    m = (V.n0 >= lo) & (V.n0 < hi)
    if m.sum() < 80: continue
    a = V.s0[m] + A; b = (V.n0[m] - V.s0[m]) + B
    mu = a / (a + b); sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    cob = ((alvo[m] >= mu - Z * sd) & (alvo[m] <= mu + Z * sd)).mean()
    print(f'  {f"{lo}-{hi-1}":<18} {int(m.sum()):>9,} {cob:>10.1%} {(2*Z*sd).mean():>9.3f}')

print('\n' + '=' * 74)
print('(b) PREDITIVO cobre a taxa de uma TURMA NOVA de 30 alunos?')
print('=' * 74)
# reamostra turmas de 30 respostas da metade 1
W = G[(G.n0 >= 20) & (G.n1 >= 60)]
p1 = (W.s1 / W.n1).values; n0 = W.n0.values; s0 = W.s0.values
m30 = rng.binomial(30, np.clip(p1, 0, 1)) / 30.0
a = s0 + A; b = (n0 - s0) + B
mu = a / (a + b); var_post = a * b / ((a + b) ** 2 * (a + b + 1))
print(f'  {len(W):,} questoes | turma simulada de 30 alunos')
print(f'\n  {"intervalo":<28} {"cobertura":>10} {"largura":>9}')
for lab, sd in [('so posterior (o que existe)', np.sqrt(var_post)),
                ('preditivo (+ ruido da turma)', np.sqrt(var_post + mu * (1 - mu) / 30.0))]:
    cob = ((m30 >= mu - Z * sd) & (m30 <= mu + Z * sd)).mean()
    print(f'  {lab:<28} {cob:>10.1%} {(2*Z*sd).mean():>9.3f}')

print('\n' + '=' * 74)
print('(c) COORTES REAIS: alunos antigos vs recentes')
print('=' * 74)
prim = d.groupby('user').ts.min()
corte = prim.median()
d['coorte'] = (d.user.map(prim) > corte).astype(int)
h = d.groupby(['qidx','coorte']).correct.agg(['sum','size']).unstack().dropna()
h.columns = ['sa','sb','na','nb']
H = h[(h.na >= 50) & (h.nb >= 50)]
pa = H.sa / H.na; pb = H.sb / H.nb
var_obs = (pa - pb).var()
var_bin = (pa * (1 - pa) / H.na + pb * (1 - pb) / H.nb).mean()
sig2 = max((var_obs - var_bin) / 2, 0.0)
print(f'  {len(H):,} questoes | variancia observada {var_obs:.5f} | binomial {var_bin:.5f}')
print(f'  SIGMA entre coortes reais: {np.sqrt(sig2):.4f} ({np.sqrt(sig2)*100:.1f} pontos)')
print(f'  correlacao das taxas entre as duas coortes: {pa.corr(pb):.3f}')
