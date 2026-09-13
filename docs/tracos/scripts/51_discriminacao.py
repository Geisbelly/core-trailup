"""MELHORIA: discriminacao. A precisao de marcacao e 18-27%. Melhora exigindo
CONFIRMACAO entre duas metades independentes de alunos?

Ja medi que 2,96% das questoes sao negativas numa metade e so 0,53% nas duas.
O que nunca medi: a PRECISAO de cada regra contra um criterio externo.

Criterio externo possivel no EdNet: a questao ser negativa numa TERCEIRA
particao, independente das duas usadas para marcar.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import wilson
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(11)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct'])
us = d.user.unique(); rng.shuffle(us)
d['p'] = d.user.map(pd.Series(np.arange(len(us)) % 3, index=us))   # TRES particoes
print(f'{len(d):,} respostas | 3 particoes de alunos', flush=True)

def disc(sub):
    """correlacao item-total, com a habilidade calculada EXCLUINDO a questao"""
    tot = sub.groupby('user').correct.agg(['sum','size'])
    sub = sub.join(tot, on='user')
    hab = (sub['sum'] - sub.correct) / (sub['size'] - 1).clip(lower=1)
    sub = sub.assign(hab=hab)
    g = sub.groupby('qidx')
    def r(x):
        if len(x) < 40: return np.nan
        a = x.correct.values.astype(float); b = x.hab.values
        sa, sb = a.std(), b.std()
        return np.nan if sa == 0 or sb == 0 else ((a-a.mean())*(b-b.mean())).mean()/(sa*sb)
    return g.apply(r)

D = pd.DataFrame({f'p{k}': disc(d[d.p == k]) for k in (0,1,2)}).dropna()
print(f'{len(D):,} questoes com >=40 respostas nas tres particoes')
print(f'  discriminacao media: {D.mean().mean():.3f}')
print(f'  correlacao entre particoes: {D.p0.corr(D.p1):.3f} / {D.p0.corr(D.p2):.3f} / {D.p1.corr(D.p2):.3f}\n')

print('=== precisao de cada regra (alvo: negativa na 3a particao, nao usada) ===')
alvo = (D.p2 < 0)
print(f'  prevalencia do alvo: {alvo.mean():.2%}  ({int(alvo.sum())} questoes)')
print(f'\n  {"regra (usa p0 e p1)":<40} {"marca":>7} {"precisao":>10} {"IC95":>16} {"lift":>6}')
regras = [
    ('negativa em p0', D.p0 < 0),
    ('negativa em p0 OU p1', (D.p0 < 0) | (D.p1 < 0)),
    ('negativa em p0 E p1', (D.p0 < 0) & (D.p1 < 0)),
    ('media(p0,p1) < 0', (D.p0 + D.p1) / 2 < 0),
    ('media(p0,p1) < -0,05', (D.p0 + D.p1) / 2 < -0.05),
    ('media(p0,p1) < -0,10', (D.p0 + D.p1) / 2 < -0.10),
]
for lab, m in regras:
    if m.sum() < 5:
        print(f'  {lab:<40} {m.sum():>7} {"-":>10}'); continue
    k = int(alvo[m].sum()); n = int(m.sum())
    lo, hi = wilson(k, n)
    print(f'  {lab:<40} {n:>7,} {k/n:>10.1%} [{lo:.1%},{hi:.1%}] {(k/n)/alvo.mean():>5.1f}x')

print('\n=== e se o alvo for "entre as 1% piores na 3a particao"? ===')
lim = D.p2.quantile(0.01); alvo2 = D.p2 <= lim
print(f'  limiar da 3a particao: {lim:.3f} | prevalencia {alvo2.mean():.2%}')
print(f'\n  {"regra":<40} {"marca":>7} {"precisao":>10} {"lift":>6}')
for lab, m in regras:
    if m.sum() < 5: continue
    k = int(alvo2[m].sum()); n = int(m.sum())
    print(f'  {lab:<40} {n:>7,} {k/n:>10.1%} {(k/n)/alvo2.mean():>5.1f}x')
