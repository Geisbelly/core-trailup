"""Validacao com as RESPOSTAS REAIS, nao reamostragem binomial.

A simulacao anterior sorteava acertos com rng.binomial(n, taxa) - isso impoe
independencia, entao dependencia entre respostas (mesmo aluno reencontrando a
questao, efeito de coorte) nunca apareceria. Aqui uso as primeiras n respostas
de verdade.
"""
import numpy as np, pandas as pd
from scipy.stats import beta as B, betabinom
M2='/caminho/para/os/intermediarios'
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','ts'])
df=df.sort_values('ts')
A0,B0=5.07,2.08
print('=== cobertura com respostas REAIS (primeiras n por questao) ===')
print(f"{'n':>5} │ {'questoes':>9} │ {'nominal 80%':>12} {'nominal 90%':>12} {'nominal 95%':>12}")
print('─'*60)
for n in (10,21,30,50,100):
    g=df.groupby('qidx')
    ok=g.size()>=n+30
    qs=ok[ok].index
    sub=df[df.qidx.isin(qs)]
    prim=sub.groupby('qidx').head(n).groupby('qidx').correct.agg(['sum','size'])
    rest=sub.groupby('qidx').tail(-n).groupby('qidx').correct.agg(['sum','size'])
    j=prim.join(rest,lsuffix='_a',rsuffix='_b')
    j=j[j.size_b>=30]
    a=j.sum_a+A0; b=(j.size_a-j.sum_a)+B0
    cov=[]
    for niv in (0.8,0.9,0.95):
        lo=betabinom.ppf((1-niv)/2,j.size_b,a,b); hi=betabinom.ppf(1-(1-niv)/2,j.size_b,a,b)
        cov.append(((j.sum_b>=lo)&(j.sum_b<=hi)).mean())
    print(f'{n:>5} │ {len(j):>9,} │ {cov[0]:>11.1%} {cov[1]:>12.1%} {cov[2]:>12.1%}')
print('\n(as primeiras n respostas sao as mais ANTIGAS: se a populacao que chega')
print(' primeiro difere da que chega depois, isso aparece aqui e nao na simulacao)')

print('\n=== e se corrigir o n pelo numero de alunos DISTINTOS? ===')
print(f"{'n':>5} │ {'sem correcao':>13} {'com n efetivo':>14}")
print('─'*36)
for n in (21,30,50,100):
    g=df.groupby('qidx'); ok=g.size()>=n+30; qs=ok[ok].index
    sub=df[df.qidx.isin(qs)]
    p=sub.groupby('qidx').head(n)
    prim=p.groupby('qidx').agg(s=('correct','sum'),n=('correct','size'),d=('user','nunique'))
    rest=sub.groupby('qidx').tail(-n).groupby('qidx').correct.agg(['sum','size'])
    j=prim.join(rest); j=j[j['size']>=30]
    for corr,lab in ((False,''),(True,'')):
        fat=(j.d/j.n) if corr else 1.0
        a=j.s*fat+A0; b=(j.n-j.s)*fat+B0
        lo=betabinom.ppf(.05,j['size'],a,b); hi=betabinom.ppf(.95,j['size'],a,b)
        c=((j['sum']>=lo)&(j['sum']<=hi)).mean()
        if not corr: c0=c
        else: print(f'{n:>5} │ {c0:>12.1%} {c:>13.1%}')
