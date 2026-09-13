"""Estimador final: intervalo com prior empirico Beta(5.1, 2.1), forca 7.2."""
import os, numpy as np, pandas as pd
from scipy.stats import beta as B, betabinom
HERE=os.path.dirname(os.path.abspath(__file__))
q=pd.read_parquet(f'{HERE}/questoes.parquet').reset_index()
rng=np.random.RandomState(20260912)
p=q.verdade.values
esp=np.sqrt(np.mean(p*(1-p)/q.size_b.values)); v=max(p.var()-esp**2,1e-6); mu=p.mean()
k=mu*(1-mu)/v-1; A0,B0=mu*k,(1-mu)*k
print(f'prior empirico: Beta({A0:.2f}, {B0:.2f}) | forca {k:.1f} | media {A0/(A0+B0):.3f}\n')
print('=== cobertura posterior preditiva com o prior empirico ===')
print(f"{'n':>5} │ {'80%':>8} {'90%':>8} {'95%':>8} │ {'largura 90%':>12} {'erro abs':>9}")
print('─'*56)
for n in (5,10,21,30,50,100):
    sub=q[q.size_a>=n]
    ac=rng.binomial(n,np.clip(sub.sum_a/sub.size_a,0,1))
    a=ac+A0; b=(n-ac)+B0
    nB=sub.size_b.values; acB=sub.sum_b.values
    cov=[]
    for nivel in (0.8,0.9,0.95):
        lo=betabinom.ppf((1-nivel)/2,nB,a,b); hi=betabinom.ppf(1-(1-nivel)/2,nB,a,b)
        cov.append(((acB>=lo)&(acB<=hi)).mean())
    l9,h9=B.ppf(.05,a,b),B.ppf(.95,a,b)
    pt=a/(a+b)
    print(f'{n:>5} │ {cov[0]:>7.1%} {cov[1]:>8.1%} {cov[2]:>8.1%} │ {np.mean(h9-l9):>12.3f} {np.mean(np.abs(pt-sub.verdade.values)):>9.3f}')
print('\n=== o que da para AFIRMAR a partir do intervalo (90%) ===')
print(f"{'n':>5} │ {'todo abaixo de 50%':>19} {'todo acima de 80%':>18} {'so o intervalo':>15}")
print('─'*62)
for n in (10,21,30,50,100):
    sub=q[q.size_a>=n]; verd=sub.verdade.values
    ac=rng.binomial(n,np.clip(sub.sum_a/sub.size_a,0,1))
    a=ac+A0; b=(n-ac)+B0
    lo,hi=B.ppf(.05,a,b),B.ppf(.95,a,b)
    d=hi<.50; f=lo>.80
    ad=(verd[d]<.50).mean() if d.sum() else float('nan')
    af=(verd[f]>.80).mean() if f.sum() else float('nan')
    print(f'{n:>5} │ {d.mean():>8.1%} (acerta {ad:>4.0%}) {f.mean():>7.1%} (acerta {af:>4.0%}) {(~(d|f)).mean():>14.1%}')
print(f'\n=== largura do intervalo: quando ele fica util ===')
for n in (5,10,21,30,50,100,200):
    sub=q[q.size_a>=min(n,100)]
    ac=rng.binomial(n,np.clip(sub.sum_a/sub.size_a,0,1))
    a=ac+A0;b=(n-ac)+B0
    w=np.mean(B.ppf(.95,a,b)-B.ppf(.05,a,b))
    print(f'  {n:>3} respostas -> largura media {w:.3f}  ({"util" if w<0.20 else "larga demais"})')
