"""Por que a cobertura quebrou? Duas causas candidatas, testadas separadamente.

(1) O ALVO TREME: a "verdade" e a taxa numa amostra finita, com ruido binomial
    proprio. O teste certo e POSTERIOR PREDITIVO: o intervalo deve cobrir a
    CONTAGEM observada na metade B, nao a taxa latente que ninguem ve.
(2) VIES DO ENCOLHIMENTO: puxar para 0,668 desloca o centro de questoes
    extremas. Testado variando a forca do prior.
"""
import os, numpy as np, pandas as pd
from scipy.stats import beta as B, betabinom
HERE=os.path.dirname(os.path.abspath(__file__))
q=pd.read_parquet(f'{HERE}/questoes.parquet').reset_index()
G=0.668; rng=np.random.RandomState(20260912)
q=q[q.size_a>=30]
print(f'{len(q):,} questoes com >=30 respostas em cada metade\n')

print('=== (1) teste correto: cobertura POSTERIOR PREDITIVA ===')
print('    o intervalo cobre a contagem de acertos observada na metade B?\n')
print(f"{'n_A':>5} │ {'prior':>6} │ {'nominal 80%':>12} {'nominal 90%':>12} │ {'largura 90%':>12}")
print('─'*58)
for nA in (10,21,30,50,100):
    sub=q[q.size_a>=nA]
    acA=rng.binomial(nA,np.clip(sub.sum_a/sub.size_a,0,1))
    nB=sub.size_b.values; acB=sub.sum_b.values
    for prior in (15,):
        a=acA+prior*G; b=(nA-acA)+prior*(1-G)
        cov={}
        for nivel in (0.8,0.9):
            lo=betabinom.ppf((1-nivel)/2,nB,a,b); hi=betabinom.ppf(1-(1-nivel)/2,nB,a,b)
            cov[nivel]=((acB>=lo)&(acB<=hi)).mean()
        lo9=B.ppf(.05,a,b); hi9=B.ppf(.95,a,b)
        print(f'{nA:>5} │ {prior:>6} │ {cov[0.8]:>11.1%} {cov[0.9]:>12.1%} │ {np.mean(hi9-lo9):>12.3f}')

print('\n=== (2) o encolhimento enviesa? cobertura por forca do prior ===')
print(f"{'prior':>6} │ {'cobertura 90%':>14} {'largura':>8} │ {'vies medio':>11} {'erro abs':>9}")
print('─'*58)
nA=30
sub=q[q.size_a>=nA]; verd=sub.verdade.values
acA=rng.binomial(nA,np.clip(sub.sum_a/sub.size_a,0,1))
nB=sub.size_b.values; acB=sub.sum_b.values
for prior in (0,3,8,15,30,60):
    a=acA+prior*G+1e-9; b=(nA-acA)+prior*(1-G)+1e-9
    lo=betabinom.ppf(.05,nB,a,b); hi=betabinom.ppf(.95,nB,a,b)
    cov=((acB>=lo)&(acB<=hi)).mean()
    pt=(acA+prior*G)/(nA+prior)
    print(f'{prior:>6} │ {cov:>13.1%} {np.mean(B.ppf(.95,a,b)-B.ppf(.05,a,b)):>8.3f} │ '
          f'{np.mean(pt-verd):>+11.3f} {np.mean(np.abs(pt-verd)):>9.3f}')

print('\n=== (3) ha sobredispersao? as questoes variam mais que o binomial ===')
p=q.verdade.values
print(f'  desvio real entre questoes        : {p.std():.4f}')
esp=np.sqrt(np.mean(p*(1-p)/q.size_b.values))
print(f'  desvio esperado so por ruido      : {esp:.4f}')
print(f'  variacao genuina entre questoes   : {np.sqrt(max(p.var()-esp**2,0)):.4f}')
mu=p.mean(); v=max(p.var()-esp**2,1e-6)
k=mu*(1-mu)/v-1
print(f'  -> prior Beta que reflete essa dispersao: a={mu*k:.1f} b={(1-mu)*k:.1f} (forca {k:.1f})')
print(f'     (o prior 15 usado ate agora {"esta perto" if abs(k-15)<8 else "NAO bate"} disso)')
