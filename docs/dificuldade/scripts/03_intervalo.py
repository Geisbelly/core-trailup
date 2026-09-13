"""Faixa de VALORES em vez de categoria: intervalo de credibilidade da dificuldade.

posterior Beta(acertos + p*G, erros + p*(1-G)) -> intervalo.
Teste que importa: COBERTURA. O intervalo de 90% contem a verdade em 90%?
E largura: com quantas respostas ele fica estreito o bastante para ser util?

Compara a versao exata (scipy) com a aproximacao de Wilson no logit, que nao
precisa de dependencia nenhuma - se a cobertura bater, a implementacao fica leve.
"""
import os, numpy as np, pandas as pd
from scipy.stats import beta as B
HERE=os.path.dirname(os.path.abspath(__file__))
q=pd.read_parquet(f'{HERE}/questoes.parquet').reset_index()
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
G=0.668
print(f'{len(q):,} questoes | verdade = taxa numa metade das respostas (>=30)')
rng=np.random.RandomState(20260912)

def exato(ac,n,prior,g,nivel):
    a=ac+prior*g; b=(n-ac)+prior*(1-g)
    lo=B.ppf((1-nivel)/2,a,b); hi=B.ppf(1-(1-nivel)/2,a,b)
    return lo,hi
def wilson(ac,n,prior,g,nivel):
    """aproximacao normal sobre a taxa encolhida, com n efetivo = n + prior"""
    from math import sqrt
    z={0.5:0.674,0.8:1.282,0.9:1.645,0.95:1.960}[nivel]
    ne=n+prior; p=(ac+prior*g)/ne
    se=np.sqrt(np.clip(p*(1-p),1e-9,None)/ne)
    return np.clip(p-z*se,0,1), np.clip(p+z*se,0,1)

print('\n=== COBERTURA: o intervalo contem a verdade? ===')
print(f"{'n':>5} │ {'nivel':>6} │ {'exato (Beta)':>22} │ {'Wilson (sem dependencia)':>26}")
print(f"{'':>5} │ {'':>6} │ {'cobertura':>11} {'largura':>9} │ {'cobertura':>11} {'largura':>13}")
print('─'*78)
linhas=[]
for n in (5,10,21,30,50,100):
    sub=q[q.size_a>=n]
    verd=sub.verdade.values
    ac=rng.binomial(n,np.clip(sub.sum_a/sub.size_a,0,1))
    for nivel in (0.8,0.9):
        le,he=exato(ac,n,15,G,nivel); lw,hw=wilson(ac,n,15,G,nivel)
        ce=((verd>=le)&(verd<=he)).mean(); cw=((verd>=lw)&(verd<=hw)).mean()
        print(f'{n:>5} │ {nivel:>5.0%} │ {ce:>10.1%} {np.mean(he-le):>9.3f} │ {cw:>10.1%} {np.mean(hw-lw):>13.3f}')
        linhas.append((n,nivel,ce,np.mean(he-le),cw,np.mean(hw-lw)))

print('\n=== o intervalo como decisao: quando da para AFIRMAR algo ===')
print('   (usando o intervalo de 90% exato, corte de referencia 50% e 80%)')
print(f"{'n':>5} {'largura':>8} │ {'certamente dificil':>19} {'certamente facil':>17} {'so intervalo':>13}")
print('─'*70)
for n in (5,10,21,30,50,100):
    sub=q[q.size_a>=n]; verd=sub.verdade.values
    ac=rng.binomial(n,np.clip(sub.sum_a/sub.size_a,0,1))
    lo,hi=exato(ac,n,15,G,0.9)
    dif=hi<0.50; fac=lo>0.80; indef=~(dif|fac)
    acc_d=(verd[dif]<0.50).mean() if dif.sum() else float('nan')
    acc_f=(verd[fac]>0.80).mean() if fac.sum() else float('nan')
    print(f'{n:>5} {np.mean(hi-lo):>8.3f} │ {dif.mean():>7.1%} (acerta {acc_d:>4.0%}) '
          f'{fac.mean():>6.1%} (acerta {acc_f:>4.0%}) {indef.mean():>12.1%}')
print('\n  "certamente dificil" = o intervalo INTEIRO fica abaixo de 50%')
print('  a afirmacao so aparece quando o dado sustenta; o resto devolve so o intervalo')
