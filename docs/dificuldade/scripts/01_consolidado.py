"""Tudo em funcao de n, medido de uma vez no mesmo recorte.

n = quantos ALUNOS responderam a questao.
Verdade = taxa medida nas respostas seguintes (>=30), independente das n primeiras.
"""
import os, numpy as np, pandas as pd
from scipy.stats import beta as B, betabinom
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
A0,B0=5.07,2.08; FRONT=40.0
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','lat_final','ts']).sort_values('ts')
print(f'{len(df):,} respostas | {df.qidx.nunique():,} questoes\n')

print('=== 1. quanta evidencia as questoes acumulam ===')
n=df.groupby('qidx').size()
al=df.groupby('qidx').user.nunique()
print(f'  respostas por questao: mediana {n.median():.0f} | p10 {n.quantile(.1):.0f} | p90 {n.quantile(.9):.0f}')
print(f'  alunos distintos     : mediana {al.median():.0f} ({al.median()/n.median():.0%} das respostas)')
for lim in (5,10,21,30,50,100):
    print(f'  questoes com >={lim:>3} alunos: {(al>=lim).mean():>5.1%}  ({(al>=lim).sum():>6,})')

print('\n=== 2. o que cada n compra ===')
linhas=[]
for N in (5,10,15,21,30,50,100,200):
    g=df.groupby('qidx')
    ok=g.size()>=N+30
    sub=df[df.qidx.isin(ok[ok].index)]
    p=sub.groupby('qidx').head(N)
    prim=p.groupby('qidx').agg(s=('correct','sum'),n=('correct','size'),
                               d=('user','nunique'),lat=('lat_final','median'))
    rest=sub.groupby('qidx').tail(-N).groupby('qidx').agg(sv=('correct','sum'),nv=('correct','size'),
                                                          latv=('lat_final','median'))
    j=prim.join(rest); j=j[j.nv>=30]
    verd=j.sv/j.nv
    peso=j.d/j.n
    a=j.s*peso+A0; b=(j.n-j.s)*peso+B0
    media=a/(a+b); sd=np.sqrt(a*b/((a+b)**2*(a+b+1)))
    lo,hi=np.clip(media-1.645*sd,0,1),np.clip(media+1.645*sd,0,1)
    lop=betabinom.ppf(.05,j.nv,a,b); hip=betabinom.ppf(.95,j.nv,a,b)
    cob=((j.sv>=lop)&(j.sv<=hip)).mean()
    ritmo_ok=((j.lat>FRONT)==(j.latv>FRONT)).mean()
    afirma=((hi<.50)|(lo>.80))
    acerta_af=(((hi<.50)&(verd<.50))|((lo>.80)&(verd>.80)))[afirma].mean() if afirma.sum() else np.nan
    linhas.append({'n':N,'questoes':len(j),'ritmo':ritmo_ok,'largura':np.mean(hi-lo),
                   'erro_abs':np.mean(np.abs(media-verd)),'cobertura':cob,
                   'afirma':afirma.mean(),'acerto_afirm':acerta_af})
T=pd.DataFrame(linhas).set_index('n')
print(T.round(3).to_string())
print("""
  ritmo        = acerta rapida/lenta (fronteira 40s)
  largura      = tamanho do intervalo de 90% sobre o acerto
  erro_abs     = |estimativa - verdade| media
  cobertura    = o intervalo de 90% contem a verdade
  afirma       = fracao em que da para cravar 'dificil' ou 'facil'
  acerto_afirm = quando crava, acerta""")

print('\n=== 3. o que isso significa numa turma real ===')
for turma,passadas in [(30,1),(30,2),(30,3),(25,2),(40,1)]:
    N=turma*passadas
    r=T.index[np.argmin(np.abs(T.index-N))]
    row=T.loc[r]
    print(f'  turma de {turma}, {passadas}x na atividade -> n={N:>3} | ritmo {row.ritmo:.0%} | '
          f'acerto +-{row.largura/2:.0%} | crava em {row.afirma:.0%} das questoes')
