"""Isola EVIDENCIA de COMPOSICAO: todas as linhas sobre AS MESMAS questoes.

Na tabela anterior, cada n usava as questoes com >=n+30 respostas - conjuntos
diferentes, e questoes muito respondidas sao mais dificeis. Aqui fixo o conjunto
nas questoes com >=230 respostas e vario so o n.
"""
import os, numpy as np, pandas as pd
from scipy.stats import betabinom
M2='/caminho/para/os/intermediarios'
A0,B0=5.07,2.08; FRONT=40.0
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','lat_final','ts']).sort_values('ts')
tot=df.groupby('qidx').size()
qs=tot[tot>=230].index
sub=df[df.qidx.isin(qs)]
print(f'conjunto FIXO: {len(qs):,} questoes com >=230 respostas')
print(f'acerto medio do conjunto: {sub.correct.mean():.3f} (o geral e {df.correct.mean():.3f})\n')
verd=sub.groupby('qidx').tail(-200).groupby('qidx').agg(sv=('correct','sum'),nv=('correct','size'),
                                                        latv=('lat_final','median'))
linhas=[]
for N in (5,10,15,21,30,50,100,200):
    p=sub.groupby('qidx').head(N).groupby('qidx').agg(s=('correct','sum'),n=('correct','size'),
                                                      d=('user','nunique'),lat=('lat_final','median'))
    j=p.join(verd); j=j[j.nv>=30]
    v=j.sv/j.nv; peso=j.d/j.n
    a=j.s*peso+A0; b=(j.n-j.s)*peso+B0
    m=a/(a+b); sd=np.sqrt(a*b/((a+b)**2*(a+b+1)))
    lo,hi=np.clip(m-1.645*sd,0,1),np.clip(m+1.645*sd,0,1)
    lop=betabinom.ppf(.05,j.nv,a,b); hip=betabinom.ppf(.95,j.nv,a,b)
    af=((hi<.50)|(lo>.80))
    ac=(((hi<.50)&(v<.50))|((lo>.80)&(v>.80)))[af].mean() if af.sum() else np.nan
    linhas.append({'n':N,'questoes':len(j),'ritmo':((j.lat>FRONT)==(j.latv>FRONT)).mean(),
        'largura':np.mean(hi-lo),'erro_abs':np.mean(np.abs(m-v)),
        'cobertura':((j.sv>=lop)&(j.sv<=hip)).mean(),'crava':af.mean(),'acerta':ac})
T=pd.DataFrame(linhas).set_index('n')
print('=== mesmas questoes em todas as linhas ===')
print(T.round(3).to_string())
print('\n=== comparacao com a tabela de conjunto variavel ===')
ant={5:0.000,10:0.002,15:0.043,21:0.051,30:0.122,50:0.172,100:0.232,200:0.266}
print(f"{'n':>5} {'crava (conj. fixo)':>19} {'crava (conj. variavel)':>23} {'diferenca':>11}")
for N in T.index:
    print(f'{N:>5} {T.loc[N,"crava"]:>18.1%} {ant[N]:>22.1%} {T.loc[N,"crava"]-ant[N]:>+11.1%}')
