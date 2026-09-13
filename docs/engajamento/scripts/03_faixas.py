"""Estabilidade dos eixos, checagem de Simpson e faixas calibradas."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, spearman
A=pd.read_parquet(f'{HERE}/eixos.parquet')
M2=HERE.replace('/engaj','/m2'); DIA=86400*1000.0
EIXOS=['freq','volume','prof']

print('=== traco? metades independentes das respostas da janela 1 ===')
d=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','ts','expl_sec']).sort_values(['user','ts'])
ini=d.groupby('user').ts.transform('min'); d['dia']=(d.ts-ini)/DIA
d=d[(d.dia<30)&(d.user.isin(A.index))]
rng=np.random.RandomState(7); d['h']=rng.randint(0,2,len(d))
def eixos(s):
    dt=s.groupby('user').ts.diff()/1000.0
    s=s.assign(sessao=((dt.isna())|(dt>1800)).groupby(s.user).cumsum())
    g=s.groupby('user')
    B=pd.DataFrame({'n':g.size(),'ses':g.sessao.max(),
                    'da':g.dia.apply(lambda x:x.astype(int).nunique()),
                    'expl':g.expl_sec.median()})
    B=B[B.n>=15]
    return pd.DataFrame({'freq':B.da/30,'volume':B.n/B.ses.clip(lower=1),
                         'prof':np.log1p(B.expl)})
a0,a1=eixos(d[d.h==0]),eixos(d[d.h==1])
c=a0.index.intersection(a1.index)
for e in EIXOS:
    print(f'    {e:>8}: {spearman(a0.loc[c,e].values,a1.loc[c,e].values):+.3f}  ({len(c):,} alunos)')

print('\n=== volume negativo e Simpson? efeito dentro de cada quintil de freq ===')
A['qf']=pd.qcut(A.freq,5,labels=False,duplicates='drop')
for q in sorted(A.qf.dropna().unique()):
    S=A[A.qf==q]
    print(f'    quintil freq {int(q)+1}: AUC(volume) {roc_auc_score(S.ret,S.volume):.3f} | ret {S.ret.mean():.1%} | n={len(S):,}')

print('\n=== faixas calibradas (decil de cada eixo -> retencao observada) ===')
for e in EIXOS:
    A[f'd_{e}']=pd.qcut(A[e],10,labels=False,duplicates='drop')
    t=A.groupby(f'd_{e}').agg(ret=('ret','mean'),lo=(e,'min'),hi=(e,'max'),n=(e,'size'))
    print(f'  -- {e} --')
    for i,r in t.iterrows():
        print(f'    d{int(i)+1:>2}: [{r.lo:8.3f},{r.hi:8.3f}]  retencao {r.ret:5.1%}  n={int(r.n):,}')

print('\n=== 3 faixas por eixo (corte nos tercis) — o que se entrega ===')
for e in EIXOS:
    q=A[e].quantile([1/3,2/3]).values
    lab=np.where(A[e]<=q[0],'baixo',np.where(A[e]<=q[1],'medio','alto'))
    t=pd.DataFrame({'f':lab,'ret':A.ret}).groupby('f').ret.agg(['mean','size']).reindex(['baixo','medio','alto'])
    print(f'  {e:>8}: cortes {q[0]:.3f} / {q[1]:.3f} | retencao ' +
          ' '.join(f'{i}={r["mean"]:.1%}' for i,r in t.iterrows()))
