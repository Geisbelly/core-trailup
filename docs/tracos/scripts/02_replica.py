"""O grupo do aluno replica? Metade A das respostas dele vs metade B.

Se o grupo for traco, o aluno cai no mesmo dos dois lados. Se for ruido de
amostragem, cai aleatoriamente - e a silhueta bonita era do formato da nuvem,
nao de grupos reais.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
df=pd.read_parquet(f'{M2}/responses_v3.parquet')
q=df.groupby('qidx').agg(qlat=('lat_final','median'),qac=('correct','mean'),qn=('correct','size'))
q=q[q.qn>=50]; df=df[df.qidx.isin(q.index)]
df['lat_rel']=df.lat_final/df.qidx.map(q.qlat)
df['chute']=((df.lat_rel<0.30)&(df.correct==0)).astype(int)
df['dif_q']=1-df.qidx.map(q.qac)
df=df.sort_values(['user','ts'])
dt=df.groupby('user').ts.diff()/1000.0
df['sessao']=((dt.isna())|(dt>1800)).groupby(df.user).cumsum()
rng=np.random.RandomState(20260912); df['h']=rng.randint(0,2,len(df))
F=['acerto','chute','lat_rel','lat_var','trocas','dif_enfrentada','expl','por_sessao']
def feats(d):
    g=d.groupby('user')
    A=pd.DataFrame({'n':g.size(),'acerto':g.correct.mean(),'chute':g.chute.mean(),
        'lat_rel':g.lat_rel.median(),'lat_var':g.lat_rel.std(),
        'trocas':g.n_trocas.apply(lambda s:(s>0).mean()),'dif_enfrentada':g.dif_q.mean(),
        'expl':g.expl_sec.median(),'por_sessao':g.size()/g.sessao.max().clip(lower=1)})
    return A
A0=feats(df[df.h==0]); A1=feats(df[df.h==1])
comum=A0.index[(A0.n>=50)].intersection(A1.index[(A1.n>=50)])
A0,A1=A0.loc[comum],A1.loc[comum]
print(f'{len(comum):,} alunos com >=50 respostas em cada metade\n')
def prep(A):
    X=A[F].copy(); X['chute']=np.log1p(X.chute*100); X['expl']=np.log1p(X.expl)
    return X
qt=QuantileTransformer(output_distribution='normal',random_state=0).fit(prep(A0))
Z0,Z1=qt.transform(prep(A0)),qt.transform(prep(A1))
h=HDBSCAN(min_cluster_size=200,min_samples=10).fit(Z0)
l0=h.labels_
l1=h.fit_predict(Z1)
k0=len(set(l0))-(1 if -1 in l0 else 0); k1=len(set(l1))-(1 if -1 in l1 else 0)
print(f'grupos na metade A: {k0} | na metade B: {k1}')
m=(l0!=-1)&(l1!=-1)
print(f'ARI entre as duas metades (mesmos alunos): {adjusted_rand_score(l0[m],l1[m]):+.3f}')
print(f'  (1,0 = o aluno cai sempre no mesmo grupo | 0,0 = aleatorio)\n')
print('=== para comparar: quais TRACOS do aluno replicam? ===')
for c in F:
    r=A0[c].corr(A1[c])
    print(f'  {c:>16}: correlacao A-B {r:+.3f}  {"TRACO estavel" if r>0.5 else ("fraco" if r>0.25 else "nao replica")}')
print('\n=== a silhueta bonita era do que? ===')
print(f'  silhueta dos grupos em A: {silhouette_score(Z0[l0!=-1],l0[l0!=-1]):.3f}')
rnd=np.random.RandomState(1).randint(0,2,len(Z0))
print(f'  silhueta de uma divisao ALEATORIA: {silhouette_score(Z0,rnd):.3f}')
from sklearn.cluster import KMeans
km=KMeans(2,n_init=10,random_state=0).fit_predict(Z0)
print(f'  silhueta de k-means k=2 (sempre acha algo): {silhouette_score(Z0,km):.3f}')
