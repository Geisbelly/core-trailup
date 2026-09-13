"""O que sao os 2 grupos que o HDBSCAN encontrou - e o que e o 'ruido'?"""
import os, numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
HERE=os.path.dirname(os.path.abspath(__file__))
q=pd.read_parquet(f'{HERE}/questoes_agrupadas.parquet')
F=['acerto','lat_med','trocas','lat_1a']
X=q[F].copy(); X['lat_med']=np.log1p(X.lat_med); X['lat_1a']=np.log1p(X.lat_1a)
Xs=StandardScaler().fit_transform(X)
lab=HDBSCAN(min_cluster_size=200,min_samples=10).fit(Xs).labels_
q['h']=lab
print('=== perfil dos grupos do HDBSCAN ===')
p=q.groupby('h').agg(questoes=('qidx','size'),acerto=('acerto','mean'),acerto_sd=('acerto','std'),
    lat_med=('lat_med','median'),lat_sd=('lat_med','std'),trocas=('trocas','mean'),n=('n','median'))
p.index=['RUIDO' if i==-1 else f'grupo {i}' for i in p.index]
print(p.round(3).to_string())
print('\n=== a separacao e no TEMPO ou no ACERTO? ===')
for c in ('acerto','lat_med','trocas'):
    a=q[q.h==0][c]; b=q[q.h==1][c]
    d=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    print(f'  {c:>8}: grupo0 {a.mean():>7.2f} | grupo1 {b.mean():>7.2f} | separacao (d de Cohen) {abs(d):.2f}')
print('\n=== o "ruido" e o que? ===')
r=q[q.h==-1]
print(f'  {len(r):,} questoes ({len(r)/len(q):.1%})')
print(f'  acerto {r.acerto.mean():.2f} (desvio {r.acerto.std():.2f}) vs {q[q.h>=0].acerto.mean():.2f} ({q[q.h>=0].acerto.std():.2f}) nos grupos')
print(f'  latencia {r.lat_med.median():.0f}s vs {q[q.h>=0].lat_med.median():.0f}s')
print(f'  -> {"sao os EXTREMOS (cauda), nao um grupo" if r.acerto.std()>q[q.h>=0].acerto.std() else "sao intermediarias"}')
print('\n=== silhueta: 2 grupos contra 4 ===')
from sklearn.cluster import KMeans
for k in (2,3,4,5):
    km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs)
    print(f'  k-means k={k}: {silhouette_score(Xs,km.labels_,sample_size=5000,random_state=0):.3f}')
m=lab!=-1
print(f'  HDBSCAN (2 grupos, sem ruido): {silhouette_score(Xs[m],lab[m]):.3f}')
print('\n=== a estrutura e continua ou discreta? ===')
# se for continua, as features sao unimodais; se discreta, bimodais
from scipy.stats import gaussian_kde
for c,lg in (('acerto',False),('lat_med',True)):
    v=np.log1p(q[c]) if lg else q[c]
    kde=gaussian_kde(v); xs=np.linspace(v.min(),v.max(),300); d=kde(xs)
    picos=((d[1:-1]>d[:-2])&(d[1:-1]>d[2:])).sum()
    print(f'  {c:>8}: {picos} pico(s) na densidade -> {"bimodal (grupos reais)" if picos>1 else "unimodal (continuo)"}')
