"""K-means sobre as questoes para descobrir os rotulos, em vez de eu arbitrar cortes.

Agrupa por PERFIL DE RESPOSTA, nao so por taxa de acerto: uma questao pode ser
lenta-e-acertada (exige trabalho) ou rapida-e-errada (pega o aluno no chute).
Sao coisas pedagogicamente diferentes que um corte em 50/80 nao separa.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(f'{M2}/responses_v3.parquet')
print(f'{len(df):,} respostas | {df.qidx.nunique():,} questoes')
q=df.groupby('qidx').agg(
    n=('correct','size'), alunos=('user','nunique'), acerto=('correct','mean'),
    lat_med=('lat_final','median'), lat_p75=('lat_final',lambda s:s.quantile(.75)),
    trocas=('n_trocas',lambda s:(s>0).mean()), lat_1a=('lat_1a','median'),
    expl=('expl_sec','median'),
).reset_index()
q=q[q.n>=100]                     # perfil so e estavel com volume
print(f'{len(q):,} questoes com >=100 respostas\n')
F=['acerto','lat_med','trocas','lat_1a']
X=q[F].copy(); X['lat_med']=np.log1p(X.lat_med); X['lat_1a']=np.log1p(X.lat_1a)
Xs=StandardScaler().fit_transform(X)
print('=== quantos grupos? ===')
for k in range(2,7):
    km=KMeans(n_clusters=k,n_init=10,random_state=0).fit(Xs)
    sil=silhouette_score(Xs,km.labels_,sample_size=5000,random_state=0)
    print(f'  k={k} | silhueta {sil:.3f} | inercia {km.inertia_:>9.0f} | '
          f'grupos: {", ".join(str(int(c)) for c in np.bincount(km.labels_))}')
K=int(os.environ.get('K','4'))
km=KMeans(n_clusters=K,n_init=20,random_state=0).fit(Xs)
q['grupo']=km.labels_
print(f'\n=== perfil de cada grupo (k={K}) ===')
p=q.groupby('grupo').agg(questoes=('qidx','size'),acerto=('acerto','mean'),
    lat_med=('lat_med','median'),trocas=('trocas','mean'),lat_1a=('lat_1a','median'))
p['%']=(p.questoes/len(q)*100).round(1)
print(p.round(3).to_string())
# nomear pelos extremos
print('\n=== leitura dos grupos ===')
for g,r in p.iterrows():
    nome=[]
    nome.append('dificil' if r.acerto<0.55 else ('facil' if r.acerto>0.82 else 'media'))
    nome.append('lenta' if r.lat_med>q.lat_med.median()*1.3 else ('rapida' if r.lat_med<q.lat_med.median()*0.7 else ''))
    nome.append('muita hesitacao' if r.trocas>q.trocas.median()*1.4 else '')
    print(f'  grupo {g}: {r.questoes:>5.0f} questoes ({r["%"]:>4.1f}%) - '
          f'{", ".join(x for x in nome if x)} | acerto {r.acerto:.0%}, {r.lat_med:.0f}s, troca em {r.trocas:.0%}')
q.to_parquet(f'{HERE}/questoes_agrupadas.parquet',index=False)
import pickle
pickle.dump({'kmeans':km,'scaler':StandardScaler().fit(X),'feats':F},open(f'{HERE}/kmeans.pkl','wb'))
print(f'\n-> questoes_agrupadas.parquet ({len(q):,} questoes)')
