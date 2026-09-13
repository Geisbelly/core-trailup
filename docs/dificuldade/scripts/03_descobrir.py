"""Descobrir QUANTOS grupos existem, em vez de arbitrar.

Tres metodos que nao recebem k:
  HDBSCAN  - densidade; so pede o tamanho minimo de um grupo, e marca ruido
  Dirichlet Process GMM - recebe um teto e PODodd as componentes que nao usa
  Mean Shift - a largura de banda sai do proprio dado

Se os tres convergirem, a estrutura e do dado. Se divergirem, era artefato.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
HERE=os.path.dirname(os.path.abspath(__file__))
q=pd.read_parquet(f'{HERE}/questoes_agrupadas.parquet')
F=['acerto','lat_med','trocas','lat_1a']
X=q[F].copy(); X['lat_med']=np.log1p(X.lat_med); X['lat_1a']=np.log1p(X.lat_1a)
Xs=StandardScaler().fit_transform(X)
print(f'{len(q):,} questoes, {len(F)} features\n')

res={}
print('=== HDBSCAN (densidade; so pede o tamanho minimo de grupo) ===')
for mcs in (50,100,200,400):
    h=HDBSCAN(min_cluster_size=mcs,min_samples=10).fit(Xs)
    lab=h.labels_; k=len(set(lab))-(1 if -1 in lab else 0)
    ruido=(lab==-1).mean()
    sil=silhouette_score(Xs[lab!=-1],lab[lab!=-1]) if k>1 and (lab!=-1).sum()>k else float('nan')
    tam=np.bincount(lab[lab!=-1]) if k else []
    print(f'  min_cluster_size={mcs:>3} -> {k} grupos | ruido {ruido:>5.1%} | silhueta {sil:.3f} | '
          f'tamanhos {sorted(tam,reverse=True)[:6]}')
    res[f'hdbscan{mcs}']=lab

print('\n=== Dirichlet Process GMM (teto de 15, poda o que nao usa) ===')
for wcp in (1e-2,1.0,100.0):
    b=BayesianGaussianMixture(n_components=15,weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=wcp,max_iter=500,random_state=0,n_init=2).fit(Xs)
    w=b.weights_; efet=(w>0.02).sum()
    lab=b.predict(Xs); k=len(set(lab))
    print(f'  prior={wcp:>6} -> {efet} componentes com peso >2% (usa {k}) | '
          f'pesos {np.sort(w)[::-1][:6].round(3)}')
    res[f'dpgmm{wcp}']=lab

print('\n=== Mean Shift (largura de banda estimada do dado) ===')
sub=Xs[np.random.RandomState(0).choice(len(Xs),3000,replace=False)]
for quant in (0.15,0.25,0.35):
    bw=estimate_bandwidth(sub,quantile=quant,random_state=0)
    ms=MeanShift(bandwidth=bw,bin_seeding=True).fit(sub)
    k=len(set(ms.labels_))
    print(f'  quantile={quant} -> largura {bw:.2f} | {k} grupos | '
          f'tamanhos {sorted(np.bincount(ms.labels_),reverse=True)[:6]}')

print('\n=== os metodos concordam entre si? (ARI) ===')
km=q.grupo.values
pares=[('hdbscan100','dpgmm1.0'),('hdbscan200','dpgmm1.0'),('hdbscan100','hdbscan200')]
for a,b in pares:
    m=(res[a]!=-1)
    print(f'  {a:>12} x {b:<12} ARI {adjusted_rand_score(res[a][m],res[b][m]):+.3f}')
for a in ('hdbscan100','hdbscan200','dpgmm1.0'):
    m=(res[a]!=-1)
    print(f'  {a:>12} x {"kmeans(k=4)":<12} ARI {adjusted_rand_score(res[a][m],km[m]):+.3f}')
np.save(f'{HERE}/labels_hdbscan.npy',res['hdbscan200'])
