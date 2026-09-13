"""O modelo esta com fome de dado? Curva de aprendizado.

Se a performance ainda sobe em 1.167 exemplos, buscar mais dado (questoes de
concurso, provas publicas) se paga. Se plateou, o limite e outro.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_pred.parquet').reset_index(drop=True)
d['aluno']=d.aluno.fillna(pd.Series([f'anon_{i}' for i in range(len(d))]))
F=['no_jaccard','no_cobertura','no_peso','cob_conceitos_centrais','razao_tam','divagacao',
   'aresta_jaccard','aresta_cobertura','aresta_peso','densidade','n_arestas',
   'sem_media','sem_frac','cos_tfidf','cos_lsa','log_len']
F=[c for c in F if c in d.columns]
y=d.nota.values; g=d.aluno.values
rng=np.random.RandomState(0)
print(f'{len(d)} exemplos | {len(F)} features\n')
print('=== curva de aprendizado (media de 5 sorteios) ===')
print(f"{'exemplos de treino':>20} {'Spearman':>10} {'QWK':>8}")
print('-'*42)
prev=None
for frac in (0.1,0.2,0.35,0.5,0.7,0.85,1.0):
    sp=[];qw=[]
    for rep in range(5):
        P=np.full(len(d),np.nan)
        for tr,te in GroupKFold(5).split(d,y,groups=g):
            k=max(30,int(len(tr)*frac))
            sel=rng.choice(tr,size=min(k,len(tr)),replace=False)
            m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
                min_samples_leaf=20,l2_regularization=1.0,random_state=rep).fit(d[F].iloc[sel],y[sel])
            P[te]=m.predict(d[F].iloc[te])
        sp.append(spearmanr(P,y).statistic)
        qw.append(cohen_kappa_score(np.clip(np.round(P),1,5).astype(int),
                                    np.clip(np.round(y),1,5).astype(int),weights='quadratic'))
    n=int(len(d)*0.8*frac)
    print(f'{n:>20,} {np.mean(sp):>10.3f} {np.mean(qw):>8.3f}'+
          (f'   (+{np.mean(sp)-prev:.3f})' if prev is not None else ''))
    prev=np.mean(sp)
print('\n=== e se o limite for o NUMERO DE TAREFAS, nao de respostas? ===')
for k in (2,4,6,8):
    tasks=sorted(d.Task.unique())[:k]
    sub=d[d.Task.isin(tasks)]
    P=np.full(len(sub),np.nan)
    for tr,te in GroupKFold(4).split(sub,sub.nota,groups=sub.aluno):
        m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
            min_samples_leaf=20,random_state=0).fit(sub[F].iloc[tr],sub.nota.iloc[tr])
        P[te]=m.predict(sub[F].iloc[te])
    print(f'  {k} tarefas ({len(sub):>5} respostas): Spearman {spearmanr(P,sub.nota).statistic:.3f}')
