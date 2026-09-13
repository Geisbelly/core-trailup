"""O grafo se paga? Contra similaridade simples e contra o teto do proprio LLM."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_grafo.parquet')
# 47 alunos sem pseudonimo: cada um vira grupo proprio (nunca compartilham fold)
d['aluno']=d.aluno.fillna(pd.Series([f'anon_{i}' for i in range(len(d))],index=d.index))
y=d.nota.values; g=d.aluno.values
print(f'{len(d)} respostas | {d.aluno.nunique()} grupos de aluno | {d.Task.nunique()} tarefas')
# similaridades simples, para comparar
tf=TfidfVectorizer(token_pattern=r'[A-Za-zÄÖÜäöüß]{3,}',min_df=2,sublinear_tf=True)
A=tf.fit_transform(pd.concat([d.resp,d.gab]).astype(str))
R,Gb=A[:len(d)],A[len(d):]
d['cos_tfidf']=np.asarray(R.multiply(Gb).sum(1)).ravel()/(
    np.sqrt(np.asarray(R.multiply(R).sum(1)).ravel()*np.asarray(Gb.multiply(Gb).sum(1)).ravel())+1e-9)
sv=TruncatedSVD(150,random_state=0).fit(A)
Rl,Gl=normalize(sv.transform(R)),normalize(sv.transform(Gb))
d['cos_lsa']=(Rl*Gl).sum(1)
NO=['no_jaccard','no_cobertura','no_peso','cob_conceitos_centrais','razao_tam','divagacao']
AR=['aresta_jaccard','aresta_cobertura','aresta_peso','densidade','n_arestas']
SE=['sem_media','sem_frac']
def qwk(a,b): return cohen_kappa_score(np.clip(np.round(a),1,5).astype(int),
                                       np.clip(np.round(b),1,5).astype(int),weights='quadratic')
def av(nome,p):
    return {'modelo':nome,'Spearman':spearmanr(p,y).statistic,'QWK':qwk(p,y),'MAE':mean_absolute_error(y,p)}
def roda(F,nome,mod=None):
    m=mod or HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
        min_samples_leaf=30,l2_regularization=1.0,random_state=0)
    p=cross_val_predict(m,d[F],y,groups=g,cv=GroupKFold(5))
    return av(nome,p)
rows=[av('so o comprimento',d.log_len.values),
      av('cosseno TF-IDF (resposta x gabarito)',d.cos_tfidf.values*4+1),
      av('cosseno LSA',d.cos_lsa.values*4+1)]
rows.append(roda(['log_len'],'modelo so com comprimento'))
rows.append(roda(['cos_tfidf','cos_lsa','log_len'],'similaridade + comprimento'))
rows.append(roda(NO+['log_len'],'GRAFO: so os nos (conceitos)'))
rows.append(roda(NO+AR+['log_len'],'GRAFO: nos + arestas (relacoes)'))
rows.append(roda(NO+AR+SE+['log_len'],'GRAFO completo (+ semantica LSA)'))
rows.append(roda(NO+AR+SE+['cos_tfidf','cos_lsa','log_len'],'GRAFO + similaridade'))
print('=== validacao cruzada agrupada por ALUNO ===')
print(pd.DataFrame(rows).set_index('modelo').round(3).to_string())
r1=d['Run1_Content_AI Evaluation'].values
r23=d[['Run2_Content_AI Evaluation','Run3_Content_AI Evaluation']].mean(1).values
print(f"\nteto: concordancia do proprio LLM (Run1 x media de Run2,3): "
      f"Spearman {spearmanr(r1,r23).statistic:.3f} | QWK {qwk(r1,r23):.3f} | MAE {mean_absolute_error(r23,r1):.3f}")

print('\n=== generaliza para tarefa NUNCA VISTA? (leave-one-Task-out) ===')
F=NO+AR+SE+['cos_tfidf','cos_lsa','log_len']
m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
    min_samples_leaf=30,l2_regularization=1.0,random_state=0)
p=cross_val_predict(m,d[F],y,groups=d.Task.values,cv=LeaveOneGroupOut())
print(f'  Spearman {spearmanr(p,y).statistic:.3f} | QWK {qwk(p,y):.3f} | MAE {mean_absolute_error(y,p):.3f}')
for t in sorted(d.Task.unique()):
    k=(d.Task==t).values
    if k.sum()<40: continue
    print(f'    tarefa {t}: n={k.sum():>4} | Spearman {spearmanr(p[k],y[k]).statistic:+.3f}')
print('\n=== o gate: separar "adequada" de "precisa feedback" ===')
from sklearn.metrics import roc_auc_score
for lim in (3.0,3.5,4.0):
    alvo=(y>=lim).astype(int)
    print(f'  nota >= {lim}: {alvo.mean():.0%} da base | AUC {roc_auc_score(alvo,p):.3f}')
d['pred']=p; d.to_parquet(f'{HERE}/classex_pred.parquet')
