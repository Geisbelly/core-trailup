"""O gargalo e representacao? Encoder multilingue pre-treinado vs features de superficie.

Inclui COBERTURA PONTO A PONTO: quebra o gabarito em frases e mede se cada uma
encontra correspondencia semantica na resposta. E o mais proximo de como um
humano corrige - ponto por ponto, nao por sobreposicao de vocabulario.
"""
import os, re, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_pred.parquet').reset_index(drop=True)
d['aluno']=d.aluno.fillna(pd.Series([f'anon_{i}' for i in range(len(d))]))
print('carregando encoder multilingue...')
M=SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
def frases(t):
    p=[s.strip() for s in re.split(r'(?<=[.!?;])\s+|\n+',str(t)) if len(s.strip())>15]
    return p or [str(t)[:300]]
print('embutindo textos...')
E_r=M.encode(list(d.resp.astype(str)),normalize_embeddings=True,batch_size=64,show_progress_bar=False)
E_g=M.encode(list(d.gab.astype(str)),normalize_embeddings=True,batch_size=64,show_progress_bar=False)
E_q=M.encode(list(d.q.astype(str)),normalize_embeddings=True,batch_size=64,show_progress_bar=False)
d['emb_cos_gab']=(E_r*E_g).sum(1); d['emb_cos_q']=(E_r*E_q).sum(1)
print('cobertura ponto a ponto...')
cov_med=[];cov_min=[];cov_frac=[];prec=[]
for i in range(len(d)):
    fg=frases(d.gab.iloc[i]); fr=frases(d.resp.iloc[i])
    A=M.encode(fg,normalize_embeddings=True,show_progress_bar=False)
    B=M.encode(fr,normalize_embeddings=True,show_progress_bar=False)
    S=A@B.T
    cov_med.append(float(S.max(1).mean())); cov_min.append(float(S.max(1).min()))
    cov_frac.append(float((S.max(1)>0.5).mean())); prec.append(float(S.max(0).mean()))
d['cov_med']=cov_med; d['cov_min']=cov_min; d['cov_frac']=cov_frac; d['prec_resp']=prec
y=d.nota.values; g=d.aluno.values
GRAFO=[c for c in ['no_jaccard','no_cobertura','no_peso','cob_conceitos_centrais','razao_tam',
    'divagacao','aresta_jaccard','aresta_cobertura','aresta_peso','densidade','n_arestas'] if c in d]
EMB=['emb_cos_gab','emb_cos_q','cov_med','cov_min','cov_frac','prec_resp']
def qwk(a,b): return cohen_kappa_score(np.clip(np.round(a),1,5).astype(int),
                                       np.clip(np.round(b),1,5).astype(int),weights='quadratic')
def roda(F,nome,grp=None):
    m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
        min_samples_leaf=30,l2_regularization=1.0,random_state=0)
    cv=GroupKFold(5) if grp is None else LeaveOneGroupOut()
    p=cross_val_predict(m,d[F],y,groups=(g if grp is None else grp),cv=cv)
    return {'modelo':nome,'Spearman':spearmanr(p,y).statistic,'QWK':qwk(p,y),'MAE':mean_absolute_error(y,p)}
print('\n=== correlacao isolada das features novas ===')
for c in EMB: print(f'  {c:>14}: {d[c].corr(d.nota,method="spearman"):+.3f}')
rows=[roda(GRAFO+['log_len'],'GRAFO (superficie)'),
      roda(['emb_cos_gab','emb_cos_q','log_len'],'encoder: so o cosseno do texto todo'),
      roda(EMB+['log_len'],'encoder + cobertura ponto a ponto'),
      roda(GRAFO+EMB+['log_len'],'GRAFO + encoder')]
print('\n=== agrupado por aluno ===')
print(pd.DataFrame(rows).set_index('modelo').round(3).to_string())
r1=d['Run1_Content_AI Evaluation'].values; r23=d[['Run2_Content_AI Evaluation','Run3_Content_AI Evaluation']].mean(1).values
print(f'\nteto (LLM x ele mesmo): Spearman {spearmanr(r1,r23).statistic:.3f} | QWK {qwk(r1,r23):.3f}')
print('\n=== tarefa nunca vista (leave-one-Task-out) ===')
for F,n in [(GRAFO+['log_len'],'GRAFO'),(EMB+['log_len'],'encoder'),(GRAFO+EMB+['log_len'],'os dois')]:
    r=roda(F,n,grp=d.Task.values); print(f'  {n:>10}: Spearman {r["Spearman"]:.3f} | QWK {r["QWK"]:.3f}')
d.to_parquet(f'{HERE}/classex_emb.parquet')
