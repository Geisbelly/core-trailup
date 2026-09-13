"""Posicionamento realista: pre-filtro de dois lados, LLM no meio."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_emb.parquet').reset_index(drop=True)
d['aluno']=d.aluno.fillna(pd.Series([f'anon_{i}' for i in range(len(d))]))
F=[c for c in ['no_jaccard','no_cobertura','no_peso','cob_conceitos_centrais','razao_tam','divagacao',
   'aresta_jaccard','aresta_cobertura','aresta_peso','densidade','n_arestas',
   'emb_cos_gab','emb_cos_q','cov_med','cov_min','cov_frac','prec_resp','log_len'] if c in d]
y=d.nota.values
m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
    min_samples_leaf=30,l2_regularization=1.0,random_state=0)
p=cross_val_predict(m,d[F],y,groups=d.aluno,cv=GroupKFold(5))
print(f'modelo final: Spearman {spearmanr(p,y).statistic:.3f} | {len(F)} features\n')
print('=== pre-filtro de dois lados: LLM so no meio ===')
print(f"{'faixa baixa':>12} {'faixa alta':>12} │ {'LLM em':>8} {'erro na ponta baixa':>20} {'erro na ponta alta':>19}")
print('-'*82)
for lo,hi in [(.05,.95),(.10,.90),(.15,.85),(.20,.80),(.25,.75)]:
    a,b=np.quantile(p,lo),np.quantile(p,hi)
    baixo=p<=a; alto=p>=b; meio=~(baixo|alto)
    # na ponta baixa, dizemos "precisa de feedback" -> erro = era boa (>=3,5)
    e_baixo=(y[baixo]>=3.5).mean()
    # na ponta alta, dizemos "adequada" -> erro = nao era (<3,5)
    e_alto=(y[alto]<3.5).mean()
    print(f'{lo:>11.0%} {hi:>12.0%} │ {meio.mean():>7.0%} {e_baixo:>19.0%} {e_alto:>18.0%}')
print('\n=== e se a exigencia for so ordenar para o professor revisar? ===')
ordem=np.argsort(-p)
for k in (10,20,50):
    piores=ordem[-k:]
    print(f'  as {k} respostas de menor previsao: nota real media {y[piores].mean():.2f} (geral {y.mean():.2f})')
