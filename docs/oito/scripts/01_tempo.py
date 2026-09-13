"""(1) Tempo esperado de resposta - regressao, nao classificacao.

Serve para: estimar duracao da atividade, e detectar aluno muito acima do
esperado NAQUELA questao - sem inferir estado emocional.
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
d=pd.read_parquet(f'{M2}/features_v3.parquet')
rng=np.random.RandomState(20260912)
u=d.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict(); s.update(dict.fromkeys(u[:int(.8*n)],'train'))
d['split']=d.user.map(s)
tr=d[d.split=='train']; te=d[d.split=='test'].copy()
ql=tr.groupby('qidx').lat_final.agg(['median','size'])
GM=tr.lat_final.median()
for X in (tr,te):
    X['q_lat']=np.log1p(X.qidx.map(ql['median']).fillna(GM))
    X['q_n']=np.log1p(X.qidx.map(ql['size']).fillna(0))
y_tr=np.log1p(tr.lat_final.clip(1,600)); y_te=np.log1p(te.lat_final.clip(1,600))
F=['q_lat','q_n','lat_med_prev','n_prev','acc_prev','ewma_part','n_prev_part','part',
   'seen_before','pos_sessao','ntags','diagnosis','streak','acc5']
m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.06,max_depth=6,
    min_samples_leaf=200,random_state=0,categorical_features=[F.index('part')]).fit(tr[F],y_tr)
p=m.predict(te[F])
print(f'treino {len(tr):,} | teste {len(te):,} | latencia mediana {np.expm1(y_te).median():.1f}s\n')
print('=== prever o tempo de resposta (log-segundos) ===')
base_g=np.full(len(te),y_tr.median())
base_q=te.q_lat.values
base_a=np.log1p(te.lat_med_prev.clip(1,600))
for nome,pp in [('sempre a mediana global',base_g),('mediana DA QUESTAO',base_q),
                ('mediana DO ALUNO',base_a),('modelo',p)]:
    seg=np.expm1(pp)
    print(f'  {nome:>24}: R2 {r2_score(y_te,pp):>6.3f} | erro mediano {np.median(np.abs(np.expm1(y_te)-seg)):>5.1f}s '
          f'| erro relativo {np.median(np.abs(np.expm1(y_te)-seg)/np.expm1(y_te)):>5.1%}')
print('\n=== detectar "muito acima do esperado" ===')
res=np.expm1(y_te)/np.expm1(p)
for lim in (2,3,5):
    m2=res>lim
    print(f'  {lim}x acima do esperado: {m2.mean():>5.2%} das respostas | '
          f'acerto nessas {te.y[m2].mean():.1%} (base {te.y.mean():.1%})')
