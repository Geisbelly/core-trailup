"""M2 v2: mesmo protocolo da v1 (split por usuario, dificuldade so do treino).
Compara v1 x v2 x v2 sem tags, para saber de onde veio o ganho."""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features_v2.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
tr_u,va_u,te_u=u[:int(.8*n)],u[int(.8*n):int(.9*n)],u[int(.9*n):]
s=pd.Series('test',index=u).to_dict(); s.update(dict.fromkeys(tr_u,'train')); s.update(dict.fromkeys(va_u,'val'))
df['split']=df.user.map(s)
tr,va,te=(df[df.split==k].copy() for k in ('train','val','test'))
g=tr.groupby('qidx').y.agg(['mean','size']); G=tr.y.mean()
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
for d in (tr,va,te):
    d['q_dif']=d.qidx.map(diff).fillna(G).astype(np.float32)
    d['q_n']=np.log1p(d.qidx.map(g['size']).fillna(0)).astype(np.float32)
print(f'treino {len(tr):,} | val {len(va):,} | teste {len(te):,}')

V1=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
    'dt_log','dt_part_log','seen_before','ntags','diagnosis','part','q_dif','q_n']
NOVAS_SEM_TAG=['acc5','acc10','n_sessao','pos_sessao','dt_rel','bundle_pos','acc_bundle']
TAG=['tag_acc','tag_n','tag_acc_min','tag_acc_max','tag_ewma','tag_novo']

def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def run(F,nome,**kw):
    par=dict(max_iter=400,learning_rate=0.06,max_depth=6,min_samples_leaf=200,
             l2_regularization=1.0,random_state=0,categorical_features=[F.index('part')])
    par.update(kw)
    m=HistGradientBoostingClassifier(**par).fit(tr[F],tr.y)
    p=np.clip(m.predict_proba(te[F])[:,1],1e-6,1-1e-6)
    return m,p,{'modelo':nome,'n_feats':len(F),'AUC':roc_auc_score(te.y,p),
                'logloss':log_loss(te.y,p),'brier':brier_score_loss(te.y,p),'ECE':ece(te.y.values,p)}

rows=[]; ps={}
for F,nm in [(V1,'v1 (15 feats) — referencia'),
             (V1+NOVAS_SEM_TAG,'v2 sem tags (janela+tempo+sessao)'),
             (V1+TAG,'v1 + tags'),
             (V1+NOVAS_SEM_TAG+TAG,'v2 COMPLETO')]:
    m,p,r=run(F,nm); rows.append(r); ps[nm]=p
print('\n=== teste (usuarios nunca vistos) ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())

# tuning leve no melhor conjunto
F=V1+NOVAS_SEM_TAG+TAG
print('\n=== ajuste de hiperparametros (val) ===')
best=None
for lr,it,dep,leaf in [(0.06,400,6,200),(0.04,800,7,150),(0.03,1200,8,100),(0.05,600,None,120)]:
    par=dict(max_iter=it,learning_rate=lr,max_depth=dep,min_samples_leaf=leaf,
             l2_regularization=1.0,random_state=0,categorical_features=[F.index('part')])
    m=HistGradientBoostingClassifier(**par).fit(tr[F],tr.y)
    a=roc_auc_score(va.y,m.predict_proba(va[F])[:,1])
    print(f'  lr={lr} it={it} depth={dep} leaf={leaf} -> val AUC {a:.4f}')
    if best is None or a>best[0]: best=(a,par,m)
m=best[2]; p=np.clip(m.predict_proba(te[F])[:,1],1e-6,1-1e-6)
fin={'modelo':'v2 COMPLETO + ajuste','n_feats':len(F),'AUC':roc_auc_score(te.y,p),
     'logloss':log_loss(te.y,p),'brier':brier_score_loss(te.y,p),'ECE':ece(te.y.values,p)}
rows.append(fin); ps['final']=p
print('\n=== final ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print(f"\nganho v1 -> final: {fin['AUC']-rows[0]['AUC']:+.4f} AUC | "
      f"{rows[0]['logloss']-fin['logloss']:+.4f} log-loss")

print('\n=== cold start (respostas previas do aluno) ===')
for lo,hi,lab in [(0,4,'1a a 5a'),(5,19,'6a a 20a'),(20,99,'21a a 100a'),(100,10**9,'101a+')]:
    k=((te.n_prev>=lo)&(te.n_prev<=hi)).values
    if k.sum()<500: continue
    print(f'  {lab:12s} {k.sum():>9,} | v1 {roc_auc_score(te.y[k],ps["v1 (15 feats) — referencia"][k]):.4f} '
          f'| final {roc_auc_score(te.y[k],p[k]):.4f}')
import pickle; pickle.dump({'model':m,'feats':F,'diff':diff,'global':G},open(f'{HERE}/m2_model_v2.pkl','wb'))
json.dump(rows,open(f'{HERE}/resultados_v2.json','w'),ensure_ascii=False,indent=1,default=str)
print(f"\nm2_model_v2.pkl {os.path.getsize(f'{HERE}/m2_model_v2.pkl')/1e6:.1f} MB")
