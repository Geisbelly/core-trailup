import os, sys, gc, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
HERE=os.path.dirname(os.path.abspath(__file__))
CONF=sys.argv[1]
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features_v2.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict()
s.update(dict.fromkeys(u[:int(.8*n)],'train')); s.update(dict.fromkeys(u[int(.8*n):int(.9*n)],'val'))
sp=df.user.map(s).values
V1=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
    'dt_log','dt_part_log','seen_before','ntags','diagnosis','part','q_dif','q_n']
NEW=['acc5','acc10','n_sessao','pos_sessao','dt_rel','bundle_pos','acc_bundle']
TAG=['tag_acc','tag_n','tag_acc_min','tag_acc_max','tag_ewma','tag_novo']
SETS={'v1':V1,'v2_sem_tag':V1+NEW,'v1_tag':V1+TAG,'v2':V1+NEW+TAG}
F=SETS[CONF.split('|')[0]]
# dificuldade so do treino
m_tr=sp=='train'
g=df.loc[m_tr].groupby('qidx').y.agg(['mean','size']); G=float(df.loc[m_tr,'y'].mean())
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
df['q_dif']=df.qidx.map(diff).fillna(G).astype(np.float32)
df['q_n']=np.log1p(df.qidx.map(g['size']).fillna(0)).astype(np.float32)
X=df[F].astype(np.float32); y=df.y.values.astype(np.int8)
del df; gc.collect()
par=dict(max_iter=400,learning_rate=0.06,max_depth=6,min_samples_leaf=200,
         l2_regularization=1.0,random_state=0,categorical_features=[F.index('part')])
if '|' in CONF:
    lr,it,dep,leaf=CONF.split('|')[1].split(',')
    par.update(learning_rate=float(lr),max_iter=int(it),
               max_depth=None if dep=='none' else int(dep),min_samples_leaf=int(leaf))
m=HistGradientBoostingClassifier(**par).fit(X[m_tr],y[m_tr])
def ece(yy,pp,b=10):
    e=0.0; i=np.digitize(pp,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        mm=i==k
        if mm.sum(): e+=mm.mean()*abs(yy[mm].mean()-pp[mm].mean())
    return e
out={'conf':CONF,'n_feats':len(F)}
for nome,msk in [('val',sp=='val'),('test',sp=='test')]:
    p=np.clip(m.predict_proba(X[msk])[:,1],1e-6,1-1e-6); yy=y[msk]
    out[nome]={'AUC':roc_auc_score(yy,p),'logloss':log_loss(yy,p),
               'brier':brier_score_loss(yy,p),'ECE':ece(yy,p)}
    if nome=='test': np.save(f'{HERE}/p_{CONF.split("|")[0]}.npy',p)
print(json.dumps(out,ensure_ascii=False),flush=True)
import pickle
if CONF.startswith('v2'): pickle.dump({'model':m,'feats':F,'diff':diff,'global':G},open(f'{HERE}/m2_v2_{CONF.replace("|","_").replace(",","_")}.pkl','wb'))
