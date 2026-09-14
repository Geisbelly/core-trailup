"""M2 v3: rotulo corrigido + latencia + hesitacao. Dois alvos, dois modelos."""
import os,sys,gc,json,pickle,numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss,average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ.get('TRAILUP_OUTPUT_DIR', HERE)
ALVO=sys.argv[1]; CONF=sys.argv[2] if len(sys.argv)>2 else 'v3'
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{OUT}/features_v3.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict()
s.update(dict.fromkeys(u[:int(.8*n)],'train')); s.update(dict.fromkeys(u[int(.8*n):int(.9*n)],'val'))
df['split']=df.user.map(s); m_tr=(df.split=='train').values
g=df.loc[m_tr].groupby('qidx').y.agg(['mean','size']); G=float(df.loc[m_tr,'y'].mean())
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
df['q_dif']=df.qidx.map(diff).fillna(G).astype(np.float32)
df['q_n']=np.log1p(df.qidx.map(g['size']).fillna(0)).astype(np.float32)
BASE=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
      'dt_log','dt_part_log','seen_before','ntags','diagnosis','part','q_dif','q_n',
      'acc5','acc10','n_sessao','pos_sessao']
NOVO=['log_lat','log_lat1','lat_rel','lat_med_prev','n_trocas','trocas_prev',
      'log_expl','n_aulas','n_quits','mobile','q_no_bundle']
F=BASE if CONF=='base' else BASE+NOVO
if ALVO=='gate':
    df=df.sort_values(['user','part']).reset_index(drop=True)
    fut=df.groupby(['user','part'],sort=False).y.transform(
        lambda s:s.shift(-1).rolling(10,min_periods=10).mean().shift(-9))
    df['t']=(fut<0.5).astype('float'); df=df[fut.notna()]
else:
    df['t']=df.y.astype(float)
# BLINDAGEM: estas so existem DEPOIS que o aluno responde a questao atual.
# Usa-las para prever a propria resposta e vazamento.
POSTERIOR={'n_trocas','log_lat','log_lat1','lat_rel','lat_final','lat_1a','q_no_bundle'}
assert ALVO in ('proxima','gate'), f'alvo invalido: {ALVO}'
assert CONF in ('base','v3'), f'conf invalida: {CONF}'
if ALVO=='proxima':
    F=[c for c in F if c not in POSTERIOR]
    print(f'  (removidas por serem posteriores a resposta: {sorted(POSTERIOR & set(BASE+NOVO))})',flush=True)
tr,va,te=(df[df.split==k] for k in ('train','val','test'))
print(f'alvo={ALVO} conf={CONF} feats={len(F)} | treino {len(tr):,} teste {len(te):,} | positivos {te.t.mean():.1%}',flush=True)
del df; gc.collect()
def ece(y,p,b=10):
    e=0.0;i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
best=None
for lr,it,dep,leaf in [(0.06,400,6,200),(0.04,700,7,150)]:
    m=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,max_depth=dep,min_samples_leaf=leaf,
        l2_regularization=1.0,random_state=0,categorical_features=[F.index('part')]).fit(tr[F],tr.t)
    a=roc_auc_score(va.t,m.predict_proba(va[F])[:,1])
    if best is None or a>best[0]: best=(a,m)
m=best[1]; p=np.clip(m.predict_proba(te[F])[:,1],1e-6,1-1e-6); y=te.t.values
r={'alvo':ALVO,'conf':CONF,'n_feats':len(F),'AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p),
   'logloss':log_loss(y,p),'brier':brier_score_loss(y,p),'ECE':ece(y,p)}
print(json.dumps(r,ensure_ascii=False),flush=True)
if ALVO=='gate' and CONF=='v3':
    print('\n=== precisao por taxa de disparo ===',flush=True)
    for taxa in (0.05,0.10,0.15,0.20):
        t=np.quantile(p,1-taxa); k=p>=t
        print(f'  dispara {taxa:>4.0%} | limiar {t:.2f} | precisao {y[k].mean():>6.1%} | cobertura {y[k].sum()/y.sum():>5.0%}',flush=True)
    pickle.dump({'model':m,'feats':F,'diff':diff,'global':G},open(f'{OUT}/m2_gate_v3.pkl','wb'))
if ALVO=='proxima' and CONF=='v3':
    pickle.dump({'model':m,'feats':F,'diff':diff,'global':G},open(f'{OUT}/m2_model_v3.pkl','wb'))
