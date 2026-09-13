"""M1 v2 - leitura = (aluno, sessao, texto), rotulo declarado pelo aluno."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
R['log_dur']=np.log1p(R.dur); R['log_ativo']=np.log1p(R.ativo_sec)
R['log_gapmed']=np.log1p(R.gap_med); R['log_gapmax']=np.log1p(R.gap_max)
R['press_min']=R.n_press/(R.dur/60); R['focus_min']=R.n_focus/(R.dur/60)
R['token_min']=R.n_token/(R.dur/60)

TRANSF=['log_dur','log_ativo','ativo_ratio','ev_min','press_min','focus_min',
        'log_gapmed','log_gapmax','frac_pausa','n_dialog','revisita']
ARES=['token_min','n_explic']

def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def regra(d):
    s=np.full(len(d),0.5)
    s=np.where(d.ativo_ratio<0.35,0.84,s)
    s=np.where((d.ev_min>=4)&(d.dur<35),0.63,s)
    s=np.where((d.ativo_ratio>=0.65)&(d.dur>=60),0.12,s)
    return s

def cv(model,X,y,g):
    p=cross_val_predict(model,X,y,groups=g,cv=GroupKFold(5),method='predict_proba')[:,1]
    return p

MODELOS=[('logistica',make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.3))),
         ('boosting',HistGradientBoostingClassifier(max_iter=250,learning_rate=0.05,max_depth=3,
                     min_samples_leaf=40,l2_regularization=5.0,random_state=0)),
         ('random forest',RandomForestClassifier(n_estimators=400,max_depth=6,min_samples_leaf=20,
                     random_state=0,n_jobs=-1))]

for alvo,mk,desc in [('travado',lambda d:(d.difficulty>=4).astype(int),'dificuldade declarada >= 4'),
                     ('muito travado',lambda d:(d.difficulty>=5).astype(int),'dificuldade declarada = 5'),
                     ('desinteresse',lambda d:(d.interestingness<=2).astype(int),'interesse declarado <= 2')]:
    col='difficulty' if 'travado' in alvo else 'interestingness'
    d=R[R[col].notna()].copy(); y=mk(d).values; g=d.actor.values
    print(f'\n{"="*76}\nALVO: {alvo} ({desc})')
    print(f'{len(d):,} leituras | {d.actor.nunique()} alunos | positivos {y.mean():.1%}')
    rows=[]
    pr=np.clip(regra(d),0.01,0.99)
    rows.append({'modelo':'REGRA TrailUp (analogo S2/S3/S5)','AUC':roc_auc_score(y,pr),
                 'AP':average_precision_score(y,pr),'Brier':brier_score_loss(y,pr),'ECE':ece(y,pr)})
    for c,lab in [('dur','so duracao'),('ativo_ratio','so ativo_ratio'),('revisita','so revisitas')]:
        v=d[c].values.astype(float)
        rows.append({'modelo':f'{lab} (1 feature)','AUC':max(roc_auc_score(y,v),roc_auc_score(y,-v)),
                     'AP':np.nan,'Brier':np.nan,'ECE':np.nan})
    for nm,m in MODELOS:
        p=cv(m,d[TRANSF],y,g)
        rows.append({'modelo':f'M1 {nm} (transferivel, {len(TRANSF)} feats)','AUC':roc_auc_score(y,p),
                     'AP':average_precision_score(y,p),'Brier':brier_score_loss(y,p),'ECE':ece(y,p)})
    p=cv(MODELOS[0][1],d[TRANSF+ARES],y,g)
    rows.append({'modelo':'M1 logistica (+ consulta de palavra: so ARES)','AUC':roc_auc_score(y,p),
                 'AP':average_precision_score(y,p),'Brier':brier_score_loss(y,p),'ECE':ece(y,p)})
    print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())

# coeficientes: o que o modelo aprendeu
d=R[R.difficulty.notna()].copy(); y=(d.difficulty>=4).astype(int).values
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.3)).fit(d[TRANSF],y)
co=pd.Series(lr[-1].coef_[0],index=TRANSF).sort_values()
print(f'\n=== coeficientes (alvo travado), padronizados ===')
print(co.round(3).to_string())
