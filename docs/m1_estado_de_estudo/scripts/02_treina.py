"""M1 - faixa de estudo, treinado no rotulo DECLARADO pelo aluno (ARES).

Alvo principal: `travado` = o aluno declarou dificuldade >= 4 logo apos ler.
E o unico rotulo de estado, declarado e nao inferido, disponivel sem banco proprio.

Amostra pequena (197 alunos): GroupKFold por aluno, out-of-fold, modelos regularizados.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
ep=pd.read_parquet(f'{HERE}/episodios.parquet')
ep=ep[ep.dur>=15]                      # episodio com leitura de fato
ep['ativo']=1-ep.frac_longa            # proxy de active_ratio (fracao fora de pausa longa)
ep['log_dur']=np.log1p(ep.dur); ep['log_gap']=np.log1p(ep.gap_med)
ep['press_min']=ep.n_press/(ep.dur/60); ep['focus_min']=ep.n_focus/(ep.dur/60)

TRANSFERIVEL=['log_dur','ev_min','press_min','focus_min','ativo','log_gap','gap_p90','n_dialog']
SO_ARES=['n_token','n_explic']         # consulta de palavra: nao existe no TrailUp

def ece(y,p,bins=10):
    e=0.0; idx=np.digitize(p,np.linspace(0,1,bins+1)[1:-1])
    for b in range(bins):
        m=idx==b
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def regra_trailup(d):
    """Analogo mais fiel possivel das regras S2/S3/S5 com o que o ARES tem.
    anomalo: pouca atividade; fragmentado: muitos eventos curtos; fluido: o resto."""
    s=np.full(len(d),0.5)
    s=np.where(d.ativo<0.35, 0.84, s)                       # IsolationForest: anomalo
    s=np.where((d.ev_min>=4)&(d.dur<35), 0.63, s)           # fragmentado
    s=np.where((d.ativo>=0.65)&(d.dur>=60), 0.12, s)        # fluido
    return s

def avalia(nome, X, y, g, modelo):
    cv=GroupKFold(n_splits=5)
    p=cross_val_predict(modelo, X, y, groups=g, cv=cv, method='predict_proba')[:,1]
    return {'modelo':nome,'AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p),
            'Brier':brier_score_loss(y,p),'ECE':ece(y,p)}, p

for alvo,expr,desc in [('travado','difficulty>=4','dificuldade declarada alta'),
                       ('desinteresse','interestingness<=2','interesse declarado baixo')]:
    col='difficulty' if alvo=='travado' else 'interestingness'
    d=ep[ep[col].notna()].copy()
    y=(d[col]>=4).astype(int).values if alvo=='travado' else (d[col]<=2).astype(int).values
    g=d.actor.values
    print(f'\n{"="*74}\nALVO: {alvo}  ({expr})  -  {desc}')
    print(f'{len(d):,} episodios | {d.actor.nunique()} alunos | positivos {y.mean():.1%}')
    rows=[]
    pr=regra_trailup(d)
    rows.append({'modelo':'REGRA TrailUp (analogo S2/S3/S5)','AUC':roc_auc_score(y,pr),
                 'AP':average_precision_score(y,pr),'Brier':brier_score_loss(y,np.clip(pr,0,1)),
                 'ECE':ece(y,np.clip(pr,0,1))})
    rows.append({'modelo':'so duracao do episodio','AUC':roc_auc_score(y,-d.dur.values),
                 'AP':average_precision_score(y,-d.dur.values),'Brier':np.nan,'ECE':np.nan})
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.3))
    hgb=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=3,
        min_samples_leaf=40,l2_regularization=5.0,random_state=0)
    r,_=avalia('M1 logistica (so transferivel)',d[TRANSFERIVEL],y,g,lr); rows.append(r)
    r,_=avalia('M1 boosting (so transferivel)',d[TRANSFERIVEL],y,g,hgb); rows.append(r)
    r,_=avalia('M1 logistica (+ consulta de palavra, so ARES)',d[TRANSFERIVEL+SO_ARES],y,g,lr); rows.append(r)
    r,_=avalia('M1 boosting (+ consulta de palavra, so ARES)',d[TRANSFERIVEL+SO_ARES],y,g,hgb); rows.append(r)
    print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())

# --- alvo comportamental: o assignment acabou abandonado?
print(f'\n{"="*74}\nALVO: disperso  (assignment terminou Expired ou nunca submetido)')
d=ep[ep.status.notna()].copy()
y=d.status.isin(['Expired','Not submitted yet']).astype(int).values
g=d.actor.values
print(f'{len(d):,} episodios | {d.actor.nunique()} alunos | positivos {y.mean():.1%}')
rows=[]
pr=regra_trailup(d)
rows.append({'modelo':'REGRA TrailUp (analogo)','AUC':roc_auc_score(y,pr),'AP':average_precision_score(y,pr)})
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.3))
hgb=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.05,max_depth=3,
    min_samples_leaf=40,l2_regularization=5.0,random_state=0)
for nm,F,m in [('M1 logistica (transferivel)',TRANSFERIVEL,lr),('M1 boosting (transferivel)',TRANSFERIVEL,hgb)]:
    r,_=avalia(nm,d[F],y,g,m); rows.append({'modelo':nm,'AUC':r['AUC'],'AP':r['AP']})
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
