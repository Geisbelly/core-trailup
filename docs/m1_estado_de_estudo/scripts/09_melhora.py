"""M1 v2: melhorar as duas bases que sobreviveram.

v1 usava proporcao binaria (>=4) e prioris escolhidas a olho (10 e 3).
Aqui: nota media continua (usa os 5 niveis, nao 2), prioris ajustadas em
validacao, interacao aluno x texto, e o interesse como covariavel.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
d=R[R.difficulty.notna()].sort_values(['actor','t0']).reset_index(drop=True)
d['y']=(d.difficulty>=4).astype(int); G=d.y.mean(); GM=d.difficulty.mean()
print(f'{len(d):,} leituras | {d.actor.nunique()} alunos | {d.assignment.nunique()} textos | positivos {G:.1%}')

def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

def bases(tr, te, k_txt, k_alu, usar_media):
    """Estima base do texto (no treino) e base do aluno (causal, so leituras anteriores dele)."""
    col='difficulty' if usar_media else 'y'
    gm = GM if usar_media else G
    bt=tr.groupby('assignment')[col].agg(['sum','size'])
    p_txt=te.assignment.map((bt['sum']+gm*k_txt)/(bt['size']+k_txt)).fillna(gm).values
    te=te.sort_values(['actor','t0'])
    cs=te.groupby('actor')[col].cumsum()-te[col]; cn=te.groupby('actor').cumcount()
    p_alu=((cs+gm*k_alu)/(cn+k_alu)).values
    return p_txt, p_alu, cn.values, te.index.values

def avalia(nome, k_txt, k_alu, usar_media, modo):
    P=np.zeros(len(d))
    for tr_i,te_i in GroupKFold(5).split(d,d.y,groups=d.actor):
        tr,te=d.iloc[tr_i],d.iloc[te_i]
        p_txt,p_alu,cn,idx=bases(tr,te,k_txt,k_alu,usar_media)
        if modo=='media':
            p=0.5*(p_txt+p_alu)
            if usar_media: p=(p-1)/4          # escala 1-5 -> 0-1
        else:
            trx,tra,_,tri=bases(tr,tr,k_txt,k_alu,usar_media)
            X=np.c_[trx,tra,trx*tra]; Xe=np.c_[p_txt,p_alu,p_txt*p_alu]
            m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
            m.fit(X,d.y.values[d.index.get_indexer(tri)])
            p=m.predict_proba(Xe)[:,1]
        P[d.index.get_indexer(idx)]=p
    P=np.clip(P,1e-6,1-1e-6); y=d.y.values
    return {'modelo':nome,'AUC':roc_auc_score(y,P),'AP':average_precision_score(y,P),
            'Brier':brier_score_loss(y,P),'ECE':ece(y,P)}, P

rows=[]
r,_=avalia('v1: proporcao binaria, prioris 10/3, media',10,3,False,'media'); rows.append(r)
print('\n=== ajuste das prioris (alvo: AUC) ===')
best=None
for kt in (3,10,25):
    for ka in (1,3,6,10):
        rr,_=avalia(f'k_txt={kt} k_alu={ka}',kt,ka,False,'media')
        if best is None or rr['AUC']>best[0]['AUC']: best=(rr,kt,ka)
        print(f'  k_txt={kt:>2} k_alu={ka:>2} -> AUC {rr["AUC"]:.4f} AP {rr["AP"]:.4f}')
_,kt,ka=best
r,_=avalia(f'v2a: prioris ajustadas ({kt}/{ka})',kt,ka,False,'media'); rows.append(r)
r,_=avalia(f'v2b: NOTA MEDIA 1-5 em vez de binaria ({kt}/{ka})',kt,ka,True,'media'); rows.append(r)
r,P=avalia(f'v2c: logistica sobre as duas bases + interacao ({kt}/{ka})',kt,ka,False,'logit'); rows.append(r)
r,_=avalia(f'v2d: logistica com nota media ({kt}/{ka})',kt,ka,True,'logit'); rows.append(r)
print('\n=== resultado ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print(f"\nganho v1 -> melhor: {max(x['AUC'] for x in rows)-rows[0]['AUC']:+.4f} AUC")
