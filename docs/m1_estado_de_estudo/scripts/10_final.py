"""M1 final: logistica sobre as duas bases + interacao. Ganho = calibracao."""
import os, pickle, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
d=R[R.difficulty.notna()].sort_values(['actor','t0']).reset_index(drop=True)
d['y']=(d.difficulty>=4).astype(int); G=d.y.mean(); KT,KA=25,3
def ece(y,p,b=10):
    e=0.0;i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
def bases(tr,te):
    bt=tr.groupby('assignment').y.agg(['sum','size'])
    pt=te.assignment.map((bt['sum']+G*KT)/(bt['size']+KT)).fillna(G).values
    te=te.sort_values(['actor','t0'])
    cs=te.groupby('actor').y.cumsum()-te.y; cn=te.groupby('actor').cumcount()
    pa=((cs+G*KA)/(cn+KA)).values
    return pt,pa,cn.values,te.index.values
P=np.zeros(len(d)); NPREV=np.zeros(len(d))
for tr_i,te_i in GroupKFold(5).split(d,d.y,groups=d.actor):
    tr,te=d.iloc[tr_i],d.iloc[te_i]
    ptr,par,_,itr=bases(tr,tr); pte,pae,cn,ite=bases(tr,te)
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
    m.fit(np.c_[ptr,par,ptr*par], d.y.values[d.index.get_indexer(itr)])
    P[d.index.get_indexer(ite)]=m.predict_proba(np.c_[pte,pae,pte*pae])[:,1]
    NPREV[d.index.get_indexer(ite)]=cn
y=d.y.values; P=np.clip(P,1e-6,1-1e-6)
print(f'M1 final | AUC {roc_auc_score(y,P):.4f} | AP {average_precision_score(y,P):.4f} | '
      f'Brier {brier_score_loss(y,P):.4f} | ECE {ece(y,P):.4f}')
print('\n=== por quantidade de check-ins do proprio aluno ===')
for lo,hi,lab in [(0,0,'0'),(1,2,'1-2'),(3,5,'3-5'),(6,1000,'6+')]:
    m=(NPREV>=lo)&(NPREV<=hi)
    if m.sum()<40: continue
    try: a=roc_auc_score(y[m],P[m])
    except ValueError: a=float('nan')
    print(f'  {lab:5s} {m.sum():>5,} casos | AUC {a:.4f} | ECE {ece(y[m],P[m]):.4f}')
bt=d.groupby('assignment').y.agg(['sum','size'])
mf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))
pt,pa,_,idx=bases(d,d); mf.fit(np.c_[pt,pa,pt*pa],d.y.values[d.index.get_indexer(idx)])
pickle.dump({'model':mf,'dif_texto':((bt['sum']+G*KT)/(bt['size']+KT)).to_dict(),
             'base_global':float(G),'k_texto':KT,'k_aluno':KA},open(f'{HERE}/m1_model.pkl','wb'))
print(f"\nm1_model.pkl {os.path.getsize(f'{HERE}/m1_model.pkl')/1e3:.0f} KB")
