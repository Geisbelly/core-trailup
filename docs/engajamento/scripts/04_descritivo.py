"""Estatistica descritiva e incerteza - o que faltou no primeiro relatorio."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
A=pd.read_parquet(f'{HERE}/eixos.parquet')
EIXOS=['freq','volume','prof']
print(f'N = {len(A):,} alunos | desfecho positivo (voltou): {A.ret.sum():,} ({A.ret.mean():.1%})\n')

print('=== A. distribuicao de cada eixo, em unidade CRUA ===')
A['dias_ativos_30']=A.freq*30
A['seg_explicacao']=np.expm1(A.prof)
for nome,col,un in [('frequencia','dias_ativos_30','dias ativos em 30'),
                    ('volume','volume','respostas por sessao'),
                    ('profundidade','seg_explicacao','segundos na explicacao')]:
    q=A[col].quantile([.05,.25,.5,.75,.95])
    print(f'  {nome:>13} ({un}): p5 {q[.05]:7.1f} | p25 {q[.25]:7.1f} | mediana {q[.5]:7.1f} | p75 {q[.75]:7.1f} | p95 {q[.95]:7.1f}')

print('\n=== B. AUC com IC 95% (bootstrap, 400 reamostras de alunos) ===')
rng=np.random.RandomState(20260913)
def boot_auc(y,s,B=400):
    y=np.asarray(y); s=np.asarray(s); n=len(y); out=[]
    for _ in range(B):
        i=rng.randint(0,n,n)
        if y[i].sum() in (0,len(i)): continue
        out.append(roc_auc_score(y[i],s[i]))
    return np.percentile(out,[2.5,97.5])
for c in EIXOS+['n','acerto']:
    a=roc_auc_score(A.ret,A[c]); lo,hi=boot_auc(A.ret.values,A[c].values)
    print(f'    {c:>8}: AUC {a:.3f}  IC95 [{lo:.3f}, {hi:.3f}]')

print('\n=== C. taxa de retencao por faixa, com IC 95% (Wilson) ===')
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
CORTES={'freq':(0.100,0.233),'volume':(12.5,19.619),'prof':(2.175,2.865)}
for e in EIXOS:
    lo_,hi_=CORTES[e]
    f=np.where(A[e]<=lo_,'baixo',np.where(A[e]<=hi_,'medio','alto'))
    print(f'  -- {e} --')
    for nome in ['baixo','medio','alto']:
        m=f==nome; k=int(A.ret[m].sum()); n=int(m.sum())
        l,h=wilson(k,n)
        print(f'    {nome:>5}: {k:5,}/{n:5,} = {k/n:5.1%}  IC95 [{l:.1%}, {h:.1%}]')

print('\n=== D. o modelo dos 3 eixos esta CALIBRADO? (30% fora do treino) ===')
msk=rng.rand(len(A))<0.7; tr,te=A[msk],A[~msk]
X=np.log1p(tr[EIXOS].clip(lower=0)); mu,sd=X.mean(),X.std()
w=logistic_fit(((X-mu)/sd).values,tr.ret.values)
Xe=np.log1p(te[EIXOS].clip(lower=0)); p=logistic_pred(w,((Xe-mu)/sd).values)
print(f'    {len(te):,} alunos fora do treino | AUC {roc_auc_score(te.ret.values,p):.3f}')
b=pd.qcut(p,10,labels=False,duplicates='drop')
t=pd.DataFrame({'b':b,'p':p,'y':te.ret.values}).groupby('b').agg(prev=('p','mean'),obs=('y','mean'),n=('y','size'))
ece=float((t.n/t.n.sum()*(t.prev-t.obs).abs()).sum())
for i,r in t.iterrows():
    print(f'      decil {int(i)+1:>2}: previsto {r.prev:5.1%} | observado {r.obs:5.1%} | n={int(r.n):,}')
print(f'    ECE (erro medio de calibracao) = {ece:.3f}')
print('    coeficientes (log-odds por 1 desvio-padrao):')
for nome,val in zip(['intercepto']+EIXOS,w): print(f'      {nome:>10}: {val:+.3f}')
