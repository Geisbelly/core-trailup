"""Desenho recalibrado, validado nas duas bases. Bandas, IC, calibracao."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet'); O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
BASES=[('EdNet',E),('OULAD',O)]
def boot(y,s,B=300,seed=1):
    r=np.random.RandomState(seed); o=[]
    for _ in range(B):
        i=r.randint(0,len(y),len(y))
        if y[i].sum() in (0,len(i)): continue
        o.append(roc_auc_score(y[i],s[i]))
    return np.percentile(o,[2.5,97.5])
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h

print('='*74); print('A. CADA MEDIDA, NAS DUAS BASES, COM IC'); print('='*74)
print(f'{"medida":<16} {"EdNet AUC":>11} {"IC95":>16} {"OULAD AUC":>11} {"IC95":>16} {"replica?":>9}')
for c in ['dias_recentes','freq','tend','prof','volume']:
    row=[]
    for _,D in BASES:
        a=roc_auc_score(D.ret,D[c]); lo,hi=boot(D.ret.values,D[c].values)
        row.append((a,lo,hi))
    (a1,l1,h1),(a2,l2,h2)=row
    rep='SIM' if (a1-0.5)*(a2-0.5)>0 and min(abs(a1-0.5),abs(a2-0.5))>0.03 else 'NAO'
    print(f'{c:<16} {a1:>11.3f} [{l1:.3f},{h1:.3f}] {a2:>11.3f} [{l2:.3f},{h2:.3f}] {rep:>9}')

print('\n'+'='*74); print('B. MODELO RECALIBRADO vs ANTERIOR (fora do treino)'); print('='*74)
res={}
for nome,D in BASES:
    rng=np.random.RandomState(4); m=rng.rand(len(D))<0.7; tr,te=D[m],D[~m]
    def fit(cols):
        X=tr[cols].replace([np.inf,-np.inf],0).fillna(0); Xe=te[cols].replace([np.inf,-np.inf],0).fillna(0)
        mu,sd=X.mean(),X.std().replace(0,1)
        w=logistic_fit(((X-mu)/sd).values,tr.ret.values)
        p=logistic_pred(w,((Xe-mu)/sd).values)
        return roc_auc_score(te.ret.values,p),p,te.ret.values
    res[nome]={}
    for lab,cols in [('anterior: 3 eixos',['freq','volume','prof']),
                     ('novo: recencia + freq',['dias_recentes','freq']),
                     ('novo + profundidade',['dias_recentes','freq','prof'])]:
        a,p,y=fit(cols); res[nome][lab]=(a,p,y)
print(f'{"modelo":<26} {"EdNet":>8} {"OULAD":>8} {"ganho medio":>12}')
for lab in ['anterior: 3 eixos','novo: recencia + freq','novo + profundidade']:
    a1=res['EdNet'][lab][0]; a2=res['OULAD'][lab][0]
    base=(res['EdNet']['anterior: 3 eixos'][0]+res['OULAD']['anterior: 3 eixos'][0])/2
    print(f'{lab:<26} {a1:>8.3f} {a2:>8.3f} {((a1+a2)/2-base):>+12.3f}')

print('\n'+'='*74); print('C. CALIBRACAO DO MODELO NOVO (decis, fora do treino)'); print('='*74)
def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
for nome in ['EdNet','OULAD']:
    a,p,y=res[nome]['novo + profundidade']
    b=pd.qcut(p,10,labels=False,duplicates='drop')
    t=pd.DataFrame({'b':b,'p':p,'y':y}).groupby('b').agg(prev=('p','mean'),obs=('y','mean'),n=('y','size'))
    print(f'  -- {nome} (AUC {a:.3f}, ECE {ece(y,p):.3f}) --')
    for i,r in t.iterrows():
        print(f'     decil {int(i)+1:>2}: previsto {r.prev:5.1%} | observado {r.obs:5.1%} | n={int(r.n):,}')

print('\n'+'='*74); print('D. BANDAS POR PERCENTIL DA COORTE (o que se entrega)'); print('='*74)
for nome,D in BASES:
    print(f'  -- {nome} (retencao base {D.ret.mean():.1%}) --')
    for e in ['dias_recentes','freq','prof']:
        q=D[e].quantile([1/3,2/3]).values
        f=np.where(D[e]<=q[0],0,np.where(D[e]<=q[1],1,2))
        out=[]
        for k,lab in [(0,'baixo'),(1,'medio'),(2,'alto')]:
            s=D.ret[f==k]; lo,hi=wil(int(s.sum()),len(s))
            out.append(f'{lab} {s.mean():5.1%} [{lo:.1%},{hi:.1%}]')
        print(f'     {e:<14} cortes {q[0]:7.3f}/{q[1]:7.3f} | '+' | '.join(out))
