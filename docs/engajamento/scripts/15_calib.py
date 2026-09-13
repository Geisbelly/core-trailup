"""A logistica linear perdeu calibracao. Testar formas que consertem."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet'); O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
def wil(k,n,z=1.96):
    if n==0: return (float('nan'),float('nan'))
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h

print('='*74); print('A. QUAL FORMA CALIBRA MELHOR'); print('='*74)
print(f'{"base":<7} {"forma":<38} {"AUC":>7} {"ECE":>7}')
for nome,D in [('EdNet',E),('OULAD',O)]:
    rng=np.random.RandomState(4); m=rng.rand(len(D))<0.7; tr,te=D[m],D[~m]
    def prep(df,modo):
        dr=df.dias_recentes.values*10
        if modo=='linear':
            return np.c_[dr,df.freq.values,df.prof.values]
        if modo=='quad':
            return np.c_[dr,dr**2,df.freq.values,df.prof.values]
        if modo=='dummies':
            X=np.zeros((len(df),11))
            X[np.arange(len(df)),np.clip(dr.astype(int),0,10)]=1
            return np.c_[X[:,1:],df.freq.values,df.prof.values]
    for modo in ('linear','quad','dummies'):
        X=prep(tr,modo); Xe=prep(te,modo)
        mu,sd=X.mean(0),X.std(0); sd[sd==0]=1
        w=logistic_fit((X-mu)/sd,tr.ret.values)
        p=logistic_pred(w,(Xe-mu)/sd)
        print(f'{nome:<7} {modo:<38} {roc_auc_score(te.ret.values,p):>7.3f} {ece(te.ret.values,p):>7.3f}')

print('\n'+'='*74); print('B. TABELA DIRETA: retencao por dias ativos nos ultimos 10'); print('='*74)
print('  (sem modelo nenhum - so a contagem. E o que o modulo vai usar.)\n')
for nome,D in [('EdNet',E),('OULAD',O)]:
    print(f'  -- {nome} (base {D.ret.mean():.1%}, n={len(D):,}) --')
    dr=(D.dias_recentes*10).round().astype(int)
    for k in range(0,11):
        m=dr==k
        if m.sum()<30: continue
        s=D.ret[m]; lo,hi=wil(int(s.sum()),len(s))
        print(f'     {k:>2} dias: retencao {s.mean():>5.1%} [{lo:.1%},{hi:.1%}]  n={int(m.sum()):>6,}')

print('\n'+'='*74); print('C. BANDAS (por quantil, tratando empates)'); print('='*74)
for nome,D in [('EdNet',E),('OULAD',O)]:
    print(f'  -- {nome} --')
    for e in ['dias_recentes','freq','prof']:
        r=D[e].rank(pct=True,method='average')
        f=np.where(r<=1/3,0,np.where(r<=2/3,1,2))
        out=[]
        for k,lab in [(0,'baixo'),(1,'medio'),(2,'alto')]:
            s=D.ret[f==k]
            if len(s)==0: out.append(f'{lab} -'); continue
            lo,hi=wil(int(s.sum()),len(s))
            out.append(f'{lab} {s.mean():5.1%} [{lo:.1%},{hi:.1%}] n={len(s):,}')
        print(f'     {e:<14} | '+' | '.join(out))
