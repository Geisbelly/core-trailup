"""Acuracia depende do LIMIAR. 92,9% pode ser um modelo que nunca alerta ninguem."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet').copy(); O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet').copy()
F=['dias_recentes','freq','prof']
for D in (E,O):
    for c in F: D[c]=D[c].replace([np.inf,-np.inf],np.nan)
    D.dropna(subset=F+['ret'],inplace=True)
rng=np.random.RandomState(7)
E['m']=rng.rand(len(E))<0.7; O['m']=rng.rand(len(O))<0.7
def treina(tr,te,cols=F):
    X=tr[cols].values.astype(float); Xe=te[cols].values.astype(float)
    mu,sd=X.mean(0),X.std(0); sd[sd==0]=1
    w=logistic_fit((X-mu)/sd,tr.ret.values.astype(float))
    return logistic_pred(w,(Xe-mu)/sd)

def wil(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h

for nome,D in [('EdNet',E),('OULAD',O)]:
    tr,te=D[D.m],D[~D.m]
    p=treina(tr,te); y=te.ret.values.astype(float)
    # ALVO E "VAI SAIR": inverte, porque a acao e sobre quem sai
    risco=1-p; saiu=1-y
    print('='*78); print(f'{nome}  (n teste {len(te):,} | saem {saiu.mean():.1%} | AUC {roc_auc_score(saiu,risco):.3f})'); print('='*78)
    print(f'  {"limiar":>22} {"alerta em":>10} {"acuracia":>9} {"bal.acc":>8} {"precisao":>9} {"IC95":>16} {"cobertura":>10}')
    lims=[('0,5 (padrao)',0.5),('taxa base',1-y.mean())]
    for q in (0.05,0.10,0.20,0.30):
        lims.append((f'top {q:.0%} de risco',float(np.quantile(risco,1-q))))
    melhor=max(np.linspace(0.05,0.95,91),key=lambda t:((risco>=t)==(saiu==1)).mean()*0+ (
        (((risco>=t)&(saiu==1)).sum()/max((saiu==1).sum(),1)+((risco<t)&(saiu==0)).sum()/max((saiu==0).sum(),1))/2))
    lims.append(('melhor bal.acc',float(melhor)))
    for lab,t in lims:
        a=risco>=t
        acc=((a.astype(int))==saiu).mean()
        sens=((a)&(saiu==1)).sum()/max((saiu==1).sum(),1)
        espec=((~a)&(saiu==0)).sum()/max((saiu==0).sum(),1)
        k=int(((a)&(saiu==1)).sum()); n=int(a.sum())
        prec=k/n if n else float('nan'); lo,hi=wil(k,n) if n else (float('nan'),)*2
        print(f'  {lab:>22} {a.mean():>10.1%} {acc:>9.1%} {(sens+espec)/2:>8.1%} {prec:>9.1%} [{lo:.1%},{hi:.1%}] {sens:>10.1%}')
    print(f'  {"nunca alertar":>22} {0.0:>10.1%} {(1-saiu.mean()):>9.1%} {50.0:>7.1f}% {"-":>9} {"":>16} {0.0:>10.1%}')
    print(f'  {"sempre alertar":>22} {1.0:>10.1%} {saiu.mean():>9.1%} {50.0:>7.1f}% {saiu.mean():>9.1%} {"":>16} {100.0:>9.1f}%')
    print()
