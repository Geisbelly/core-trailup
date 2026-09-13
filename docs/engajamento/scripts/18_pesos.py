"""Um so ordenador, treinado nas duas bases, com padronizacao POR COORTE.
Depois cada coorte calibra o proprio limiar. Extrai os pesos para o modulo."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet').copy(); O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet').copy()
F=['dias_recentes','freq','prof']
for D in (E,O):
    for c in F: D[c]=D[c].replace([np.inf,-np.inf],np.nan)
    D.dropna(subset=F+['ret'],inplace=True)

def z_coorte(D,cols):
    """padroniza DENTRO da coorte - e o que torna os pesos comparaveis entre bases"""
    X=D[cols].values.astype(float)
    return (X-X.mean(0))/np.where(X.std(0)==0,1,X.std(0))

rng=np.random.RandomState(7)
E['m']=rng.rand(len(E))<0.7; O['m']=rng.rand(len(O))<0.7
ZE,ZO=z_coorte(E,F),z_coorte(O,F)
Xtr=np.vstack([ZE[E.m.values],ZO[O.m.values]])
ytr=np.concatenate([E.ret.values[E.m.values],O.ret.values[O.m.values]]).astype(float)
w=logistic_fit(Xtr,ytr)
print('pesos do ordenador comum (z dentro da coorte):')
for n,v in zip(['intercepto']+F,w): print(f'  {n:>14}: {v:+.4f}')

print('\n=== AUC do ordenador comum, por base (so os 30% de teste) ===')
for nome,D,Z in [('EdNet',E,ZE),('OULAD',O,ZO)]:
    te=~D.m.values
    p=logistic_pred(w,Z[te]); y=D.ret.values[te].astype(float)
    # comparacao: modelo treinado so naquela base
    wi=logistic_fit(Z[D.m.values],D.ret.values[D.m.values].astype(float))
    pi=logistic_pred(wi,Z[te])
    print(f'  {nome:>6}: comum {roc_auc_score(y,p):.3f} | so a propria base {roc_auc_score(y,pi):.3f} | n={te.sum():,}')

print('\n=== e num teste mais duro: treinar em UMA base, ordenar a OUTRA ===')
for tri,tei,lab in [((E,ZE),(O,ZO),'EdNet -> OULAD'),((O,ZO),(E,ZE),'OULAD -> EdNet')]:
    (Dt,Zt),(Dv,Zv)=tri,tei
    wo=logistic_fit(Zt,Dt.ret.values.astype(float))
    print(f'  {lab:>16}: AUC {roc_auc_score(Dv.ret.values.astype(float),logistic_pred(wo,Zv)):.3f}')

print('\n=== o limiar que cada coorte precisa (mesmo ordenador) ===')
print(f'  {"base":>7} {"sai":>6} {"limiar p/ alertar 10%":>22} {"precisao":>9} {"cobertura":>10}')
for nome,D,Z in [('EdNet',E,ZE),('OULAD',O,ZO)]:
    te=~D.m.values
    risco=1-logistic_pred(w,Z[te]); saiu=1-D.ret.values[te].astype(float)
    t=float(np.quantile(risco,0.90)); a=risco>=t
    prec=((a)&(saiu==1)).sum()/max(a.sum(),1); cob=((a)&(saiu==1)).sum()/max(saiu.sum(),1)
    print(f'  {nome:>7} {saiu.mean():>6.1%} {t:>22.3f} {prec:>9.1%} {cob:>10.1%}')
print('  (o mesmo ordenador, dois limiares completamente diferentes - e o ponto)')
np.save(f'{HERE}/pesos_comuns.npy',w)
