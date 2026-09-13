"""Treinar com as DUAS bases juntas. Quatro perguntas, nesta ordem:

  A) o modelo junto so esta adivinhando de qual base veio o aluno?
  B) treinando numa e testando na OUTRA, transfere? (o teste que importa)
  C) usar PERCENTIL da propria coorte, em vez de valor absoluto, conserta?
  D) e a acuracia - com a ressalva de que taxa base 30,5% vs 92,9% a torna
     enganosa se lida sozinha.
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet').copy()
O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet').copy()
E['base']='EdNet'; O['base']='OULAD'
F=['dias_recentes','freq','prof']
for D in (E,O):
    for c in F: D[c]=D[c].replace([np.inf,-np.inf],np.nan)
    D.dropna(subset=F+['ret'],inplace=True)
P=pd.concat([E[F+['ret','base']],O[F+['ret','base']]],ignore_index=True)
print(f'EdNet {len(E):,} (retencao {E.ret.mean():.1%}) | OULAD {len(O):,} ({O.ret.mean():.1%}) | junto {len(P):,} ({P.ret.mean():.1%})')

def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
def metricas(y,p,lim=0.5):
    pred=(p>=lim).astype(int)
    acc=(pred==y).mean()
    tp=((pred==1)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
    sens=tp/max((y==1).sum(),1); espec=tn/max((y==0).sum(),1)
    return dict(AUC=roc_auc_score(y,p),acc=acc,bal=(sens+espec)/2,
                sens=sens,espec=espec,ECE=ece(y,p),
                maioria=max(y.mean(),1-y.mean()))
def treina(tr,te,cols):
    X=tr[cols].values.astype(float); Xe=te[cols].values.astype(float)
    mu,sd=X.mean(0),X.std(0); sd[sd==0]=1
    w=logistic_fit((X-mu)/sd,tr.ret.values.astype(float))
    return logistic_pred(w,(Xe-mu)/sd)

print('\n'+'='*78); print('A. O MODELO JUNTO ESTA SO ADIVINHANDO A BASE?'); print('='*78)
# quao bem as features sozinhas separam EdNet de OULAD?
P['e_oulad']=(P.base=='OULAD').astype(int)
rng=np.random.RandomState(11); m=rng.rand(len(P))<0.7
trb,teb=P[m],P[~m]
Xb=trb[F].values.astype(float); mu,sd=Xb.mean(0),Xb.std(0); sd[sd==0]=1
wb=logistic_fit((Xb-mu)/sd,trb.e_oulad.values.astype(float))
pb=logistic_pred(wb,(teb[F].values.astype(float)-mu)/sd)
print(f'  prever DE QUAL BASE o aluno veio, so com os 3 eixos: AUC {roc_auc_score(teb.e_oulad.values,pb):.3f}')
print('  (perto de 1,0 = as bases sao separaveis pelas features, e um modelo junto')
print('   pode acertar so por saber a base, nao por medir engajamento)')

print('\n'+'='*78); print('B. TREINAR NUMA, TESTAR NA OUTRA (valores absolutos)'); print('='*78)
print(f'  {"treino -> teste":<22} {"AUC":>7} {"acuracia":>9} {"bal.acc":>8} {"ECE":>7} {"maioria":>8}')
for tr_,te_,lab in [(E,O,'EdNet -> OULAD'),(O,E,'OULAD -> EdNet')]:
    p=treina(tr_,te_,F); r=metricas(te_.ret.values.astype(float),p)
    print(f'  {lab:<22} {r["AUC"]:>7.3f} {r["acc"]:>9.1%} {r["bal"]:>8.1%} {r["ECE"]:>7.3f} {r["maioria"]:>8.1%}')

print('\n'+'='*78); print('C. MESMO TESTE, COM PERCENTIL DENTRO DA PROPRIA COORTE'); print('='*78)
for D in (E,O):
    for c in F: D[c+'_pct']=D[c].rank(pct=True)
FP=[c+'_pct' for c in F]
print(f'  {"treino -> teste":<22} {"AUC":>7} {"acuracia":>9} {"bal.acc":>8} {"ECE":>7}')
for tr_,te_,lab in [(E,O,'EdNet -> OULAD'),(O,E,'OULAD -> EdNet')]:
    p=treina(tr_,te_,FP); r=metricas(te_.ret.values.astype(float),p)
    print(f'  {lab:<22} {r["AUC"]:>7.3f} {r["acc"]:>9.1%} {r["bal"]:>8.1%} {r["ECE"]:>7.3f}')

print('\n'+'='*78); print('D. MODELO JUNTO vs MODELO DE CADA BASE (mesmo conjunto de teste)'); print('='*78)
rng=np.random.RandomState(7)
E['m']=rng.rand(len(E))<0.7; O['m']=rng.rand(len(O))<0.7
trE,teE=E[E.m],E[~E.m]; trO,teO=O[O.m],O[~O.m]
junto_tr=pd.concat([trE,trO],ignore_index=True)
junto_tr_pct=junto_tr.copy()
linhas=[]
for nome,te_ in [('EdNet',teE),('OULAD',teO)]:
    proprio=treina(trE if nome=='EdNet' else trO,te_,F)
    junto=treina(junto_tr,te_,F)
    junto_p=treina(junto_tr,te_,FP)
    for lab,p in [('so a propria base',proprio),('junto (absoluto)',junto),('junto (percentil)',junto_p)]:
        r=metricas(te_.ret.values.astype(float),p); r['base']=nome; r['modelo']=lab
        linhas.append(r)
T=pd.DataFrame(linhas)
print(f'  {"teste":<7} {"modelo":<20} {"AUC":>7} {"acuracia":>9} {"bal.acc":>8} {"sens":>7} {"espec":>7} {"ECE":>7}')
for _,r in T.iterrows():
    print(f'  {r.base:<7} {r.modelo:<20} {r.AUC:>7.3f} {r.acc:>9.1%} {r.bal:>8.1%} {r.sens:>7.1%} {r.espec:>7.1%} {r.ECE:>7.3f}')
print(f'\n  referencia (chutar a classe maioria): EdNet {max(teE.ret.mean(),1-teE.ret.mean()):.1%} | OULAD {max(teO.ret.mean(),1-teO.ret.mean()):.1%}')
