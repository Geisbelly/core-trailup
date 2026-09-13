"""Engajamento em TRES EIXOS, validado contra desfecho FUTURO (nao tautologico).

Janela 1 (dias 0-29 desde a primeira resposta do aluno): calcula os eixos.
Janela 2 (dias 30-59): mede o desfecho.
So entram alunos que tiveram 60 dias de CHANCE (censura tratada).
Nenhuma feature da janela 2 entra no calculo dos eixos.
"""
import os, numpy as np, pandas as pd
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')

HERE=os.path.dirname(os.path.abspath(__file__))
sys_path=HERE
import sys; sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred, spearman
M2=HERE.replace('/engaj','/m2')
DIA=86400*1000.0
W1, W2 = 30, 60

d=pd.read_parquet(f'{M2}/responses_v3.parquet').sort_values(['user','ts'])
fim_base=d.ts.max()
ini=d.groupby('user').ts.transform('min')
d['dia']=(d.ts-ini)/DIA
d['chance']=(fim_base-ini)/DIA

# so quem teve 60 dias de janela disponivel na base
d=d[d.chance>=W2]
a=d[d.dia<W1].copy()
b=d[(d.dia>=W1)&(d.dia<W2)]
print(f'{d.user.nunique():,} alunos com 60 dias de chance | {len(a):,} respostas na janela 1')

# sessoes (gap > 30 min) dentro da janela 1
dt=a.groupby('user').ts.diff()/1000.0
a['sessao']=((dt.isna())|(dt>1800)).groupby(a.user).cumsum()

g=a.groupby('user')
A=pd.DataFrame({
    'n':g.size(),
    'sessoes':g.sessao.max(),
    'dias_ativos':g.dia.apply(lambda s: s.astype(int).nunique()),
    'span':g.dia.max()-g.dia.min(),
    'expl':g.expl_sec.median(),
    'quits':g.n_quits.sum(),
    'acerto':g.correct.mean(),
})
A=A[A.n>=30]

# --- os tres eixos, definidos para NAO serem funcao um do outro ---
A['freq']   = A.dias_ativos/W1                       # fracao dos dias em que apareceu
A['volume'] = A.n/A.sessoes.clip(lower=1)            # respostas por sessao
A['prof']   = np.log1p(A.expl)                       # segundos lendo explicacao (mediana)
EIXOS=['freq','volume','prof']

# --- desfechos futuros (janela 2), nunca usados como feature ---
nb=b.groupby('user').size()
A['ret']=(A.index.map(nb).fillna(0)>0).astype(int)          # 1. continuou
A['vol_fut']=np.log1p(A.index.map(nb).fillna(0))            # 2. quanto estudou
ac2=b.groupby('user').correct.mean()
A['ganho']=A.index.map(ac2)-A.acerto                        # 3. ganhou acerto
print(f'{len(A):,} alunos com >=30 respostas na janela 1 | retencao {A.ret.mean():.1%}\n')

print('=== os eixos se correlacionam entre si? (Spearman) ===')
print(A[EIXOS].rank().corr().round(2).to_string())

print('\n=== 1. desfecho RETENCAO (continuou nos dias 30-59) — AUC ===')
for c in EIXOS+['n','acerto']:
    print(f'    {c:>8}: {roc_auc_score(A.ret,A[c]):.3f}')

print('\n=== incremento sobre so-volume-bruto (n) — AUC out-of-sample ===')
rng=np.random.RandomState(20260913); msk=rng.rand(len(A))<0.7
tr,te=A[msk],A[~msk]
def auc(cols):
    X=np.log1p(tr[cols].clip(lower=0)); Xe=np.log1p(te[cols].clip(lower=0))
    mu,sd=X.mean(),X.std()
    w=logistic_fit(((X-mu)/sd).values,tr.ret.values)
    return roc_auc_score(te.ret.values,logistic_pred(w,((Xe-mu)/sd).values))
base=auc(['n'])
print(f'    so n                : {base:.3f}')
for c in EIXOS:
    print(f'    n + {c:<16}: {auc(["n",c]):.3f}')
print(f'    n + os tres eixos   : {auc(["n"]+EIXOS):.3f}')
print(f'    os tres eixos sem n : {auc(EIXOS):.3f}')

print('\n=== 2. desfecho VOLUME FUTURO (Spearman) ===')
for c in EIXOS+['n','acerto']:
    print(f'    {c:>8}: {spearman(A[c].values,A.vol_fut.values):+.3f}')

print('\n=== 3. desfecho GANHO DE ACERTO (Spearman, so quem voltou) ===')
S=A[A.ret==1].dropna(subset=['ganho'])
print(f'    ({len(S):,} alunos, ganho medio {S.ganho.mean():+.3f})')
for c in EIXOS+['n','acerto']:
    print(f'    {c:>8}: {spearman(S[c].values,S.ganho.values):+.3f}')

A.to_parquet(f'{HERE}/eixos.parquet')
print(f'\n-> {HERE}/eixos.parquet')
