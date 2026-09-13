"""Como representar a dificuldade de uma questao? 5 variantes, no OULAD.

Split por APRESENTACAO (turma): treina em turmas antigas, testa numa turma nova.
E o cenario real - turma nova comeca sem historico proprio.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
a=pd.read_csv(f'{O}/assessments.csv'); sa=pd.read_csv(f'{O}/studentAssessment.csv')
sa['score']=pd.to_numeric(sa.score,errors='coerce'); sa=sa.dropna(subset=['score'])
a=a.sort_values(['code_module','code_presentation','date'])
a['ordem']=a.groupby(['code_module','code_presentation']).cumcount()+1
a['slot']=a.code_module+'|'+a.assessment_type+'|'+a.ordem.astype(str)
d=sa.merge(a[['id_assessment','code_module','code_presentation','assessment_type','slot','date','ordem']],on='id_assessment')
d['turma']=d.code_module+'-'+d.code_presentation
print('distribuicao da nota:',{f'>={t}':round((d.score>=t).mean(),3) for t in (40,60,70,80,90)})
ALVO=80
d['y']=(d.score>=ALVO).astype(int)
print(f'alvo: nota >= {ALVO} -> {d.y.mean():.1%} positivos | {len(d):,} notas')

TESTE=['2014J']
tr=d[~d.code_presentation.isin(TESTE)].copy(); te=d[d.code_presentation.isin(TESTE)].copy()
print(f'treino {len(tr):,} ({tr.turma.nunique()} turmas) | teste {len(te):,} ({te.turma.nunique()} turmas)')
G=tr.y.mean()

def suaviza(soma,n,prior,base): return (soma+base*prior)/(n+prior)

# D0 global por slot
g0=tr.groupby('slot').y.agg(['sum','size'])
D0=suaviza(g0['sum'],g0['size'],20,G)
# D1 por turma (cru) - cai para o global quando a turma nao tem a avaliacao
g1=tr.groupby(['slot','turma']).y.agg(['sum','size'])
D1=suaviza(g1['sum'],g1['size'],5,G)
# D3 hierarquico: global + encolhimento da turma
g1r=g1.reset_index(); g1r['glob']=g1r.slot.map(D0)
g1r['hier']=(g1r['sum']+g1r['glob']*15)/(g1r['size']+15)
D3=g1r.set_index(['slot','turma'])['hier']
# D2 percentil dentro da turma: posicao relativa da avaliacao entre as da turma
mt=tr.groupby(['turma','slot']).y.mean().reset_index()
mt['pct']=mt.groupby('turma').y.rank(pct=True)
D2=mt.set_index(['slot','turma'])['pct']
# habilidade do aluno: media das notas ANTERIORES dele na turma (causal)
for X in (tr,te):
    X.sort_values(['id_student','date'],inplace=True)
    cs=X.groupby('id_student').y.cumsum()-X.y; cn=X.groupby('id_student').cumcount()
    X['hab']=((cs+G*3)/(cn+3)).values; X['n_ant']=cn.values

def aplica(X):
    k=list(zip(X.slot,X.turma))
    X['d0']=X.slot.map(D0).fillna(G)
    X['d1']=pd.Series(k,index=X.index).map(D1).fillna(X.d0)
    X['d3']=pd.Series(k,index=X.index).map(D3).fillna(X.d0)
    X['d2']=pd.Series(k,index=X.index).map(D2).fillna(0.5)
    return X
tr=aplica(tr); te=aplica(te)

def av(nome,p):
    p=np.clip(p,1e-6,1-1e-6)
    return {'variante':nome,'AUC':roc_auc_score(te.y,p),'logloss':log_loss(te.y,p),'brier':brier_score_loss(te.y,p)}
rows=[av('D0 dificuldade GLOBAL da questao',te.d0),
      av('D1 dificuldade POR TURMA (cru)',te.d1),
      av('D2 PERCENTIL dentro da turma',te.d2),
      av('D3 hierarquico (global + turma)',te.d3),
      av('habilidade do aluno (so)',te.hab)]
print('\n=== so a dificuldade, sem modelo (turma nova: 2014J) ===')
print(pd.DataFrame(rows).set_index('variante').round(4).to_string())

from sklearn.ensemble import HistGradientBoostingClassifier
print('\n=== com modelo: dificuldade + habilidade do aluno ===')
rows=[]
for F,nm in [(['d0','hab','n_ant'],'D0 global + aluno'),
             (['d1','hab','n_ant'],'D1 por turma + aluno'),
             (['d2','hab','n_ant'],'D2 percentil + aluno'),
             (['d3','hab','n_ant'],'D3 hierarquico + aluno'),
             (['d0','d2','hab','n_ant'],'D0 + D2 percentil + aluno'),
             (['d0','d3','d2','hab','n_ant'],'todas + aluno')]:
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_depth=4,
        min_samples_leaf=100,random_state=0).fit(tr[F],tr.y)
    rows.append(av(nm,m.predict_proba(te[F])[:,1]))
print(pd.DataFrame(rows).set_index('variante').round(4).to_string())
