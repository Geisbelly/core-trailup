"""(5) Risco de evasao semanal - OULAD, rotulo real.

Painel aluno x semana. Features so com dado ATE a semana w (sem vazamento).
Split por COORTE: treina em apresentacoes antigas, testa em 2014J - o cenario
real de turma nova.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
HERE=os.path.dirname(os.path.abspath(__file__))
si=pd.read_csv(f'{O}/studentInfo.csv'); sr=pd.read_csv(f'{O}/studentRegistration.csv')
sa=pd.read_csv(f'{O}/studentAssessment.csv'); a=pd.read_csv(f'{O}/assessments.csv')
print('lendo studentVle em blocos...')
vle=[]
for ch in pd.read_csv(f'{O}/studentVle.csv',chunksize=2_000_000):
    ch['sem']=(ch.date//7).clip(-4,39)
    vle.append(ch.groupby(['code_module','code_presentation','id_student','sem'],as_index=False).sum_click.sum())
V=pd.concat(vle).groupby(['code_module','code_presentation','id_student','sem'],as_index=False).sum_click.sum()
print(f'  {len(V):,} celulas aluno-semana com clique')
si['turma']=si.code_module+'-'+si.code_presentation
sr['un']=pd.to_numeric(sr.date_unregistration,errors='coerce')
base=si.merge(sr[['code_module','code_presentation','id_student','un','date_registration']],
              on=['code_module','code_presentation','id_student'],how='left')
print(f'{len(base):,} matriculas | evadiram {base.un.notna().mean():.1%}')
# painel semanal
SEMS=range(1,21)
linhas=[]
V=V.set_index(['code_module','code_presentation','id_student','sem']).sum_click
for w in SEMS:
    b=base[(base.un.isna())|(base.un>w*7)].copy()          # ainda matriculado na semana w
    b['alvo']=((b.un>w*7)&(b.un<=(w+4)*7)).fillna(False).astype(int)
    idx=list(zip(b.code_module,b.code_presentation,b.id_student))
    for lag,nome in [(1,'cl1'),(2,'cl2'),(4,'cl4')]:
        b[nome]=[sum(V.get((m,p,s,w-k),0) for k in range(lag)) for m,p,s in idx]
    b['dias_sem_clique']=[next((k for k in range(0,12) if V.get((m,p,s,w-k),0)>0),12) for m,p,s in idx]
    b['semana']=w
    linhas.append(b)
P=pd.concat(linhas,ignore_index=True)
P['tend']=P.cl1/(P.cl4/4+1)
P['cl1l']=np.log1p(P.cl1); P['cl4l']=np.log1p(P.cl4)
P['prev']=P.num_of_prev_attempts; P['cred']=P.studied_credits
P['reg']=pd.to_numeric(P.date_registration,errors='coerce').fillna(0)
print(f'\npainel: {len(P):,} linhas aluno-semana | positivos {P.alvo.mean():.2%}')
F=['cl1l','cl4l','tend','dias_sem_clique','semana','prev','cred','reg']
tr=P[P.code_presentation!='2014J']; te=P[P.code_presentation=='2014J']
print(f'treino {len(tr):,} ({tr.code_presentation.nunique()} coortes) | teste 2014J {len(te):,}')
def av(n,p):
    p=np.clip(p,1e-6,1-1e-6)
    return {'modelo':n,'AUC':roc_auc_score(te.alvo,p),'AP':average_precision_score(te.alvo,p),
            'Brier':brier_score_loss(te.alvo,p)}
rows=[av('regra: sem clique ha >=14 dias',(te.dias_sem_clique>=2).astype(float)*.9+.05)]
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
lr.fit(tr[F],tr.alvo); rows.append(av('logistica (exportavel p/ SQL)',lr.predict_proba(te[F])[:,1]))
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_depth=5,
    min_samples_leaf=100,random_state=0).fit(tr[F],tr.alvo)
p_gb=gb.predict_proba(te[F])[:,1]; rows.append(av('boosting',p_gb))
print('\n=== prever evasao nas proximas 4 semanas (teste: coorte 2014J) ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print('\n=== por semana do curso ===')
for w in (2,4,8,12,16):
    m=(te['semana']==w).values
    if m.sum()<200: continue
    print(f'  semana {w:>2}: {m.sum():>6,} alunos | positivos {te.alvo[m].mean():>5.1%} | '
          f'AUC {roc_auc_score(te.alvo[m],p_gb[m]):.3f}')
print('\n=== equidade: o modelo erra mais em quem tem deficiencia declarada? ===')
for v in ('N','Y'):
    m=(te.disability==v).values
    lim=np.quantile(p_gb,.9)
    fn=((p_gb<lim)&(te.alvo==1)).values[m].sum()/max(te.alvo[m].sum(),1)
    print(f'  disability={v}: {m.sum():>6,} | evasao {te.alvo[m].mean():>5.1%} | '
          f'AUC {roc_auc_score(te.alvo[m],p_gb[m]):.3f} | falso-negativo@top10% {fn:.1%}')
import pickle
pickle.dump({'logistica':lr,'feats':F},open(f'{HERE}/evasao.pkl','wb'))
co=dict(zip(F,lr[-1].coef_[0]))
print('\n=== coeficientes (exportaveis para funcao SQL) ===')
for k,v in sorted(co.items(),key=lambda x:-abs(x[1])): print(f'  {k:>18}: {v:+.4f}')
print(f'  {"intercepto":>18}: {lr[-1].intercept_[0]:+.4f}')
