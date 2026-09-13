"""(5) Evasao, agora com as features de AVALIACAO que faltavam.

Nota, entregas feitas x esperadas, atraso. Tudo com corte temporal em w.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
HERE=os.path.dirname(os.path.abspath(__file__))
si=pd.read_csv(f'{O}/studentInfo.csv'); sr=pd.read_csv(f'{O}/studentRegistration.csv')
sa=pd.read_csv(f'{O}/studentAssessment.csv'); asm=pd.read_csv(f'{O}/assessments.csv')
asm['date']=pd.to_numeric(asm.date,errors='coerce')
sa['score']=pd.to_numeric(sa.score,errors='coerce')
S=sa.merge(asm[['id_assessment','code_module','code_presentation','date','weight','assessment_type']],on='id_assessment')
S=S.dropna(subset=['date'])
S['atraso']=S.date_submitted-S.date
print(f'{len(S):,} entregas com data')
V=[]
for ch in pd.read_csv(f'{O}/studentVle.csv',chunksize=2_000_000):
    ch['sem']=(ch.date//7).clip(-4,39)
    V.append(ch.groupby(['code_module','code_presentation','id_student','sem'],as_index=False).sum_click.sum())
V=pd.concat(V).groupby(['code_module','code_presentation','id_student','sem'],as_index=False).sum_click.sum()
Vs=V.set_index(['code_module','code_presentation','id_student','sem']).sum_click
sr['un']=pd.to_numeric(sr.date_unregistration,errors='coerce')
base=si.merge(sr[['code_module','code_presentation','id_student','un','date_registration']],
              on=['code_module','code_presentation','id_student'],how='left')
# esperado: quantas avaliacoes ja venceram ate a semana w
esp=asm.groupby(['code_module','code_presentation']).date.apply(lambda d:np.sort(d.dropna().values))
Sg={k:v for k,v in S.groupby(['code_module','code_presentation','id_student'])}
linhas=[]
for w in range(1,21):
    b=base[(base.un.isna())|(base.un>w*7)].copy()
    b['alvo']=((b.un>w*7)&(b.un<=(w+4)*7)).fillna(False).astype(int)
    b['alvo_final']=(b.final_result=='Withdrawn').astype(int)
    idx=list(zip(b.code_module,b.code_presentation,b.id_student))
    for lag,nm in [(1,'cl1'),(4,'cl4')]:
        b[nm]=[sum(Vs.get((m,p,s,w-k),0) for k in range(lag)) for m,p,s in idx]
    b['sem_clique']=[next((k for k in range(0,12) if Vs.get((m,p,s,w-k),0)>0),12) for m,p,s in idx]
    nota=[];nent=[];atr=[];nesp=[]
    for m,p,s in idx:
        g=Sg.get((m,p,s))
        e=esp.get((m,p),np.array([])); ne=int((e<=w*7).sum()); nesp.append(ne)
        if g is None: nota.append(np.nan);nent.append(0);atr.append(np.nan); continue
        gg=g[g.date<=w*7]
        nota.append(gg.score.mean() if len(gg) else np.nan)
        nent.append(len(gg)); atr.append(gg.atraso.mean() if len(gg) else np.nan)
    b['nota']=nota; b['n_entregas']=nent; b['atraso']=atr; b['n_esperadas']=nesp
    b['perdidas']=(b.n_esperadas-b.n_entregas).clip(lower=0)
    b['taxa_entrega']=b.n_entregas/b.n_esperadas.replace(0,np.nan)
    b['semana']=w
    linhas.append(b)
P=pd.concat(linhas,ignore_index=True)
P['cl1l']=np.log1p(P.cl1); P['cl4l']=np.log1p(P.cl4); P['tend']=P.cl1/(P.cl4/4+1)
P['reg']=pd.to_numeric(P.date_registration,errors='coerce').fillna(0)
P['nota']=P.nota.fillna(-1); P['atraso']=P.atraso.fillna(0); P['taxa_entrega']=P.taxa_entrega.fillna(1)
print(f'painel {len(P):,} | positivos {P.alvo.mean():.2%}')
F=['cl1l','cl4l','tend','sem_clique','semana','num_of_prev_attempts','studied_credits','reg',
   'nota','n_entregas','perdidas','taxa_entrega','atraso']
tr=P[P.code_presentation!='2014J']; te=P[P.code_presentation=='2014J']
def av(n,p):
    p=np.clip(p,1e-6,1-1e-6)
    return {'modelo':n,'AUC':roc_auc_score(te.alvo,p),'AP':average_precision_score(te.alvo,p)}
rows=[av('regra: sem clique ha >=14 dias',(te.sem_clique>=2).astype(float)*.9+.05),
      av('so nota media ate agora',-te.nota.values)]
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced')).fit(tr[F],tr.alvo)
rows.append(av('logistica completa (SQL)',lr.predict_proba(te[F])[:,1]))
gb=HistGradientBoostingClassifier(max_iter=400,learning_rate=.05,max_depth=6,
    min_samples_leaf=80,random_state=0).fit(tr[F],tr.alvo)
p=gb.predict_proba(te[F])[:,1]; rows.append(av('boosting completo',p))
print('\n=== com avaliacoes (teste: coorte 2014J) ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print(f'\nbase de positivos: {te.alvo.mean():.2%}  (AP do acaso)')
print('\n=== por semana ===')
for w in (2,4,8,12,16):
    m=(te['semana']==w).values
    if m.sum()<200: continue
    print(f'  semana {w:>2}: positivos {te.alvo[m].mean():>5.1%} | AUC {roc_auc_score(te.alvo[m],p[m]):.3f} | '
          f'AP {average_precision_score(te.alvo[m],p[m]):.3f}')
print('\n=== alertar os 10% de maior risco ===')
lim=np.quantile(p,.9); k=p>=lim
print(f'  precisao {te.alvo[k].mean():.1%} (base {te.alvo.mean():.1%}, lift {te.alvo[k].mean()/te.alvo.mean():.1f}x) | '
      f'cobertura {te.alvo[k].sum()/te.alvo.sum():.0%}')
print('\n=== equidade ===')
for v in ('N','Y'):
    m=(te.disability==v).values
    print(f'  disability={v}: evasao {te.alvo[m].mean():>5.1%} | AUC {roc_auc_score(te.alvo[m],p[m]):.3f} | '
          f'cobertura@top10% {((p>=lim)&(te.alvo==1)).values[m].sum()/max(te.alvo[m].sum(),1):.0%}')

print('\n=== e se o alvo for "evade EM ALGUM MOMENTO" (como a literatura)? ===')
gb2=HistGradientBoostingClassifier(max_iter=400,learning_rate=.05,max_depth=6,
    min_samples_leaf=80,random_state=0).fit(tr[F],tr.alvo_final)
p2=gb2.predict_proba(te[F])[:,1]
print(f'  positivos {te.alvo_final.mean():.1%} | AUC {roc_auc_score(te.alvo_final,p2):.3f} | '
      f'AP {average_precision_score(te.alvo_final,p2):.3f}')
for w in (2,4,8,12):
    m=(te['semana']==w).values
    if m.sum()<200: continue
    print(f'    semana {w:>2}: AUC {roc_auc_score(te.alvo_final[m],p2[m]):.3f}')
print('\n  -> prever SE evade e mais facil que prever QUANDO; a diferenca e o custo')
print('     de querer o alerta na janela certa em vez de um rotulo de fim de curso.')
