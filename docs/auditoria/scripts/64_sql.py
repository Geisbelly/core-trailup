"""Auditoria do evasao.sql: os sinais dos coeficientes fazem sentido?

Suspeita: p_semanas_sem_acesso tem coeficiente NEGATIVO (-0,081). Mais semanas
sem acesso deveria AUMENTAR o risco, nao diminuir.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
O = '/caminho/para/os/datasets/open+university+learning+analytics+dataset'
SEM = 7
v = pd.read_csv(f'{O}/studentVle.csv', usecols=['code_module','code_presentation','id_student','date','sum_click'])
v['sem'] = (v.date // SEM).astype(int)
K = ['code_module','code_presentation','id_student','sem']
base = v.groupby(K, as_index=False).sum_click.sum()
reg = pd.read_csv(f'{O}/studentRegistration.csv')
reg['dur'] = pd.to_numeric(reg.date_unregistration, errors='coerce')
info = pd.read_csv(f'{O}/studentInfo.csv')
ass = pd.read_csv(f'{O}/assessments.csv'); sa = pd.read_csv(f'{O}/studentAssessment.csv')
ass['date'] = pd.to_numeric(ass.date, errors='coerce')
sa['score'] = pd.to_numeric(sa.score, errors='coerce')
sa = sa.merge(ass[['id_assessment','code_module','code_presentation','date']], on='id_assessment')
sa['sem'] = sa.date // SEM
sa['atraso'] = sa.date_submitted - sa.date
linhas = []
for sem in range(0, 35):
    b = base[base['sem'] == sem].copy()
    if len(b) < 500: continue
    prev = base[(base['sem'] >= sem-4) & (base['sem'] < sem)].groupby(K[:3], as_index=False).sum_click.sum()
    b = b.merge(prev.rename(columns={'sum_click':'cl4'}), on=K[:3], how='left'); b['cl4'] = b['cl4'].fillna(0)
    # semanas inteiras sem acesso antes desta
    ult = base[base['sem'] < sem].groupby(K[:3], as_index=False)['sem'].max().rename(columns={'sem':'ult'})
    b = b.merge(ult, on=K[:3], how='left')
    b['sem_clique'] = (sem - b['ult'] - 1).fillna(0).clip(lower=0)
    e = sa[sa['sem'] <= sem].groupby(K[:3], as_index=False).agg(
        nota=('score','mean'), n_entregas=('score','size'), atraso=('atraso','mean'))
    b = b.merge(e, on=K[:3], how='left')
    esperadas = ass[(ass.date // SEM) <= sem].groupby(['code_module','code_presentation'], as_index=False).size()
    b = b.merge(esperadas.rename(columns={'size':'esp'}), on=['code_module','code_presentation'], how='left')
    b['n_entregas'] = b['n_entregas'].fillna(0); b['esp'] = b['esp'].fillna(0)
    b['perdidas'] = (b['esp'] - b['n_entregas']).clip(lower=0)
    b['taxa_entrega'] = np.where(b['esp'] > 0, b['n_entregas']/b['esp'].clip(lower=1), 1.0)
    b = b.merge(reg[['code_module','code_presentation','id_student','dur']], on=K[:3], how='left')
    b = b.merge(info[['code_module','code_presentation','id_student','studied_credits']], on=K[:3], how='left')
    b['alvo'] = ((b['dur'] >= sem*SEM) & (b['dur'] < (sem+1)*SEM + 28)).astype(int)
    b = b[(b['dur'].isna()) | (b['dur'] >= sem*SEM)]
    b['semana'] = sem
    linhas.append(b)
P = pd.concat(linhas, ignore_index=True)
P['nota'] = P['nota'].fillna(-1); P['atraso'] = P['atraso'].fillna(0)
P['cl1l'] = np.log1p(P.sum_click); P['cl4l'] = np.log1p(P['cl4'])
P['tend'] = P.sum_click/(P['cl4']/4.0 + 1)
F = ['cl1l','cl4l','tend','sem_clique','semana','studied_credits','nota','n_entregas','perdidas','taxa_entrega','atraso']
P[F] = P[F].fillna(0)
tr = P[P.code_presentation != '2014J']; te = P[P.code_presentation == '2014J']
print(f'painel {len(P):,} | positivos {P.alvo.mean():.2%} | teste {len(te):,}\n')
m = LogisticRegression(max_iter=4000, class_weight='balanced').fit(tr[F], tr.alvo)
p = m.predict_proba(te[F])[:,1]
print(f'AUC (logistica, split por coorte): {roc_auc_score(te.alvo, p):.3f}\n')
SQL = {'cl1l':-0.332613,'cl4l':-0.040751,'tend':0.076028,'sem_clique':-0.081478,
       'semana':-0.032992,'studied_credits':0.007351,'nota':-0.007206,
       'n_entregas':0.011575,'perdidas':0.012417,'taxa_entrega':-1.023119,'atraso':0.007044}
print(f'  {"termo":<18} {"no SQL":>10} {"refeito":>10} {"mesmo sinal?":>13}')
ruins = []
for f_, c in zip(F, m.coef_[0]):
    s = SQL[f_]
    ok = (s >= 0) == (c >= 0)
    if not ok: ruins.append(f_)
    print(f'  {f_:<18} {s:>+10.4f} {c:>+10.4f} {"sim" if ok else "NAO":>13}')
print(f'\n  correlacao de sem_clique com o alvo: {P.sem_clique.corr(P.alvo):+.3f}')
print(f'  taxa de evasao por semanas sem acesso:')
for k in range(0, 5):
    s = P.alvo[P.sem_clique == k]
    if len(s) > 200: print(f'    {k} semanas: {s.mean():.2%}  (n={len(s):,})')
print(f'\n  sinais divergentes: {ruins if ruins else "nenhum"}')

print('\n=== coeficientes para substituir no SQL (refeitos, sem padronizar) ===')
print(f'  intercepto = {m.intercept_[0]:+.6f}')
for f_, c in zip(F, m.coef_[0]):
    print(f'  {f_:<18} {c:+.6f}')
print('\n=== e o desempenho, no mesmo split por coorte ===')
for taxa in (0.05, 0.10, 0.20):
    k = p >= np.quantile(p, 1-taxa)
    print(f'  top {taxa:.0%}: precisao {te.alvo[k].mean():.1%} | lift {te.alvo[k].mean()/te.alvo.mean():.1f}x | cobertura {te.alvo[k].sum()/te.alvo.sum():.0%}')
print(f'  base de positivos no teste: {te.alvo.mean():.2%}')
print('\n=== equidade ===')
te2 = te.merge(info[['code_module','code_presentation','id_student','disability']], on=K[:3], how='left')
pp = pd.Series(p, index=te.index)
for d_ in ('N','Y'):
    mm = (te2.disability == d_).values
    if mm.sum() < 500: continue
    kk = pp.values >= np.quantile(p, 0.9)
    print(f'  disability={d_}: evasao {te.alvo[mm].mean():.1%} | AUC {roc_auc_score(te.alvo[mm], p[mm]):.3f} | cobertura@10% {te.alvo[mm & kk].sum()/max(te.alvo[mm].sum(),1):.0%}')

print('\n=== sem class_weight: a probabilidade fica calibrada? ===')
m2 = LogisticRegression(max_iter=4000).fit(tr[F], tr.alvo)
p2 = m2.predict_proba(te[F])[:,1]
def ece(y,pp,b=10):
    e=0.0; i=np.digitize(pp,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        mm=i==k
        if mm.sum(): e+=mm.mean()*abs(y[mm].mean()-pp[mm].mean())
    return e
print(f'  {"modelo":<26} {"AUC":>7} {"ECE":>7} {"media prevista":>15} {"real":>7}')
print(f'  {"balanced":<26} {roc_auc_score(te.alvo,p):>7.3f} {ece(te.alvo.values,p):>7.3f} {p.mean():>15.3f} {te.alvo.mean():>7.3f}')
print(f'  {"sem class_weight":<26} {roc_auc_score(te.alvo,p2):>7.3f} {ece(te.alvo.values,p2):>7.3f} {p2.mean():>15.3f} {te.alvo.mean():>7.3f}')
print(f'\n  coeficientes SEM class_weight (os que devem ir para o SQL):')
print(f'  intercepto = {m2.intercept_[0]:+.6f}')
for f_, c in zip(F, m2.coef_[0]): print(f'  {f_:<18} {c:+.6f}')
for taxa in (0.05, 0.10):
    k = p2 >= np.quantile(p2, 1-taxa)
    print(f'  top {taxa:.0%}: precisao {te.alvo[k].mean():.1%} | lift {te.alvo[k].mean()/te.alvo.mean():.1f}x')

print('\n=== o que a funcao PUBLICADA devolve? (aritmetica do SQL atual) ===')
def sql_atual(r):
    z = (1.560766
         - 0.332613*np.log1p(max(r['cl1l_raw'],0))
         - 0.040751*np.log1p(max(r['cl4_raw'],0))
         + 0.076028*(r['cl1l_raw']/(r['cl4_raw']/4.0+1))
         - 0.081478*r['sem_clique']
         - 0.032992*r['semana']
         + 0.007351*76.68
         - 0.007206*r['nota']
         + 0.011575*r['n_entregas']
         + 0.012417*r['perdidas']
         - 1.023119*r['taxa_entrega']
         + 0.007044*r['atraso'])
    return 1/(1+np.exp(-z))
amostra = te.sample(20000, random_state=5).copy()
amostra['cl1l_raw'] = amostra.sum_click; amostra['cl4_raw'] = amostra['cl4']
pa = amostra.apply(sql_atual, axis=1).values
ya = amostra.alvo.values
print(f'  media devolvida pelo SQL publicado : {pa.mean():.3f}')
print(f'  taxa real de evasao                : {ya.mean():.3f}')
print(f'  fator de superestimacao            : {pa.mean()/ya.mean():.1f}x')
print(f'  ECE                                : {ece(ya, pa):.3f}')
print(f'  AUC (ordenacao ainda serve?)       : {roc_auc_score(ya, pa):.3f}')
print(f'\n  efeito isolado de sem_clique no SQL publicado:')
for k in (0, 2, 4):
    r = dict(cl1l_raw=20, cl4_raw=80, sem_clique=k, semana=10, nota=60,
             n_entregas=3, perdidas=1, taxa_entrega=0.75, atraso=0)
    print(f'    {k} semanas sem acesso -> risco {sql_atual(r):.3f}')
print('    (deveria SUBIR; no dado bruto a evasao vai de 2,11% a 4,22%)')
