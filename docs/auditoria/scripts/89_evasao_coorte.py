"""A evasao foi testada numa coorte so (2014J). O AUC 0,783 e propriedade do
modelo ou daquela coorte? Testa em CADA coorte, treinando nas outras."""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
O = '/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
SEM = 7
v = pd.read_csv(f'{O}/studentVle.csv', usecols=['code_module','code_presentation','id_student','date','sum_click'])
v['sem'] = (v.date // SEM).astype(int)
K = ['code_module','code_presentation','id_student','sem']
base = v.groupby(K, as_index=False).sum_click.sum()
reg = pd.read_csv(f'{O}/studentRegistration.csv'); reg['dur'] = pd.to_numeric(reg.date_unregistration, errors='coerce')
info = pd.read_csv(f'{O}/studentInfo.csv')
ass = pd.read_csv(f'{O}/assessments.csv'); sa = pd.read_csv(f'{O}/studentAssessment.csv')
ass['date'] = pd.to_numeric(ass.date, errors='coerce'); sa['score'] = pd.to_numeric(sa.score, errors='coerce')
sa = sa.merge(ass[['id_assessment','code_module','code_presentation','date']], on='id_assessment')
sa['sem'] = sa.date // SEM; sa['atraso'] = sa.date_submitted - sa.date
linhas = []
for sem in range(0, 35):
    b = base[base['sem'] == sem].copy()
    if len(b) < 500: continue
    prev = base[(base['sem'] >= sem-4) & (base['sem'] < sem)].groupby(K[:3], as_index=False).sum_click.sum()
    b = b.merge(prev.rename(columns={'sum_click':'cl4'}), on=K[:3], how='left'); b['cl4'] = b['cl4'].fillna(0)
    ult = base[base['sem'] < sem].groupby(K[:3], as_index=False)['sem'].max().rename(columns={'sem':'ult'})
    b = b.merge(ult, on=K[:3], how='left'); b['sem_clique'] = (sem - b['ult'] - 1).fillna(0).clip(lower=0)
    e = sa[sa['sem'] <= sem].groupby(K[:3], as_index=False).agg(nota=('score','mean'), n_entregas=('score','size'), atraso=('atraso','mean'))
    b = b.merge(e, on=K[:3], how='left')
    esp = ass[(ass.date // SEM) <= sem].groupby(['code_module','code_presentation'], as_index=False).size()
    b = b.merge(esp.rename(columns={'size':'esp'}), on=['code_module','code_presentation'], how='left')
    b['n_entregas'] = b['n_entregas'].fillna(0); b['esp'] = b['esp'].fillna(0)
    b['perdidas'] = (b['esp'] - b['n_entregas']).clip(lower=0)
    b['taxa_entrega'] = np.where(b['esp'] > 0, b['n_entregas']/b['esp'].clip(lower=1), 1.0)
    b = b.merge(reg[['code_module','code_presentation','id_student','dur','date_registration']], on=K[:3], how='left')
    b = b.merge(info[['code_module','code_presentation','id_student','studied_credits']], on=K[:3], how='left')
    b['alvo'] = ((b['dur'] >= sem*SEM) & (b['dur'] < (sem+1)*SEM + 28)).astype(int)
    b = b[(b['dur'].isna()) | (b['dur'] >= sem*SEM)]; b['semana'] = sem
    linhas.append(b)
P = pd.concat(linhas, ignore_index=True)
P['nota'] = P['nota'].fillna(-1); P['atraso'] = P['atraso'].fillna(0)
P['reg'] = pd.to_numeric(P.date_registration, errors='coerce').fillna(0)
P['cl1l'] = np.log1p(P.sum_click); P['cl4l'] = np.log1p(P['cl4'])
P['tend'] = P.sum_click/(P['cl4']/4.0 + 1)
F = ['cl1l','cl4l','tend','sem_clique','semana','studied_credits','nota','n_entregas','perdidas','taxa_entrega','atraso']
P[F] = P[F].fillna(0)
print(f'painel {len(P):,}\n')
print(f'  {"coorte de teste":<10} {"linhas":>10} {"evasao":>8} {"AUC":>7} {"lift@10%":>9}')
aucs = []
for co in sorted(P.code_presentation.unique()):
    tr, te = P[P.code_presentation != co], P[P.code_presentation == co]
    if len(te) < 5000 or te.alvo.sum() < 100: continue
    m = LogisticRegression(max_iter=4000).fit(tr[F], tr.alvo)
    p = m.predict_proba(te[F])[:, 1]
    a = roc_auc_score(te.alvo, p); k = p >= np.quantile(p, .9)
    aucs.append(a)
    print(f'  {co:<10} {len(te):>10,} {te.alvo.mean():>8.2%} {a:>7.3f} {te.alvo[k].mean()/te.alvo.mean():>8.1f}x')
a = np.array(aucs)
print(f'\n  media {a.mean():.4f} | desvio {a.std(ddof=1):.4f} | amplitude {a.max()-a.min():.4f}')
print(f'  afirmado 0,783 (coorte 2014J) -> {abs(a.mean()-0.783)/a.std(ddof=1):.1f} desvios da media')
