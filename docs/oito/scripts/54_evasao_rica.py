"""MELHORIA: evasao. Minha hipotese para o AUC 0,748 era "conjunto de features
magro - cliques agregados, sem quebra por tipo de recurso". Nunca testei.

Acrescenta: cliques por TIPO de atividade (conteudo, forum, quiz, navegacao) e
diversidade de recursos tocados na semana.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
O = '/caminho/para/os/datasets/open+university+learning+analytics+dataset'
SEM = 7
GRUPO = {'oucontent':'conteudo','resource':'conteudo','url':'conteudo','dualpane':'conteudo',
         'htmlactivity':'conteudo','forumng':'social','oucollaborate':'social','ouwiki':'social',
         'quiz':'prova','externalquiz':'prova','questionnaire':'prova',
         'subpage':'navega','homepage':'navega','page':'navega','folder':'navega',
         'glossary':'apoio','dataplus':'apoio','repeatactivity':'apoio',
         'sharedsubpage':'navega','ouelluminate':'social'}
vle = pd.read_csv(f'{O}/vle.csv', usecols=['id_site','code_module','code_presentation','activity_type'])
vle['g'] = vle.activity_type.map(GRUPO).fillna('apoio')
v = pd.read_csv(f'{O}/studentVle.csv')
v = v.merge(vle[['id_site','code_module','code_presentation','g']],
            on=['id_site','code_module','code_presentation'], how='left')
v['g'] = v.g.fillna('apoio'); v['sem'] = (v.date // SEM).astype(int)
K = ['code_module','code_presentation','id_student','sem']
print(f'{len(v):,} cliques marcados por tipo', flush=True)
agg = v.groupby(K, as_index=False).agg(cl=('sum_click','sum'), n_rec=('id_site','nunique'))
piv = v.pivot_table(index=K, columns='g', values='sum_click', aggfunc='sum').fillna(0)
piv.columns.name = None
piv = piv.reset_index()
base = agg.merge(piv, on=K, how='left').fillna(0)
base.columns.name = None
assert 'sem' in base.columns, f'coluna sem sumiu: {list(base.columns)}'
print('colunas de base:', list(base.columns), flush=True)
TIPOS = [c for c in ['conteudo','social','prova','navega','apoio'] if c in base.columns]
for t in TIPOS: base['f_'+t] = base[t] / base.cl.clip(lower=1)

reg = pd.read_csv(f'{O}/studentRegistration.csv')
info = pd.read_csv(f'{O}/studentInfo.csv')
ass = pd.read_csv(f'{O}/assessments.csv'); sa = pd.read_csv(f'{O}/studentAssessment.csv')
ass['date'] = pd.to_numeric(ass.date, errors='coerce')
sa['score'] = pd.to_numeric(sa['score'], errors='coerce')
sa = sa.merge(ass[['id_assessment','code_module','code_presentation','date']], on='id_assessment')
sa['sem'] = (sa.date // SEM)
reg['dur'] = pd.to_numeric(reg.date_unregistration, errors='coerce')
linhas = []
for sem in range(0, 35):
    b = base[base['sem'] == sem].copy()
    if len(b) < 500: continue
    prev = base[(base['sem'] >= sem-3) & (base['sem'] < sem)].groupby(K[:3], as_index=False).cl.sum()
    b = b.merge(prev.rename(columns={'cl':'cl4'}), on=K[:3], how='left')
    b['cl4'] = b['cl4'].fillna(0)
    nt = sa[sa['sem'] <= sem].groupby(['code_module','code_presentation','id_student'], as_index=False).score.mean()
    b = b.merge(nt.rename(columns={'score':'nota'}), on=K[:3], how='left')
    b = b.merge(reg[['code_module','code_presentation','id_student','date_registration','dur']], on=K[:3], how='left')
    b = b.merge(info[['code_module','code_presentation','id_student','num_of_prev_attempts','studied_credits']], on=K[:3], how='left')
    b['alvo'] = ((b['dur'] >= sem*SEM) & (b['dur'] < (sem+1)*SEM + 28)).astype(int)
    b = b[(b['dur'].isna()) | (b['dur'] >= sem*SEM)]
    b['semana'] = sem
    linhas.append(b)
P = pd.concat(linhas, ignore_index=True)
P['cl1l'] = np.log1p(P['cl']); P['cl4l'] = np.log1p(P['cl4']); P['tend'] = P['cl']/(P['cl4']/3+1)
P['reg'] = pd.to_numeric(P.date_registration, errors='coerce').fillna(0)
P['nota'] = P['nota'].fillna(-1); P['div'] = P['n_rec']
print(f'painel {len(P):,} | positivos {P["alvo"].mean():.2%}\n')
MAGRO = ['cl1l','cl4l','tend','semana','num_of_prev_attempts','studied_credits','reg','nota']
RICO = MAGRO + ['div'] + ['f_'+t for t in TIPOS]
tr = P[P.code_presentation != '2014J']; te = P[P.code_presentation == '2014J']
print(f'{"conjunto":<34} {"AUC":>7} {"AP":>7} {"lift@10%":>9}')
for lab, F in [('magro (cliques agregados)', MAGRO), ('rico (+ tipo de recurso)', RICO)]:
    for mdl, nm in [(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')), 'logistica'),
                    (HistGradientBoostingClassifier(max_iter=400, learning_rate=.05, max_depth=6, min_samples_leaf=80, random_state=0), 'boosting')]:
        m = mdl.fit(tr[F].fillna(0), tr.alvo)
        p = m.predict_proba(te[F].fillna(0))[:,1]
        k = p >= np.quantile(p, .9)
        print(f'{lab+" / "+nm:<34} {roc_auc_score(te.alvo,p):>7.3f} {average_precision_score(te.alvo,p):>7.3f} {te.alvo[k].mean()/te.alvo.mean():>8.1f}x')
print(f'\nbase de positivos no teste: {te.alvo.mean():.2%}')
