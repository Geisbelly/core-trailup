"""Condicao real de lancamento: banco vazio.

Sem historico proprio, no dia 1 o TrailUp nao tem:
  - corpus de dificuldade por questao (q_dif, q_n)  -> questao nova de professor
  - historico do aluno                              -> aluno novo
Este script mede o que sobra em cada combinacao, contra a regra atual.
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features.parquet')
u=df.user.unique(); rng.shuffle(u)
n=len(u); tr_u,te_u=u[:int(.8*n)],u[int(.9*n):]
s=pd.Series('test',index=u).to_dict(); s.update(dict.fromkeys(tr_u,'train'))
df['split']=df.user.map(s); tr,te=df[df.split=='train'].copy(),df[df.split=='test'].copy()
g=tr.groupby('qidx').y.agg(['mean','size']); G=tr.y.mean()
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
for d in (tr,te):
    d['q_dif']=d.qidx.map(diff).fillna(G).astype(np.float32)
    d['q_n']=np.log1p(d.qidx.map(g['size']).fillna(0)).astype(np.float32)

COMP=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
      'dt_log','dt_part_log','seen_before','part']
FULL=COMP+['q_dif','q_n','ntags','diagnosis']
def fit(F):
    return HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_depth=6,
        min_samples_leaf=200,l2_regularization=1.0,random_state=0,
        categorical_features=[F.index('part')]).fit(tr[F],tr.y)
m_full, m_comp = fit(FULL), fit(COMP)
te['p_full']=m_full.predict_proba(te[FULL])[:,1]
te['p_comp']=m_comp.predict_proba(te[COMP])[:,1]
y=te.y.values

def ece(yy,pp,bins=10):
    e=0.0; idx=np.digitize(pp,np.linspace(0,1,bins+1)[1:-1])
    for b in range(bins):
        m=idx==b
        if m.sum(): e+=m.mean()*abs(yy[m].mean()-pp[m].mean())
    return e

print('=== o que sobra em cada cenario de lancamento ===')
print('(aluno novo = suas primeiras 5 respostas na plataforma)\n')
novo=(te.n_prev<5).values
cen=[('banco maduro: corpus + historico', ~novo, 'p_full'),
     ('questao com corpus, ALUNO NOVO',     novo, 'p_full'),
     ('SEM corpus, aluno com historico',   ~novo, 'p_comp'),
     ('SEM corpus e ALUNO NOVO (dia 1)',    novo, 'p_comp')]
print(f"{'cenario':36s} {'linhas':>9} {'modelo':>8} {'regra':>8} {'ECE mod':>8}")
for lab,msk,col in cen:
    pm=te[col].values[msk]; yy=y[msk]
    print(f"{lab:36s} {msk.sum():>9,} {roc_auc_score(yy,pm):>8.4f} "
          f"{roc_auc_score(yy,te.regra_trailup.values[msk]):>8.4f} {ece(yy,pm):>8.4f}")

print('\n=== quanto custa NAO ter o corpus de dificuldade ===')
for lo,hi,lab in [(0,4,'1a a 5a resposta'),(5,19,'6a a 20a'),(20,10**9,'21a+')]:
    m=((te.n_prev>=lo)&(te.n_prev<=hi)).values
    a,b=roc_auc_score(y[m],te.p_full.values[m]),roc_auc_score(y[m],te.p_comp.values[m])
    print(f'  {lab:18s} com corpus {a:.4f} | sem corpus {b:.4f} | perda {a-b:+.4f}')

print('\n=== o corpus cresce rapido? AUC por n de observacoes da questao no treino ===')
qn=te.qidx.map(g['size']).fillna(0)
for lo,hi,lab in [(1,20,'1-20'),(21,100,'21-100'),(101,1000,'101-1.000'),(1001,10**9,'1.000+')]:
    m=((qn>=lo)&(qn<=hi)).values
    if m.sum()<1000: continue
    print(f'  questao vista {lab:10s} {m.sum():>9,} linhas | AUC {roc_auc_score(y[m],te.p_full.values[m]):.4f}')

print('\n=== modelo so-comportamento: metricas completas (o que iria ao ar no dia 1) ===')
p=np.clip(te.p_comp.values,1e-6,1-1e-6)
print(f'  AUC {roc_auc_score(y,p):.4f} | log-loss {log_loss(y,p):.4f} | Brier {brier_score_loss(y,p):.4f} | ECE {ece(y,p):.4f}')
r=np.clip(te.regra_trailup.values,1e-6,1-1e-6)
print(f'  regra: AUC {roc_auc_score(y,r):.4f} | log-loss {log_loss(y,r):.4f} | Brier {brier_score_loss(y,r):.4f} | ECE {ece(y,r):.4f}')
import pickle; pickle.dump({'model':m_comp,'feats':COMP},open(f'{HERE}/m2_dia1.pkl','wb'))
print(f"\nmodelo de dia 1 salvo: {os.path.getsize(f'{HERE}/m2_dia1.pkl')/1e6:.1f} MB")
