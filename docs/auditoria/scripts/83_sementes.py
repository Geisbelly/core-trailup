"""Os numeros do modulo vieram de UMA particao (random_state=7). Quanto deles
e propriedade do dado e quanto e da semente?"""
import os, sys, numpy as np, pandas as pd
S = '/caminho/para/os/intermediarios'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')))
sys.path.insert(0, S + '/engaj')
from _shim import roc_auc_score
from trailup_core import dominio as DOM
d = pd.read_parquet(f'{S}/m2/responses_v3.parquet', columns=['user','qidx','part','correct','ts']).sort_values(['user','ts'])
print(f'{len(d):,} respostas | 6 sementes diferentes\n')
print(f'  {"semente":<9} {"dominio 3 termos":>18} {"so a questao":>14} {"base de teste":>15}')
aucs = []
for seed in (7, 11, 23, 42, 101, 2026):
    rng = np.random.RandomState(seed)
    us = d.user.unique().copy(); rng.shuffle(us); tr = set(us[:int(.7*len(us))])
    m_tr = d.user.isin(tr).values
    G = float(d.correct.values[m_tr].mean())
    q = d[m_tr].groupby('qidx').correct.agg(['sum','size'])
    qd = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G).values
    k = d.groupby(['user','part']).cumcount().values
    at = (d.groupby(['user','part']).correct.cumsum() - d.correct).values
    kg = d.groupby('user').cumcount().values
    ag = (d.groupby('user').correct.cumsum() - d.correct).values
    sel = (~m_tr) & (k > 0) & (kg > 0)
    idx = np.where(sel)[0]; idx = rng.choice(idx, min(150000, len(idx)), replace=False)
    p = np.array([DOM.dominio(qd[i], int(at[i]), int(k[i]), media_global=G,
                              acertos_totais=int(ag[i]), respostas_totais=int(kg[i])).p
                  for i in idx])
    y = d.correct.values[idx].astype(float)
    a = roc_auc_score(y, p); aq = roc_auc_score(y, qd[idx])
    aucs.append(a)
    print(f'  {seed:<9} {a:>18.4f} {aq:>14.4f} {y.mean():>15.4f}')
print(f'\n  dominio: media {np.mean(aucs):.4f} | desvio {np.std(aucs):.4f} | amplitude {max(aucs)-min(aucs):.4f}')
print(f'  afirmado no modulo: 0,727 -> {"dentro" if abs(np.mean(aucs)-0.727) < 3*np.std(aucs) else "FORA"} de 3 desvios')
