"""So a tabela, vetorizado. Referencia da forma fechada: AUC 0,740."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts']).sort_values(['user','part','ts']).reset_index(drop=True)
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
c = d.correct.values.astype(np.int64); cs = np.concatenate([[0], np.cumsum(c)])
n = len(c); idx = np.arange(n); fut = np.full(n, np.nan)
ok = idx + 11 <= n
fut[ok] = (cs[idx[ok]+11] - cs[idx[ok]+1]) / 10.0
gc = d.groupby(['user','part']).cumcount().values
sz = d.groupby(['user','part']).correct.transform('size').values
fut[(sz - gc - 1) < 10] = np.nan
d['alvo'] = (fut < 0.5).astype(float); d['k'] = gc
d['ac_top'] = (d.groupby(['user','part']).correct.cumsum() - d['correct']) / np.maximum(gc, 1)
kg = d.groupby('user').cumcount().values
d['ac_glob'] = (d.groupby('user').correct.cumsum() - d['correct']) / np.maximum(kg, 1)
D = d[np.isfinite(fut) & (d.k >= 5)]
TR, TE = D[D.tr], D[~D.tr]
y = TE.alvo.values; base = float(TR.alvo.mean())
print(f'treino {len(TR):,} | teste {len(TE):,} | base {y.mean():.1%}')
print('referencia: forma fechada do modulo AUC 0,740 | boosting 0,779\n')
# forma fechada reproduzida em vetor (mesma aritmetica, sem o componente sequencial)
lin = -(0.50*TE.ac_top.values + 0.35*TE.ac_glob.values + 0.15*TE.q_dif.values)
print(f'  so o acumulado (sem o componente sequencial): AUC {roc_auc_score(y, lin):.3f}')
CORTES = {'ac_top': np.array([.3,.45,.55,.65,.75,.85]),
          'ac_glob': np.array([.45,.55,.62,.68,.75,.82]),
          'q_dif': np.array([.45,.58,.68,.78,.88]),
          'k': np.array([8,15,30,60,120])}
def chave(df):
    return pd.MultiIndex.from_arrays([np.digitize(df[c].values, CORTES[c]) for c in CORTES])
ktr, kte = chave(TR), chave(TE)
tab = pd.DataFrame({'k': ktr, 'y': TR.alvo.values}).groupby('k').y.agg(['sum','size'])
print(f'\n  {len(tab):,} celulas | {(tab["size"]<30).mean():.1%} com menos de 30 casos')
melhor = None
for prior in (20, 50, 100, 200):
    est = (tab['sum'] + base*prior) / (tab['size'] + prior)
    p = kte.map(est).astype(float); p = np.where(np.isfinite(p), p, base)
    a = roc_auc_score(y, p)
    print(f'  tabela, encolhimento {prior:>3}: AUC {a:.3f}')
    if melhor is None or a > melhor[1]: melhor = (prior, a, p)
prior, a, p_tab = melhor
def z(v): v=np.asarray(v,float); return (v-v.mean())/(v.std()+1e-9)
print(f'\n  combinando tabela (encolhimento {prior}) com o acumulado linear:')
for w in (0.3, 0.5, 0.7, 1.0):
    s = w*z(p_tab) + (1-w)*z(lin)
    print(f'    {w:.1f} tabela + {1-w:.1f} linear: AUC {roc_auc_score(y, s):.3f}')
