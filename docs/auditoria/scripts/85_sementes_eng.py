"""A checagem de sementes so foi feita no dominio. O engajamento afirma
0,856 / 0,862 e o gate 0,732 - vieram de UMA particao cada.

O engajamento e o mais exposto: os PESOS do ordenador foram ajustados numa
particao e avaliados... na mesma base. Se a particao mudar, os pesos mudam?
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E = pd.read_parquet(f'{HERE}/eixos.parquet'); O = pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
COLS = ['rec', 'freq']
for D in (E, O):
    for c in COLS: D[c] = D[c].replace([np.inf, -np.inf], np.nan)
E = E.dropna(subset=COLS + ['ret']); O = O.dropna(subset=COLS + ['ret'])
def z(D):
    X = D[COLS].values.astype(float)
    return (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
ZE, ZO = z(E), z(O)
print(f'EdNet {len(E):,} | OULAD {len(O):,} | 12 particoes\n')
print(f'  {"semente":<9} {"EdNet":>9} {"OULAD":>9} {"peso recencia":>15} {"peso freq":>11}')
res = {'E': [], 'O': [], 'wr': [], 'wf': []}
for seed in (7, 11, 23, 42, 101, 2026, 3, 55, 77, 123, 314, 999):
    rng = np.random.RandomState(seed)
    me = rng.rand(len(E)) < 0.7; mo = rng.rand(len(O)) < 0.7
    w = logistic_fit(np.vstack([ZE[me], ZO[mo]]),
                     np.concatenate([E.ret.values[me], O.ret.values[mo]]).astype(float))
    ae = roc_auc_score(E.ret.values[~me].astype(float), logistic_pred(w, ZE[~me]))
    ao = roc_auc_score(O.ret.values[~mo].astype(float), logistic_pred(w, ZO[~mo]))
    res['E'].append(ae); res['O'].append(ao); res['wr'].append(w[1]); res['wf'].append(w[2])
    print(f'  {seed:<9} {ae:>9.4f} {ao:>9.4f} {w[1]:>15.4f} {w[2]:>11.4f}')
for k, lab, afirm in [('E','EdNet',0.856), ('O','OULAD',0.862),
                      ('wr','peso recencia',0.8285), ('wf','peso frequencia',0.1358)]:
    v = np.array(res[k])
    print(f'\n  {lab:<18} media {v.mean():.4f} | desvio {v.std(ddof=1):.4f} | amplitude {v.max()-v.min():.4f}'
          f' | afirmado {afirm:.4f} -> {abs(v.mean()-afirm)/max(v.std(ddof=1),1e-9):.1f} desvios')
print(f'\n  peso da frequencia sempre positivo? {all(x > 0 for x in res["wf"])}')
print(f'  recencia pesa >4x a frequencia em todas? {all(r > 4*abs(f) for r, f in zip(res["wr"], res["wf"]))}')

print('\n=== pesos MEDIOS das 12 particoes sao melhores que os de uma so? ===')
w_med = np.array([0.0, np.mean(res['wr']), np.mean(res['wf'])])
# intercepto medio tambem
ints = []
for seed in (7, 11, 23, 42, 101, 2026, 3, 55, 77, 123, 314, 999):
    rng = np.random.RandomState(seed)
    me = rng.rand(len(E)) < 0.7; mo = rng.rand(len(O)) < 0.7
    ww = logistic_fit(np.vstack([ZE[me], ZO[mo]]),
                      np.concatenate([E.ret.values[me], O.ret.values[mo]]).astype(float))
    ints.append(ww[0])
w_med[0] = np.mean(ints)
w_atual = np.array([0.8462, 0.8285, 0.1358])
print(f'  {"conjunto de pesos":<24} {"EdNet":>9} {"OULAD":>9}')
for lab, w in [('publicado (semente 7)', w_atual), ('media das 12 particoes', w_med)]:
    ae = roc_auc_score(E.ret.values.astype(float), logistic_pred(w, ZE))
    ao = roc_auc_score(O.ret.values.astype(float), logistic_pred(w, ZO))
    print(f'  {lab:<24} {ae:>9.4f} {ao:>9.4f}')
print(f'\n  pesos medios: intercepto {w_med[0]:+.4f} | recencia {w_med[1]:+.4f} | frequencia {w_med[2]:+.4f}')
print('  (avaliados na base INTEIRA, entao sao comparaveis entre si, nao com os')
print('   numeros de teste acima - servem para escolher os coeficientes)')
