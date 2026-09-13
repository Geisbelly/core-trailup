"""O nulo da trajetoria e ausencia de efeito, ou medida ruim demais?

Se a trajetoria nao se replica entre duas metades independentes das MESMAS
respostas, ela nao mede nada - e o AUC 0,504 e ruido de medida, nao evidencia
de que a mudanca de empenho nao importa. Distincao que muda a conclusao.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import spearman, roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
BLOCO = 25
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts','lat_final'])
d = d.sort_values(['user','ts']).reset_index(drop=True)
q = d.groupby('qidx').agg(qlat=('lat_final','median'), qac=('correct','mean'), qn=('correct','size'))
q = q[q.qn >= 50]; d = d[d.qidx.isin(q.index)].copy()
d['log_qlat'] = np.log1p(d.qidx.map(q.qlat)); d['q_dif'] = 1 - d.qidx.map(q.qac)
d['hab'] = d.user.map(d.groupby('user').correct.mean())
X = np.c_[np.ones(len(d)), d.log_qlat.values, d.q_dif.values, d.hab.values]
beta, *_ = np.linalg.lstsq(X, np.log1p(d.lat_final.values), rcond=None)
d['esf'] = np.log1p(d.lat_final.values) - X @ beta
d['b'] = d.groupby('user').cumcount() // BLOCO
d['par'] = d.groupby(['user','b']).cumcount() % 2     # metades independentes DENTRO do bloco

def monta(sub):
    B = sub.groupby(['user','b']).agg(esf=('esf','mean'), n=('esf','size')).reset_index()
    B = B[B.n >= 10]
    B['nb'] = B.groupby('user').b.transform('size')
    B = B[B.nb >= 5].sort_values(['user','b'])
    def incl(s):
        out = np.full(len(s), np.nan); v = s.values
        for i in range(3, len(v)): out[i] = np.polyfit(range(3), v[i-3:i], 1)[0]
        return pd.Series(out, index=s.index)
    B['slope'] = B.groupby('user').esf.apply(incl).reset_index(level=0, drop=True)
    return B.set_index(['user','b'])[['esf','slope']]

A0, A1 = monta(d[d.par == 0]), monta(d[d.par == 1])
J = A0.join(A1, lsuffix='_a', rsuffix='_b', how='inner').dropna()
print(f'{len(J):,} blocos com as duas metades\n')
print('=== confiabilidade: as duas metades concordam? ===')
print(f'  NIVEL de empenho no bloco   : {spearman(J.esf_a.values, J.esf_b.values):+.3f}')
print(f'  TRAJETORIA (inclinacao)     : {spearman(J.slope_a.values, J.slope_b.values):+.3f}')
print('\n  (abaixo de ~0,3 a medida e ruido: nenhum desfecho poderia ser predito')
print('   por ela, e o AUC 0,50 nao prova que a mudanca de empenho nao importa)')

# teto: com essa confiabilidade, qual o maximo de correlacao observavel?
r = spearman(J.slope_a.values, J.slope_b.values)
rel = 2 * r / (1 + r) if r > 0 else 0.0     # Spearman-Brown para o bloco inteiro
print(f'\n  confiabilidade do bloco inteiro (Spearman-Brown): {rel:.3f}')
if rel > 0:
    print(f'  teto de correlacao com QUALQUER desfecho perfeito: {np.sqrt(rel):.3f}')
