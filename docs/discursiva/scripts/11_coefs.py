"""Conjunto final do pre_avaliacao: as 8 + n_nos + n_arestas. Coeficientes."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(f'{HERE}/classex_grafo.parquet').dropna(subset=['nota'])
y = d.nota.values.astype(float); g = d.aluno.values
BASE = ['no_peso','no_cobertura','cob_conceitos_centrais','divagacao',
        'razao_tam','aresta_cobertura','aresta_peso','log_len']
NOVO = BASE + ['n_nos','n_arestas']
def aval(F):
    P = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, y, groups=g):
        X = d[F].fillna(0).values
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd==0]=1
        m = Ridge(alpha=1.0).fit((X[tr]-mu)/sd, y[tr]); P[te] = m.predict((X[te]-mu)/sd)
    return (spearmanr(y,P).correlation,
            cohen_kappa_score(np.clip(np.round(y),1,5).astype(int),
                              np.clip(np.round(P),1,5).astype(int), weights='quadratic'), P)
r0,q0,_ = aval(BASE); r1,q1,P1 = aval(NOVO)
print(f'  8 features : Spearman {r0:.3f} | QWK {q0:.3f}')
print(f' 10 features : Spearman {r1:.3f} | QWK {q1:.3f}   ({r1-r0:+.3f} / {q1-q0:+.3f})')
X = d[NOVO].fillna(0).values
mu, sd = X.mean(0), X.std(0); sd[sd==0]=1
m = Ridge(alpha=1.0).fit((X-mu)/sd, y)
print('\n  coeficientes no espaco ORIGINAL (para o modulo sem dependencia):')
w = m.coef_/sd; b = m.intercept_ - float((m.coef_*mu/sd).sum())
print(f"    _INTERCEPTO = {b:.6f}")
print('    _PESOS = {')
for n,v in zip(NOVO, w): print(f"        '{n}': {v:+.6f},")
print('    }')
# triagem: as 10 piores previstas
i = np.argsort(P1)[:10]
print(f'\n  triagem: as 10 piores previstas tem nota real media {y[i].mean():.2f} (geral {y.mean():.2f}, dp {y.std():.2f})')

print('\n=== o uso REAL do modulo e triagem. Comparar a cauda, nao so o Spearman ===')
_,_,P0 = aval(BASE)
print(f'  {"quantas piores":<18} {"8 features":>12} {"10 features":>12} {"geral":>8}')
for k in (5, 10, 20, 50):
    i0 = np.argsort(P0)[:k]; i1 = np.argsort(P1)[:k]
    print(f'  {k:<18} {y[i0].mean():>12.2f} {y[i1].mean():>12.2f} {y.mean():>8.2f}')
print(f'\n  {"quantas melhores":<18} {"8 features":>12} {"10 features":>12}')
for k in (5, 10, 20, 50):
    i0 = np.argsort(-P0)[:k]; i1 = np.argsort(-P1)[:k]
    print(f'  {k:<18} {y[i0].mean():>12.2f} {y[i1].mean():>12.2f}')
