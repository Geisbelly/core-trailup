"""MELHORIA: pre_avaliacao linear. Atual 0,447; boosting 0,482; encoder 0,532.
Da para fechar parte do gap sem dependencia?"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(f'{HERE}/classex_grafo.parquet')
alvo = 'nota' if 'nota' in d.columns else [c for c in d.columns if 'nota' in c.lower()][0]
grupo = 'aluno' if 'aluno' in d.columns else [c for c in d.columns if 'alun' in c.lower() or 'user' in c.lower()][0]
# APENAS as features de grafo. NAO usar Run*_AI Evaluation nem Style_Mean:
# sao a avaliacao do proprio LLM, ou seja, vazamento direto do alvo.
X0 = ['no_jaccard','no_cobertura','no_peso','aresta_jaccard','aresta_cobertura',
      'aresta_peso','cob_conceitos_centrais','sem_media','sem_frac','divagacao',
      'razao_tam','n_nos','n_arestas','densidade','log_len']
X0 = [c for c in X0 if c in d.columns]
PROIBIDAS = [c for c in d.columns if 'AI Evaluation' in c or 'Style_Mean' in c or 'Feedback' in c]
assert not (set(X0) & set(PROIBIDAS)), 'feature vazada no conjunto'
print(f'{len(d):,} respostas | alvo={alvo} | grupo={grupo} | {len(X0)} features')
d = d.dropna(subset=[alvo]).copy()
y = d[alvo].values.astype(float)
g = d[grupo].values

def aval(F, modelo='ridge'):
    P = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, y, groups=g):
        Xtr, Xte = d[F].fillna(0).values[tr], d[F].fillna(0).values[te]
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
        m = (Ridge(alpha=1.0) if modelo == 'ridge'
             else HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0))
        m.fit((Xtr - mu) / sd, y[tr]); P[te] = m.predict((Xte - mu) / sd)
    rho = spearmanr(y, P).correlation
    qwk = cohen_kappa_score(np.clip(np.round(y), 1, 5).astype(int),
                            np.clip(np.round(P), 1, 5).astype(int), weights='quadratic')
    return rho, qwk

BASE = [c for c in ['no_peso','no_cobertura','cob_conceitos_centrais','divagacao',
                    'razao_tam','aresta_cobertura','aresta_peso','log_len'] if c in X0]
print(f'\n  {"conjunto":<44} {"Spearman":>9} {"QWK":>7}')
r, q = aval(BASE); print(f'  {"as 8 do modulo (ridge)":<44} {r:>9.3f} {q:>7.3f}')
r, q = aval(X0); print(f'  {f"todas as {len(X0)} (ridge)":<44} {r:>9.3f} {q:>7.3f}')
r, q = aval(X0, 'gb'); print(f'  {f"todas as {len(X0)} (boosting)":<44} {r:>9.3f} {q:>7.3f}')

# termos nao-lineares baratos: quadrado e interacao das 3 mais fortes
D = d.copy()
fortes = sorted(X0, key=lambda c: -abs(np.corrcoef(D[c].fillna(0).values, y)[0,1]))[:3]
print(f'\n  tres features mais correlacionadas: {fortes}')
for c in fortes:
    D[c+'_q'] = D[c].fillna(0) ** 2
D['inter'] = D[fortes[0]].fillna(0) * D[fortes[1]].fillna(0)
d = D
EXT = BASE + [c+'_q' for c in fortes] + ['inter']
r, q = aval(EXT); print(f'  {"as 8 + quadrados + interacao (ridge)":<44} {r:>9.3f} {q:>7.3f}')
EXT2 = X0 + [c+'_q' for c in fortes] + ['inter']
r, q = aval(EXT2); print(f'  {"todas + quadrados + interacao (ridge)":<44} {r:>9.3f} {q:>7.3f}')

print('\n=== quais ganhos sobrevivem SEM dependencia externa? ===')
print('  (sem_media e sem_frac exigem LSA/SVD -> sklearn. As outras sao Python puro.)')
SEM_DEP = [c for c in X0 if c not in ('sem_media','sem_frac')]
EXTRAS = [c for c in SEM_DEP if c not in BASE]
print(f'  extras sem dependencia: {EXTRAS}')
print(f'\n  {"conjunto":<44} {"Spearman":>9} {"QWK":>7}')
r,q = aval(BASE);    print(f'  {"as 8 atuais":<44} {r:>9.3f} {q:>7.3f}')
r,q = aval(SEM_DEP); print(f'  {f"as {len(SEM_DEP)} sem dependencia":<44} {r:>9.3f} {q:>7.3f}')
r,q = aval(X0);      print(f'  {f"as {len(X0)} (com LSA)":<44} {r:>9.3f} {q:>7.3f}')
# quanto cada extra acrescenta sozinho sobre as 8
print(f'\n  ganho de cada extra somado as 8:')
base_r,_ = aval(BASE)
for c in EXTRAS:
    r,_ = aval(BASE+[c])
    print(f'    +{c:<24} {r:>7.3f}  ({r-base_r:+.3f})')
