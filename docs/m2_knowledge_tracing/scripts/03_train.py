"""M2 - dominio por topico. Treino e avaliacao contra a regra em producao.

Split POR USUARIO (nunca por linha): prever a proxima resposta de um aluno ja
visto no treino e outro problema, e nao e o do TrailUp.
"""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(20260912)
df = pd.read_parquet(f'{HERE}/features.parquet')

users = df.user.unique(); rng.shuffle(users)
n = len(users); tr_u, va_u, te_u = users[:int(.8*n)], users[int(.8*n):int(.9*n)], users[int(.9*n):]
split = pd.Series('test', index=users).to_dict()
split.update(dict.fromkeys(tr_u, 'train')); split.update(dict.fromkeys(va_u, 'val'))
df['split'] = df.user.map(split)
tr, va, te = (df[df.split == s] for s in ('train', 'val', 'test'))
print(f'usuarios  treino {len(tr_u):,} | val {len(va_u):,} | teste {len(te_u):,}')
print(f'respostas treino {len(tr):,} | val {len(va):,} | teste {len(te):,}')

# --- dificuldade da questao: SO do treino (senao vaza o teste)
g = tr.groupby('qidx').y.agg(['mean', 'size'])
GLOBAL = tr.y.mean()
PRIOR = 20  # suavizacao: questao com poucas observacoes puxa para a media global
diff = ((g['mean'] * g['size'] + GLOBAL * PRIOR) / (g['size'] + PRIOR))
for d in (tr, va, te):
    d['q_dif'] = d.qidx.map(diff).fillna(GLOBAL).astype(np.float32)
    d['q_n'] = np.log1p(d.qidx.map(g['size']).fillna(0)).astype(np.float32)

FEATS = ['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part',
         'streak','dt_log','dt_part_log','seen_before','ntags','diagnosis',
         'part','q_dif','q_n']

def ece(y, p, bins=10):
    e, idx = 0.0, np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    for b in range(bins):
        m = idx == b
        if m.sum(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e

def evaluate(name, y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'modelo': name, 'AUC': roc_auc_score(y, p), 'logloss': log_loss(y, p),
            'brier': brier_score_loss(y, p), 'ECE': ece(y, p), 'acc': ((p >= .5) == y).mean()}

y_te = te.y.values
rows = []
# --- baselines
rows.append(evaluate('B0 prevalencia (sempre a media)', y_te, np.full(len(te), tr.y.mean())))
rows.append(evaluate('B1 REGRA TrailUp (dominio por part)', y_te, te.regra_trailup.values))
rows.append(evaluate('B2 dificuldade da questao', y_te, te.q_dif.values))
rows.append(evaluate('B3 acerto acumulado do aluno', y_te, te.acc_prev.values))
rows.append(evaluate('B4 EWMA do aluno no topico', y_te, te.ewma_part.values))

# --- modelos
lr = LogisticRegression(max_iter=1000, C=1.0)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(tr[FEATS], tr.y)
rows.append(evaluate('M2-a regressao logistica (15 feats)', y_te, lr.predict_proba(te[FEATS])[:, 1]))

hgb = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_depth=6, min_samples_leaf=200,
    l2_regularization=1.0, early_stopping=True, validation_fraction=None,
    random_state=0, categorical_features=[FEATS.index('part')])
hgb.fit(pd.concat([tr[FEATS]]), pd.concat([tr.y]))
p_va, p_te = hgb.predict_proba(va[FEATS])[:, 1], hgb.predict_proba(te[FEATS])[:, 1]
rows.append(evaluate('M2-b gradient boosting', y_te, p_te))

res = pd.DataFrame(rows).set_index('modelo').round(4)
print('\n=== TESTE (usuarios nunca vistos) ===')
print(res.to_string())

# --- onde a regra quebra
print('\n=== regra do TrailUp: distribuicao do dominio estimado ===')
r = te.regra_trailup.values
print(f'  no teto (>=0.98): {(r >= .98).mean():6.1%} | no piso (<=0.05): {(r <= .05).mean():6.1%} | '
      f'entre 0.2 e 0.8: {((r > .2) & (r < .8)).mean():6.1%}')
print(f'  ponto de equilibrio da regra: p* = 0.07/0.15 = {0.07/0.15:.3f} -> acima disso satura no teto')

# --- cold start: e onde o TrailUp vive (poucas observacoes por aluno)
print('\n=== por numero de respostas previas NO TOPICO (cold start) ===')
bucket = pd.cut(te.n_prev_part, [-1, 4, 19, 99, 10**9], labels=['0-4', '5-19', '20-99', '100+'])
out = []
for b in bucket.cat.categories:
    m = (bucket == b).values
    if m.sum() < 500: continue
    out.append({'n_prev_part': b, 'linhas': int(m.sum()),
                'regra': roc_auc_score(y_te[m], te.regra_trailup.values[m]),
                'logistica': roc_auc_score(y_te[m], lr.predict_proba(te[FEATS])[:, 1][m]),
                'boosting': roc_auc_score(y_te[m], p_te[m])})
print(pd.DataFrame(out).round(4).to_string(index=False))

imp = pd.Series(hgb.feature_importances_ if hasattr(hgb, 'feature_importances_') else np.zeros(len(FEATS)), index=FEATS)
json.dump({'test': res.reset_index().to_dict('records'),
           'n_users': {'train': len(tr_u), 'val': len(va_u), 'test': len(te_u)},
           'n_rows': {'train': len(tr), 'val': len(va), 'test': len(te)},
           'saturacao_regra': {'teto': float((r >= .98).mean()), 'piso': float((r <= .05).mean()),
                               'meio': float(((r > .2) & (r < .8)).mean())},
           'cold_start': out}, open(f'{HERE}/resultados.json', 'w'), ensure_ascii=False, indent=1, default=str)
import pickle; pickle.dump({'model': hgb, 'feats': FEATS, 'diff': diff, 'global': GLOBAL}, open(f'{HERE}/m2_model.pkl', 'wb'))
print(f"\nartefatos: resultados.json, m2_model.pkl ({os.path.getsize(f'{HERE}/m2_model.pkl')/1e6:.1f} MB)")
