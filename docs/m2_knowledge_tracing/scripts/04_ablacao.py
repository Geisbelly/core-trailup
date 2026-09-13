"""Duas perguntas que decidem se isso serve ao TrailUp:

1) Quanto do ganho depende de `q_dif` (dificuldade estimada em 6,9M respostas)?
   O TrailUp nao tem esse corpus: questao nova de professor nasce sem historico.
2) A `confianca` pode ser derivada do numero de observacoes, como pede o plano?
"""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(20260912)
df = pd.read_parquet(f'{HERE}/features.parquet')
users = df.user.unique(); rng.shuffle(users)
n = len(users); tr_u, te_u = users[:int(.8*n)], users[int(.9*n):]
s = pd.Series('test', index=users).to_dict(); s.update(dict.fromkeys(tr_u, 'train'))
df['split'] = df.user.map(s)
tr, te = df[df.split == 'train'], df[df.split == 'test']
g = tr.groupby('qidx').y.agg(['mean','size']); GLOBAL = tr.y.mean()
diff = (g['mean']*g['size'] + GLOBAL*20)/(g['size']+20)
for d in (tr, te):
    d['q_dif'] = d.qidx.map(diff).fillna(GLOBAL).astype(np.float32)
    d['q_n'] = np.log1p(d.qidx.map(g['size']).fillna(0)).astype(np.float32)

COMPORTAMENTO = ['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part',
                 'streak','dt_log','dt_part_log','seen_before','part']
CONTEUDO = ['q_dif','q_n','ntags','diagnosis']
y_te = te.y.values

def run(feats, nome):
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
        categorical_features=[feats.index('part')] if 'part' in feats else None)
    m.fit(tr[feats], tr.y)
    p = np.clip(m.predict_proba(te[feats])[:,1], 1e-6, 1-1e-6)
    return m, p, {'conjunto': nome, 'n_feats': len(feats), 'AUC': roc_auc_score(y_te,p),
                  'logloss': log_loss(y_te,p), 'brier': brier_score_loss(y_te,p)}

rows=[]
m_full, p_full, r = run(COMPORTAMENTO+CONTEUDO, 'completo (comportamento + conteudo)'); rows.append(r)
m_comp, p_comp, r = run(COMPORTAMENTO, 'SO comportamento (sem dificuldade da questao)'); rows.append(r)
_, _, r = run(CONTEUDO, 'SO conteudo (so a questao)'); rows.append(r)
MINIMO = ['acc_prev','ewma_part','n_prev_part','streak','dt_log','part']
m_min, p_min, r = run(MINIMO, 'MINIMO (6 feats que o TrailUp ja tem hoje)'); rows.append(r)
print('=== ablacao (teste, usuarios nunca vistos) ===')
print(pd.DataFrame(rows).set_index('conjunto').round(4).to_string())

print('\n=== questoes frias: desempenho em questoes com POUCO historico no treino ===')
qn = te.qidx.map(g['size']).fillna(0)
for lo,hi,lab in [(0,0,'nunca vista (0)'),(1,50,'1-50'),(51,500,'51-500'),(501,10**9,'500+')]:
    m = ((qn>=lo)&(qn<=hi)).values
    if m.sum()<500: continue
    print(f'  {lab:16s} {m.sum():7,} linhas | completo {roc_auc_score(y_te[m],p_full[m]):.4f} | '
          f'so comportamento {roc_auc_score(y_te[m],p_comp[m]):.4f} | minimo {roc_auc_score(y_te[m],p_min[m]):.4f}')

print('\n=== confianca derivada do n de observacoes no topico ===')
print('  (erro absoluto medio entre probabilidade prevista e taxa real observada)')
for lo,hi,lab in [(0,4,'0-4'),(5,19,'5-19'),(20,99,'20-99'),(100,10**9,'100+')]:
    m = ((te.n_prev_part>=lo)&(te.n_prev_part<=hi)).values
    if m.sum()<500: continue
    err = abs(p_full[m].mean()-y_te[m].mean())
    bins = np.digitize(p_full[m], np.linspace(0,1,11)[1:-1]); e=0.0
    for b in range(10):
        mm = bins==b
        if mm.sum(): e += mm.mean()*abs(y_te[m][mm].mean()-p_full[m][mm].mean())
    print(f'  n_prev_part {lab:6s} {m.sum():7,} linhas | AUC {roc_auc_score(y_te[m],p_full[m]):.4f} | '
          f'ECE {e:.4f} | vies {err:+.4f}')

json.dump({'ablacao': rows}, open(f'{HERE}/ablacao.json','w'), ensure_ascii=False, indent=1, default=str)
import pickle; pickle.dump({'model':m_comp,'feats':COMPORTAMENTO}, open(f'{HERE}/m2_comportamento.pkl','wb'))
print(f"\nmodelo so-comportamento salvo ({os.path.getsize(f'{HERE}/m2_comportamento.pkl')/1e6:.1f} MB)")
