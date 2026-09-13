"""dominio.PRIOR_GLOBAL=8 e engajamento.JANELA_RECENTE=10 foram escolhidos sem
medida. Varre os dois, com bootstrap para nao trocar constante por ruido."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
def boot(y, s, B=200, seed=1):
    r = np.random.RandomState(seed); n = len(y); o = []
    for _ in range(B):
        i = r.randint(0, n, n)
        if y[i].sum() in (0, len(i)): continue
        o.append(roc_auc_score(y[i], s[i]))
    return np.percentile(o, [2.5, 97.5])
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','part','correct','ts']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.cumsum() - d['correct']
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.cumsum() - d['correct']
te = d[~d.tr & (d.k > 0) & (d.kg > 0)].sample(120000, random_state=1)
y = te.correct.values.astype(float)
qd = te.q_dif.values; at = te.ac_top.values; kk = te.k.values
ag = te.ac_glob.values; kg = te.kg.values
print(f'{len(te):,} previsoes\n=== dominio.PRIOR_GLOBAL (encolhimento do acerto global) ===')
print(f'  {"prior":<8} {"AUC":>7} {"IC95":>18}')
for pg in (2, 4, 8, 16, 32, 64):
    alu = (at + G*3)/(kk + 3)
    glo = (ag + G*pg)/(kg + pg)
    p = 0.60*qd + 0.15*alu + 0.25*glo
    a = roc_auc_score(y, p); lo, hi = boot(y, p, B=40)
    print(f'  {pg:<8} {a:>7.4f} [{lo:.4f}, {hi:.4f}]' + ('   <- padrao' if pg == 8 else ''))
print('\n=== dominio.PRIOR_ALUNO (encolhimento do acerto no topico) ===')
for pa in (1, 2, 3, 5, 8, 16):
    alu = (at + G*pa)/(kk + pa)
    glo = (ag + G*8)/(kg + 8)
    p = 0.60*qd + 0.15*alu + 0.25*glo
    a = roc_auc_score(y, p); lo, hi = boot(y, p, B=40)
    print(f'  {pa:<8} {a:>7.4f} [{lo:.4f}, {hi:.4f}]' + ('   <- padrao' if pa == 3 else ''))
