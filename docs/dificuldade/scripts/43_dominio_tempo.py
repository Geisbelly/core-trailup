"""MELHORIA 2 e 3: dominio e tempo esperado. Split por ALUNO em tudo."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us)
tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = d.loc[d.tr, 'correct'].mean()
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum'] + G*20) / (q['size'] + 20)).fillna(G)
d['q_n'] = d.qidx.map(q['size']).fillna(0)

# acumulados causais do aluno NO TOPICO
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_ant'] = d.groupby(['user','part']).correct.transform(lambda s: s.shift().expanding().mean())
te = d[~d.tr & d.ac_ant.notna()].copy()
print(f'teste: {len(te):,} respostas de {te.user.nunique():,} alunos | acerto {te.correct.mean():.3f}\n')

print('=== DOMINIO: formula fechada ===')
y = te.correct.values
print(f'  {"variante":<44} {"AUC":>7}')
print(f'  {"so a questao":<44} {roc_auc_score(y, te.q_dif.values):>7.3f}')
print(f'  {"so o aluno (acerto no topico)":<44} {roc_auc_score(y, te.ac_ant.values):>7.3f}')
best = None
for w in (0.5, 0.6, 0.7, 0.8, 0.9):
    s = w*te.q_dif.values + (1-w)*te.ac_ant.values
    a = roc_auc_score(y, s)
    if w == 0.7: atual = a
    if best is None or a > best[1]: best = (w, a)
    print(f'  {f"linear, peso da questao = {w:.1f}":<44} {a:>7.3f}' + ('   <- atual' if w == 0.7 else ''))
# encolhimento do termo do aluno por n
for k in (3, 8, 20):
    conf = te.k.values / (te.k.values + k)
    al = conf*te.ac_ant.values + (1-conf)*G
    s = 0.7*te.q_dif.values + 0.3*al
    print(f'  {f"aluno encolhido para a media global (k={k})":<44} {roc_auc_score(y, s):>7.3f}')
# espaco de logito
def logit(p): p = np.clip(p, .02, .98); return np.log(p/(1-p))
for w in (0.6, 0.7, 0.8):
    s = w*logit(te.q_dif.values) + (1-w)*logit(te.ac_ant.values)
    print(f'  {f"combinacao em LOGITO, peso {w:.1f}":<44} {roc_auc_score(y, s):>7.3f}')
# + n de respostas do aluno no topico como terceiro termo
s = 0.7*logit(te.q_dif.values) + 0.3*logit(te.ac_ant.values) + 0.05*np.log1p(te.k.values)
print(f'  {"logito 0,7 + log(n) do aluno":<44} {roc_auc_score(y, s):>7.3f}')

print('\n=== TEMPO ESPERADO: R2 sobre log da latencia ===')
ql = d[d.tr].groupby('qidx').lat_final.agg(['median','size'])
gl = d.loc[d.tr, 'lat_final'].median()
t = d[~d.tr].copy()
yv = np.log1p(t.lat_final.values)
def r2(p): return 1 - ((yv-p)**2).sum() / ((yv-yv.mean())**2).sum()
print(f'  {"variante":<44} {"R2":>7}')
print(f'  {"media global":<44} {r2(np.full(len(t), np.log1p(gl))):>7.3f}')
med = t.qidx.map(ql['median']).fillna(gl).values
print(f'  {"mediana da questao (atual)":<44} {r2(np.log1p(med)):>7.3f}')
nq = t.qidx.map(ql['size']).fillna(0).values
for k in (5, 20, 50):
    wgt = nq / (nq + k)
    p = wgt*np.log1p(med) + (1-wgt)*np.log1p(gl)
    print(f'  {f"mediana encolhida para a global (k={k})":<44} {r2(p):>7.3f}')
# media do log em vez de mediana do bruto
ql2 = d[d.tr].assign(ll=np.log1p(d.lat_final)).groupby('qidx').ll.mean()
p = t.qidx.map(ql2).fillna(np.log1p(gl)).values
print(f'  {"media do LOG da latencia da questao":<44} {r2(p):>7.3f}')
