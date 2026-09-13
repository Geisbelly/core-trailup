"""Da para capturar o ganho da proporcao INTERPOLANDO as duas tabelas que o
modulo ja tem, em vez de guardar uma familia de curvas?"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
d['k'] = d.groupby(['user','qidx']).cumcount()
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
d['n_ac'] = d.groupby(['user','qidx']).correct.cumsum() - d['correct']
R = d[d.t_ant.notna()].copy()
R['dias'] = (R.ts - R.t_ant) / 1000 / 86400
R['prop'] = R.n_ac / R.k
te = R[~R.tr]; yv = te.correct.values
_A = [(0.04,.901),(1,.907),(3,.873),(7,.859),(21,.835),(60,.819),(180,.831)]
_E = [(0.04,.836),(1,.793),(3,.715),(7,.661),(21,.614),(60,.581),(180,.575)]
def interp(t, dias):
    if dias <= t[0][0]: return t[0][1]
    for (d0,r0),(d1,r1) in zip(t, t[1:]):
        if dias <= d1: return r0 + (dias-d0)/(d1-d0)*(r1-r0)
    return t[-1][1]
def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
dias = te.dias.values; acu = te.ac_ant.values; pr = te['prop'].values
pa = np.array([interp(_A, x) for x in dias]); pe = np.array([interp(_E, x) for x in dias])
print(f'  {"variante":<52} {"AUC":>7} {"ECE":>7}')
p_bin = np.where(acu == 1, pa, pe)
print(f'  {"tabelas do modulo, so ultimo resultado (atual)":<52} {roc_auc_score(yv,p_bin):>7.3f} {ece(yv,p_bin):>7.3f}')
p_int = pe + pr * (pa - pe)
print(f'  {"INTERPOLADO pela proporcao de acertos":<52} {roc_auc_score(yv,p_int):>7.3f} {ece(yv,p_int):>7.3f}')
w = 0.5*pr + 0.5*acu
p_mix = pe + w * (pa - pe)
print(f'  {"meio a meio entre proporcao e ultimo":<52} {roc_auc_score(yv,p_mix):>7.3f} {ece(yv,p_mix):>7.3f}')
