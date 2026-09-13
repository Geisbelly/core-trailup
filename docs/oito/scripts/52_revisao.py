"""MELHORIA: revisao. A curva atual distingue so "acertou / errou na anterior".

Duas hipoteses nunca testadas:
  (a) EFEITO DE ESPACAMENTO: a cada reencontro a curva deveria achatar - o
      3o encontro esquece mais devagar que o 2o.
  (b) o HISTORICO do aluno naquela questao (acertou 2 de 3 vezes) carrega mais
      que so a ultima.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
d['k'] = d.groupby(['user','qidx']).cumcount()
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
# vetorizado: soma dos acertos ANTERIORES = cumsum inclusivo menos o atual
d['n_ac'] = d.groupby(['user','qidx']).correct.cumsum() - d['correct']
R = d[d.t_ant.notna()].copy()
R['dias'] = (R.ts - R.t_ant) / 1000 / 86400
R['bin'] = pd.cut(R.dias, [-.01,.04,1,3,7,21,60,10**6],
                  labels=['<1h','1d','3d','7d','21d','60d','60d+'])
print(f'{len(R):,} reencontros | {R.user.nunique():,} alunos\n')

print('=== (a) EFEITO DE ESPACAMENTO: a curva achata a cada reencontro? ===')
print(f'  {"intervalo":<8} ' + ' '.join(f'{f"k={k}":>14}' for k in (1,2,3,4)))
for b in ['<1h','1d','3d','7d','21d','60d','60d+']:
    row = []
    for k in (1,2,3,4):
        s = R[(R.bin == b) & (R.k == k) & (R.ac_ant == 1)]
        row.append(f'{s.correct.mean():.3f} (n={len(s):>6,})' if len(s) >= 200 else '        -     ')
    print(f'  {b:<8} ' + ' '.join(f'{c:>14}' for c in row))
print('  (so quem ACERTOU no encontro anterior; k = qual reencontro e este)')

print('\n=== (b) o historico completo carrega mais que so a ultima? ===')
tr, te = R[R.tr], R[~R.tr]
def curva(sub, chaves):
    return sub.groupby(chaves).correct.mean()
yv = te.correct.values
# modelo atual: (bin, acertou_antes)
c1 = curva(tr, ['bin','ac_ant'])
p1 = pd.MultiIndex.from_arrays([te.bin, te.ac_ant]).map(c1).astype(float)
# + numero do reencontro
te2 = te.assign(kc=te.k.clip(upper=4)); tr2 = tr.assign(kc=tr.k.clip(upper=4))
c2 = curva(tr2, ['bin','ac_ant','kc'])
p2 = pd.MultiIndex.from_arrays([te2.bin, te2.ac_ant, te2.kc]).map(c2).astype(float)
# + proporcao de acertos anteriores
tr3 = tr2.assign(pa=(tr2.n_ac/tr2.k).round(1)); te3 = te2.assign(pa=(te2.n_ac/te2.k).round(1))
c3 = curva(tr3, ['bin','pa'])
p3 = pd.MultiIndex.from_arrays([te3.bin, te3.pa]).map(c3).astype(float)
c4 = curva(tr3, ['bin','pa','kc'])
p4 = pd.MultiIndex.from_arrays([te3.bin, te3.pa, te3.kc]).map(c4).astype(float)
def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
print(f'  {"modelo":<46} {"AUC":>7} {"ECE":>7} {"sem cobertura":>14}')
for lab, p in [('intervalo x acertou-antes (atual)', p1),
               ('+ numero do reencontro', p2),
               ('intervalo x proporcao de acertos', p3),
               ('intervalo x proporcao x reencontro', p4)]:
    pv = np.asarray(p, dtype=float)
    m = np.isfinite(pv)
    print(f'  {lab:<46} {roc_auc_score(yv[m], pv[m]):>7.3f} {ece(yv[m], pv[m]):>7.3f} {1-m.mean():>13.1%}')
