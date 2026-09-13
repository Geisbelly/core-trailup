"""prioridade_revisao = 1 - retencao. Ou seja: quanto mais esquecido, mais
prioritario. Isso e verdade?

A literatura de espacamento diz o contrario: revisar quando a retencao esta
BAIXA DEMAIS vira reaprendizado do zero. O ganho maior estaria num meio-termo.

Teste: entre reencontros, qual nivel de retencao prevista rende o maior GANHO
DURADOURO - medido no encontro SEGUINTE (o terceiro).
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import wilson
from trailup_core import revisao as REV
M2 = HERE.replace('/engaj', '/m2')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts']).sort_values(['user','ts'])
d['k'] = d.groupby(['user','qidx']).cumcount()
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
# o encontro SEGUINTE ao reencontro
d['prox_ok'] = d.groupby(['user','qidx']).correct.shift(-1)
d['prox_ts'] = d.groupby(['user','qidx']).ts.shift(-1)
R = d[(d.k >= 1) & d.t_ant.notna() & d.prox_ok.notna()].copy()
R['dias'] = (R.ts - R.t_ant) / 1000 / 86400
R['gap_prox'] = (R.prox_ts - R.ts) / 1000 / 86400
R['ret_prev'] = [REV.retencao(x, bool(a)) for x, a in zip(R.dias, R.ac_ant)]
R['prio'] = 1 - R.ret_prev
print(f'{len(R):,} reencontros que tem um encontro seguinte\n')
print('=== o que a funcao assume: menor retencao = maior prioridade ===')
print('=== o que o dado diz: qual retencao rende mais no encontro SEGUINTE? ===')
R['fx'] = pd.qcut(R.ret_prev, 5, labels=False, duplicates='drop')
print(f'\n  {"quintil de retencao prevista":<30} {"retencao":>10} {"acerta no reencontro":>21} {"acerta no SEGUINTE":>20}')
for q in sorted(R.fx.dropna().unique()):
    s = R[R.fx == q]
    lo, hi = wilson(int(s.prox_ok.sum()), len(s))
    print(f'  q{int(q)+1:<29} {s.ret_prev.mean():>10.3f} {s.correct.mean():>21.1%} {s.prox_ok.mean():>13.1%} [{lo:.0%},{hi:.0%}]')
print('\n  (se prioridade = 1-retencao estivesse certa, revisar o mais esquecido')
print('   deveria render mais no encontro seguinte)')

print('\n=== controlando o que importa: entre os que ERRARAM no reencontro ===')
E = R[R.correct == 0]
print(f'  {len(E):,} casos | {"quintil":<10} {"retencao":>10} {"acerta no SEGUINTE":>20}')
for q in sorted(E.fx.dropna().unique()):
    s = E[E.fx == q]
    if len(s) < 300: continue
    lo, hi = wilson(int(s.prox_ok.sum()), len(s))
    print(f'  {"":<12} q{int(q)+1:<8} {s.ret_prev.mean():>10.3f} {s.prox_ok.mean():>13.1%} [{lo:.0%},{hi:.0%}]  n={len(s):,}')
print('\n=== e entre os que ACERTARAM no reencontro ===')
A = R[R.correct == 1]
for q in sorted(A.fx.dropna().unique()):
    s = A[A.fx == q]
    if len(s) < 300: continue
    lo, hi = wilson(int(s.prox_ok.sum()), len(s))
    print(f'  {"":<12} q{int(q)+1:<8} {s.ret_prev.mean():>10.3f} {s.prox_ok.mean():>13.1%} [{lo:.0%},{hi:.0%}]  n={len(s):,}')
