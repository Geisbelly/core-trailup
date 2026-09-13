"""As duas que sobraram:
 1. revisao.dias_ate_revisar(alvo) promete: "revise nesse dia e a retencao
    estara no alvo". Promessa direta - da para conferir no dado.
 2. gate.MIN_RESPOSTAS = 5, nunca medido.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson
from trailup_core import revisao as REV
M2 = HERE.replace('/engaj', '/m2')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','part','correct','ts']).sort_values(['user','ts'])
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
R = d[d.t_ant.notna()].copy()
R['dias'] = (R.ts - R.t_ant)/1000/86400

print('=== 1. dias_ate_revisar cumpre a promessa? ===')
print('  se a funcao diz "revise em N dias para reter X%", quem reencontra')
print('  perto de N deveria acertar perto de X%.\n')
print(f'  {"alvo":<8} {"acertou antes":<15} {"dias sugeridos":>15} {"acerto real perto":>19} {"IC95":>16}')
for alvo in (0.60, 0.70, 0.75, 0.80, 0.85):
    for ac in (True, False):
        n = REV.dias_ate_revisar(acertou_antes=ac, retencao_alvo=alvo)
        if not np.isfinite(n):
            print(f'  {alvo:<8.2f} {str(ac):<15} {"nunca cai ao alvo":>15}'); continue
        lo, hi = n*0.7, n*1.4
        m = (R.dias >= lo) & (R.dias <= hi) & (R.ac_ant == (1 if ac else 0))
        if m.sum() < 300:
            print(f'  {alvo:<8.2f} {str(ac):<15} {n:>15.1f} {"poucos casos":>19}'); continue
        real = R.correct[m].mean(); l, h = wilson(int(R.correct[m].sum()), int(m.sum()))
        erro = real - alvo
        print(f'  {alvo:<8.2f} {str(ac):<15} {n:>15.1f} {real:>19.1%} [{l:.1%},{h:.1%}] {erro:+.1%}')

print('\n=== 2. gate.MIN_RESPOSTAS = 5 ===')
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = (d.groupby(['user','part']).correct.cumsum() - d['correct'])/np.maximum(d.k, 1)
c = d.correct.values.astype(np.int64); cs = np.concatenate([[0], np.cumsum(c)])
nn = len(c); idx = np.arange(nn); fut = np.full(nn, np.nan)
ok = idx + 11 <= nn
fut[ok] = (cs[idx[ok]+11] - cs[idx[ok]+1])/10.0
gc = d.groupby(['user','part']).cumcount().values
sz = d.groupby(['user','part']).correct.transform('size').values
fut[(sz - gc - 1) < 10] = np.nan
d['alvo'] = (fut < 0.5).astype(float)
D = d[np.isfinite(fut)]
print(f'  AUC do acerto acumulado no topico, por n de respostas ja dadas:')
print(f'  {"n no topico":<14} {"casos":>10} {"AUC":>8}')
for lo, hi in [(1,3),(3,5),(5,10),(10,20),(20,50),(50,10**9)]:
    m = (D.k >= lo) & (D.k < hi)
    if m.sum() < 1000: continue
    s = D[m]
    print(f'  {f"{lo}-{hi-1}":<14} {int(m.sum()):>10,} {roc_auc_score(s.alvo.values, -s.ac_top.values):>8.3f}'
          + ('   <- abaixo do minimo' if hi <= 5 else ''))
