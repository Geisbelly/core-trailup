"""A tabela de confianca do ritmo veio da medida de 98% que a auditoria
refutou. Remedir a curva inteira."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trailup_core.ritmo import FRONTEIRA_SEG
from _shim import wilson
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(21)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['qidx','lat_final'])
tam = d.groupby('qidx').size()
grandes = tam[tam >= 200].index
d = d[d.qidx.isin(grandes)]
print(f'{len(grandes):,} questoes com >=200 respostas (referencia estavel)\n')
ref = d.groupby('qidx').lat_final.median()
rot_ref = (ref > FRONTEIRA_SEG)
print(f'  {"n da amostra":<14} {"acerto":>9} {"IC95":>18} {"questoes":>9}')
novo = []
for n in (3, 5, 8, 10, 15, 21, 30, 50):
    acertos = 0; tot = 0
    for _ in range(3):                       # tres reamostragens independentes
        am = d.groupby('qidx').lat_final.apply(
            lambda s, n=n: s.sample(min(n, len(s)), random_state=rng.randint(10**6)).median())
        rot = (am > FRONTEIRA_SEG)
        acertos += int((rot == rot_ref).sum()); tot += len(rot)
    p = acertos/tot; lo, hi = wilson(acertos, tot)
    print(f'  {n:<14} {p:>9.3f} [{lo:.3f}, {hi:.3f}] {len(ref):>9,}')
    novo.append((n, round(p, 3)))
print('\n  tabela para o modulo (n, acerto medido):')
print('  _CONF = [' + ', '.join(f'({n}, {p:.3f})' for n, p in novo if n in (5,10,21,50)) + ']')
