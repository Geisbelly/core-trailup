"""Por que "caiu" tem retencao MAIOR que "estavel" dentro da mesma recencia?

Hipotese: tend = (j3-j1)/2. Fixando j3 (a recencia), "caiu" so pode significar
j1 ALTO. Ou seja, dentro de um estrato de recencia, "caiu" identifica quem era
MAIS ativo antes - e atividade total prediz retencao.

Se for isso, o rotulo "engajamento caiu" nao marca risco: marca quem tinha
mais engajamento. E o alerta apontaria para o lado errado.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson
E = pd.read_parquet(f'{HERE}/eixos.parquet'); O = pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
for nome, D in [('EdNet', E), ('OULAD', O)]:
    print('=' * 72); print(f'{nome}'); print('=' * 72)
    D = D.copy()
    D['grupo'] = np.where(D.tend < -0.25, 'caiu', np.where(D.tend.abs() <= 0.25, 'estável', 'subiu'))
    print(f'  {"recência":<12} {"grupo":<9} {"dias ativos j1":>15} {"total j1+j2+j3":>15} {"retenção":>10} {"n":>8}')
    for lo, hi, lab in [(1,3,'1-3 dias'), (4,7,'4-7 dias')]:
        sel = D[(D.rec >= lo) & (D.rec <= hi)]
        for g in ['caiu','estável','subiu']:
            s = sel[sel.grupo == g]
            if len(s) < 50: continue
            tot = (s.j1 + s.j2 + s.j3).mean()
            print(f'  {lab:<12} {g:<9} {s.j1.mean():>15.1f} {tot:>15.1f} {s.ret.mean():>10.1%} {len(s):>8,}')
        print()

print('=' * 72); print('CONTROLE: fixando o TOTAL de dias ativos, a tendência separa?'); print('=' * 72)
for nome, D in [('EdNet', E), ('OULAD', O)]:
    D = D.copy(); D['total'] = D.j1 + D.j2 + D.j3
    D['grupo'] = np.where(D.tend < -0.25, 'caiu', np.where(D.tend.abs() <= 0.25, 'estável', 'subiu'))
    print(f'\n  -- {nome} --')
    print(f'  {"total de dias":<16} {"caiu":>22} {"estável":>22} {"subiu":>22}')
    for lo, hi, lab in [(3,6,'3-6'), (7,12,'7-12'), (13,20,'13-20')]:
        sel = D[(D.total >= lo) & (D.total <= hi)]
        cells = []
        for g in ['caiu','estável','subiu']:
            s = sel.ret[sel.grupo == g]
            if len(s) < 50: cells.append('          -          '); continue
            l, h = wilson(int(s.sum()), len(s))
            cells.append(f'{s.mean():>5.1%} [{l:.0%},{h:.0%}] n={len(s):>5,}')
        print(f'  {lab:<16} ' + ' '.join(f'{c:>22}' for c in cells))
