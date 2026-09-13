"""JANELA_RECENTE de 10 para 14: recomputa TUDO que dependia de 10.
REFERENCIA, PESOS do ordenador comum, e o corte estrutural."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred, wilson
M2 = HERE.replace('/engaj', '/m2'); DIA = 86400*1000.0
O_DIR = '/caminho/para/os/datasets/open+university+learning+analytics+dataset'
W1, JR = 30, 14
E = pd.read_parquet(f'{HERE}/eixos.parquet')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','ts']).sort_values(['user','ts'])
ini = d.groupby('user').ts.transform('min'); d['dia'] = (d.ts - ini)/DIA
d = d[(d.dia < W1) & (d.user.isin(E.index))]
E['rec14'] = d[d.dia >= (W1-JR)].groupby('user').dia.apply(
    lambda x: x.astype(int).nunique()).reindex(E.index).fillna(0).astype(int)
O = pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
v = pd.read_csv(f'{O_DIR}/studentVle.csv', usecols=['code_module','code_presentation','id_student','date'])
v['aluno'] = v.code_module+'-'+v.code_presentation+'|'+v.id_student.astype(str)
v = v[(v.date >= 0) & (v.date < W1) & (v.aluno.isin(O.index))]
O['rec14'] = v[v.date >= (W1-JR)].groupby('aluno').date.nunique().reindex(O.index).fillna(0).astype(int)

print('=== REFERENCIA nova: retencao por dias ativos nos ultimos 14 ===')
for nome, D in [('EdNet', E), ('OULAD', O)]:
    t = D.groupby('rec14').ret.agg(['mean','size'])
    linha = ', '.join(f'{int(k)}: .{int(round(r["mean"]*1000)):03d}' for k, r in t.iterrows() if r['size'] >= 30)
    print(f"    '{nome}': {{{linha}}},")
    print(f"      (celulas com <30 alunos: {int((t['size']<30).sum())})")

print('\n=== PESOS novos: ordenador comum sobre z da coorte ===')
def z(D, cols):
    X = D[cols].values.astype(float)
    return (X - X.mean(0))/np.where(X.std(0)==0, 1, X.std(0))
rng = np.random.RandomState(7)
E['m'] = rng.rand(len(E)) < 0.7; O['m'] = rng.rand(len(O)) < 0.7
COLS = ['rec14','freq']
ZE, ZO = z(E, COLS), z(O, COLS)
w = logistic_fit(np.vstack([ZE[E.m.values], ZO[O.m.values]]),
                 np.concatenate([E.ret.values[E.m.values], O.ret.values[O.m.values]]).astype(float))
print(f"    PESOS = {{'intercepto': {w[0]:.4f}, 'recencia': {w[1]:.4f}, 'frequencia': {w[2]:.4f}}}")
print(f'\n  {"base":<8} {"modelo unico":>14} {"so aquela base":>16}')
for nome, D, Z in [('EdNet', E, ZE), ('OULAD', O, ZO)]:
    te = ~D.m.values
    wi = logistic_fit(Z[D.m.values], D.ret.values[D.m.values].astype(float))
    print(f'  {nome:<8} {roc_auc_score(D.ret.values[te].astype(float), logistic_pred(w, Z[te])):>14.3f}'
          f' {roc_auc_score(D.ret.values[te].astype(float), logistic_pred(wi, Z[te])):>16.3f}')
print('\n  (com JANELA_RECENTE=10 era: EdNet 0,856 | OULAD 0,862)')

print('\n=== CORTE_RECENCIA: o (0, 3) continua valendo em 14 dias? ===')
for nome, D in [('EdNet', E), ('OULAD', O)]:
    print(f'  -- {nome} --')
    for lo, hi, lab in [(0,0,'0 dias'),(1,3,'1-3'),(4,6,'4-6'),(1,4,'1-4'),(5,14,'5+'),(4,14,'4+'),(7,14,'7+')]:
        m = (D.rec14 >= lo) & (D.rec14 <= hi)
        if m.sum() < 100: continue
        l, h = wilson(int(D.ret[m].sum()), int(m.sum()))
        print(f'     {lab:<8} {m.mean():>6.1%} dos alunos | retencao {D.ret[m].mean():>6.1%} [{l:.1%},{h:.1%}]')
