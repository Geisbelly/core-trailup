"""TRAJETORIA DE ENGAJAMENTO: a tendencia acrescenta algo ao nivel?

No empenho a trajetoria foi testada porque o NIVEL nao predizia nada.
Aqui e o contrario: a recencia prediz muito (0,810 / 0,863). Entao a
pergunta muda - a tendencia ACRESCENTA sobre o nivel?

A pergunta de produto, que e a que decide: dois alunos com a MESMA recencia,
um em queda e outro estavel, sao diferentes? Se nao forem, "engajamento caiu"
nao e informacao nova - e so recencia com outro nome.

Janela 0-29 dividida em tres sub-janelas de 10 dias:
    j1 = dias 0-9 | j2 = dias 10-19 | j3 = dias 20-29 (= a recencia)
    tendencia = inclinacao dos dias ativos nas tres
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred, wilson
M2 = HERE.replace('/engaj', '/m2'); DIA = 86400 * 1000.0
O_DIR = '/caminho/para/os/datasets/open+university+learning+analytics+dataset'
W1 = 30

def sub_janelas_ednet(idx):
    d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','ts'])
    d = d.sort_values(['user','ts'])
    ini = d.groupby('user').ts.transform('min'); d['dia'] = (d.ts - ini) / DIA
    d = d[(d.dia < W1) & (d.user.isin(idx))]
    out = {}
    for k, (lo, hi) in enumerate([(0,10),(10,20),(20,30)]):
        s = d[(d.dia >= lo) & (d.dia < hi)]
        out[f'j{k+1}'] = s.groupby('user').dia.apply(lambda x: x.astype(int).nunique())
    return pd.DataFrame(out).reindex(idx).fillna(0)

def sub_janelas_oulad(idx):
    v = pd.read_csv(f'{O_DIR}/studentVle.csv',
                    usecols=['code_module','code_presentation','id_student','date','sum_click'])
    v['aluno'] = v.code_module + '-' + v.code_presentation + '|' + v.id_student.astype(str)
    v = v[(v.date >= 0) & (v.date < W1) & (v.aluno.isin(idx))]
    out = {}
    for k, (lo, hi) in enumerate([(0,10),(10,20),(20,30)]):
        s = v[(v.date >= lo) & (v.date < hi)]
        out[f'j{k+1}'] = s.groupby('aluno').date.nunique()
    return pd.DataFrame(out).reindex(idx).fillna(0)

E = pd.read_parquet(f'{HERE}/eixos.parquet')
O = pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
print('montando sub-janelas...', flush=True)
E = E.join(sub_janelas_ednet(E.index)); O = O.join(sub_janelas_oulad(O.index))
for D in (E, O):
    D['rec'] = D.j3
    D['tend'] = (D.j3 - D.j1) / 2.0                       # inclinacao por janela
    D['tend_rel'] = D.tend / (D[['j1','j2','j3']].mean(axis=1) + 1)
    D['estavel'] = (D[['j1','j2','j3']].std(axis=1) / (D[['j1','j2','j3']].mean(axis=1) + 1))
BASES = [('EdNet', E), ('OULAD', O)]

print('\n' + '=' * 76); print('A. CADA MEDIDA SOZINHA'); print('=' * 76)
print(f'  {"medida":<26} {"EdNet":>8} {"OULAD":>8}')
for c, lab in [('rec','recencia (dias ativos j3)'), ('freq','frequencia (30 dias)'),
               ('tend','tendencia (j3-j1)/2'), ('tend_rel','tendencia relativa'),
               ('estavel','instabilidade (cv)')]:
    row = []
    for _, D in BASES:
        v = D[c].values.astype(float); m = np.isfinite(v)
        row.append(roc_auc_score(D.ret.values[m], v[m]))
    print(f'  {lab:<26} {row[0]:>8.3f} {row[1]:>8.3f}')

print('\n' + '=' * 76); print('B. A TENDENCIA ACRESCENTA SOBRE A RECENCIA?'); print('=' * 76)
print(f'  {"modelo":<34} {"EdNet":>8} {"OULAD":>8}')
for lab, cols in [('só recência', ['rec']),
                  ('recência + frequência', ['rec','freq']),
                  ('recência + tendência', ['rec','tend']),
                  ('recência + freq + tendência', ['rec','freq','tend']),
                  ('recência + freq + instabilidade', ['rec','freq','estavel'])]:
    row = []
    for _, D in BASES:
        rng = np.random.RandomState(4); m = rng.rand(len(D)) < 0.7
        tr, te = D[m], D[~m]
        X = tr[cols].replace([np.inf,-np.inf], 0).fillna(0)
        Xe = te[cols].replace([np.inf,-np.inf], 0).fillna(0)
        mu, sd = X.mean(), X.std().replace(0, 1)
        w = logistic_fit(((X - mu) / sd).values, tr.ret.values)
        row.append(roc_auc_score(te.ret.values, logistic_pred(w, ((Xe - mu) / sd).values)))
    print(f'  {lab:<34} {row[0]:>8.3f} {row[1]:>8.3f}')

print('\n' + '=' * 76)
print('C. A PERGUNTA DE PRODUTO: dentro da MESMA recencia, a queda separa?')
print('=' * 76)
for nome, D in BASES:
    print(f'\n  -- {nome} (retenção base {D.ret.mean():.1%}) --')
    print(f'  {"recência":<16} {"caiu":>20} {"estável":>20} {"subiu":>20}')
    for r_lo, r_hi, lab in [(0,0,'0 dias'), (1,3,'1-3 dias'), (4,7,'4-7 dias'), (8,10,'8-10 dias')]:
        sel = D[(D.rec >= r_lo) & (D.rec <= r_hi)]
        if len(sel) < 200: continue
        cells = []
        for cond, cl in [(sel.tend < -0.25,'caiu'), (sel.tend.abs() <= 0.25,'estável'), (sel.tend > 0.25,'subiu')]:
            s = sel.ret[cond]
            if len(s) < 50: cells.append('        -        '); continue
            lo, hi = wilson(int(s.sum()), len(s))
            cells.append(f'{s.mean():>5.1%} [{lo:.0%},{hi:.0%}] n={len(s):>5,}')
        print(f'  {lab:<16} ' + ' '.join(f'{c:>20}' for c in cells))

print('\n' + '=' * 76); print('D. OULAD: e contra o abandono FORMAL (desmatricula)?'); print('=' * 76)
print(f'  base: {O.saiu_formal.mean():.1%} desmatriculam nos dias 30-59')
print(f'  {"medida":<26} {"AUC":>8}')
for c in ['rec','freq','prof','tend','estavel']:
    v = -O[c].values.astype(float); m = np.isfinite(v)
    print(f'  {c:<26} {roc_auc_score(O.saiu_formal.values[m], v[m]):>8.3f}')
E.to_parquet(f'{HERE}/eixos.parquet'); O.to_parquet(f'{HERE}/oulad_eixos.parquet')
print('\n-> eixos atualizados')
