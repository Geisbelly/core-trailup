"""engajamento.JANELA_RECENTE = 10 define o eixo mais forte do modulo, e foi
escolhido sem medida. Varre nas DUAS bases, com IC95."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2'); DIA = 86400*1000.0
O_DIR = '/caminho/para/os/datasets/open+university+learning+analytics+dataset'
W1 = 30
def boot(y, s, B=150, seed=1):
    r = np.random.RandomState(seed); n = len(y); o = []
    for _ in range(B):
        i = r.randint(0, n, n)
        if y[i].sum() in (0, len(i)): continue
        o.append(roc_auc_score(y[i], s[i]))
    return np.percentile(o, [2.5, 97.5])

E = pd.read_parquet(f'{HERE}/eixos.parquet')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','ts']).sort_values(['user','ts'])
ini = d.groupby('user').ts.transform('min'); d['dia'] = (d.ts - ini)/DIA
d = d[(d.dia < W1) & (d.user.isin(E.index))]
print(f'EdNet: {len(E):,} alunos | retencao {E.ret.mean():.1%}')
print(f'  {"janela recente":<16} {"AUC":>7} {"IC95":>18}')
for jr in (3, 5, 7, 10, 14, 20, 30):
    s = d[d.dia >= (W1-jr)].groupby('user').dia.apply(lambda x: x.astype(int).nunique())
    v = (s.reindex(E.index).fillna(0)/jr).values
    a = roc_auc_score(E.ret.values, v); lo, hi = boot(E.ret.values, v)
    print(f'  {jr:<16} {a:>7.3f} [{lo:.3f}, {hi:.3f}]' + ('   <- padrao' if jr == 10 else ''))

O = pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
v_ = pd.read_csv(f'{O_DIR}/studentVle.csv', usecols=['code_module','code_presentation','id_student','date'])
v_['aluno'] = v_.code_module + '-' + v_.code_presentation + '|' + v_.id_student.astype(str)
v_ = v_[(v_.date >= 0) & (v_.date < W1) & (v_.aluno.isin(O.index))]
print(f'\nOULAD: {len(O):,} alunos | retencao {O.ret.mean():.1%}')
print(f'  {"janela recente":<16} {"AUC":>7} {"IC95":>18}')
for jr in (3, 5, 7, 10, 14, 20, 30):
    s = v_[v_.date >= (W1-jr)].groupby('aluno').date.nunique()
    val = (s.reindex(O.index).fillna(0)/jr).values
    a = roc_auc_score(O.ret.values, val); lo, hi = boot(O.ret.values, val)
    print(f'  {jr:<16} {a:>7.3f} [{lo:.3f}, {hi:.3f}]' + ('   <- padrao' if jr == 10 else ''))
