"""RECAL_A e RECAL_B foram ajustados numa particao e afetam TODO p que o
modulo devolve. Eles se movem entre particoes? E o ponto fixo?"""
import os, sys, math, numpy as np, pandas as pd
sys.path.insert(0, '/caminho/para/o/repo')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
from trailup_core import dominio as DOM
M2 = HERE.replace('/engaj', '/m2')
def sig(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))
def logit(p): p = np.clip(p, 1e-4, 1-1e-4); return np.log(p/(1-p))
def ece(y, p, b=10):
    e = 0.0; i = np.digitize(p, np.linspace(0, 1, b+1)[1:-1])
    for k in range(b):
        m = i == k
        if m.sum(): e += m.mean()*abs(y[m].mean()-p[m].mean())
    return e
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct']).reset_index(drop=True)
U = d.user.values; Q = d.qidx.values; C = d.correct.values.astype(float)
gp = d.groupby(['user','part']); k = gp.cumcount().values
AT = (gp.correct.cumsum() - d.correct).values
gu = d.groupby('user'); kg = gu.cumcount().values
AG = (gu.correct.cumsum() - d.correct).values
print(f'{len(d):,} respostas | 8 particoes\n')
print(f'  {"semente":<9} {"RECAL_A":>10} {"RECAL_B":>10} {"ponto fixo":>12} {"ECE cru":>9} {"ECE recal":>10}')
A, B, F, Ec, Er = [], [], [], [], []
for seed in (7, 11, 23, 42, 101, 2026, 314, 999):
    rng = np.random.RandomState(seed)
    us = np.unique(U).copy(); rng.shuffle(us)
    tr = np.zeros(U.max()+1, bool); tr[us[:int(.7*len(us))]] = True
    m = tr[U]; G = float(C[m].mean())
    q = pd.DataFrame({'q': Q[m], 'c': C[m]}).groupby('q').c.agg(['sum','size'])
    qd = pd.Series(Q).map((q['sum']+G*20)/(q['size']+20)).fillna(G).values
    alu = (AT + G*DOM.PRIOR_ALUNO)/(k + DOM.PRIOR_ALUNO)
    glo = (AG + G*DOM.PRIOR_GLOBAL)/(kg + DOM.PRIOR_GLOBAL)
    praw = DOM.PESO_QUESTAO*qd + DOM.PESO_TOPICO*alu + DOM.PESO_GLOBAL*glo
    sel_tr = m & (k > 0) & (kg > 0); sel_te = (~m) & (k > 0) & (kg > 0)
    itr = rng.choice(np.where(sel_tr)[0], 200000, replace=False)
    ite = rng.choice(np.where(sel_te)[0], 200000, replace=False)
    X = np.c_[np.ones(len(itr)), logit(praw[itr])]; y = C[itr]
    w = np.zeros(2)
    for _ in range(60):
        pr = sig(X @ w); W = np.clip(pr*(1-pr), 1e-6, None)
        H = X.T @ (X*W[:, None]) + 1e-6*np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - pr))
    fixo = 1/(1+math.exp(-(w[0]/(1-w[1])))) if abs(1-w[1]) > 1e-9 else float('nan')
    pc = sig(w[0] + w[1]*logit(praw[ite]))
    e0, e1 = ece(C[ite], praw[ite]), ece(C[ite], pc)
    A.append(w[0]); B.append(w[1]); F.append(fixo); Ec.append(e0); Er.append(e1)
    print(f'  {seed:<9} {w[0]:>10.4f} {w[1]:>10.4f} {fixo:>12.4f} {e0:>9.4f} {e1:>10.4f}')
print()
for lab, v, afirm in [('RECAL_A', A, DOM.RECAL_A), ('RECAL_B', B, DOM.RECAL_B),
                      ('ponto fixo', F, 0.6412), ('ECE recalibrado', Er, 0.0083)]:
    v = np.array(v); sd = v.std(ddof=1)
    print(f'  {lab:<18} media {v.mean():>9.4f} | desvio {sd:.4f} | afirmado {afirm:>9.4f}'
          f' | {abs(v.mean()-afirm)/sd:.1f} desvios' + ('  <- FORA' if abs(v.mean()-afirm)/sd > 2 else ''))
print(f'\n  ECE cru medio {np.mean(Ec):.4f} -> recalibrado {np.mean(Er):.4f} '
      f'(melhora em {sum(1 for a,b in zip(Ec,Er) if b < a)}/8 particoes)')
