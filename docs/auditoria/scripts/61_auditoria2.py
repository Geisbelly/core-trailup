"""Auditoria, parte 2: gate, revisao e chute. Tudo vetorizado."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
from trailup_core import revisao as REV, chute as CHU, gate as GAT
M2 = HERE.replace('/engaj', '/m2')
FALHAS = []
def check(nome, afirmado, medido, tol):
    ok = abs(afirmado - medido) <= tol
    print(f'  [{"OK " if ok else "FALHA"}] {nome:<50} afirmado {afirmado:>7.3f}  medido {medido:>7.3f}', flush=True)
    if not ok: FALHAS.append((nome, afirmado, medido))

rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
print(f'{len(d):,} respostas | acerto global {G:.4f}', flush=True)

print('\n=== 5. REVISAO ===', flush=True)
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
d['kq'] = d.groupby(['user','qidx']).cumcount()
d['n_ac'] = d.groupby(['user','qidx']).correct.cumsum() - d['correct']
R = d[d.t_ant.notna() & ~d.tr].copy()
R['dias'] = (R.ts - R.t_ant)/1000/86400
Sr = R.sample(min(150000, len(R)), random_state=3)
pb = np.array([REV.retencao(x, bool(a)) for x,a in zip(Sr.dias, Sr.ac_ant)])
pi = np.array([REV.retencao(x, bool(a), proporcao_acertos=float(n)/max(int(k),1))
               for x,a,n,k in zip(Sr.dias, Sr.ac_ant, Sr.n_ac, Sr.kq)])
yv = Sr.correct.values
check('revisao: binario (so o ultimo resultado)', 0.665, roc_auc_score(yv, pb), 0.02)
check('revisao: interpolado pela proporcao', 0.669, roc_auc_score(yv, pi), 0.02)

print('\n=== 6. CHUTE ===', flush=True)
qq = d[d.tr].groupby('qidx').lat_final.agg(['median','size']); qq = qq[qq['size']>=50]
E = d[~d.tr & d.qidx.isin(qq.index)].copy()
prox = E[E.kq==1].set_index(['user','qidx']).correct.rename('depois')
P = E[(E.kq==0)&(E.correct==0)].set_index(['user','qidx']).join(prox, how='inner')
P['med'] = P.index.get_level_values('qidx').map(qq['median'])
P['nq'] = P.index.get_level_values('qidx').map(qq['size'])
ch = np.array([CHU.foi_chute(l, False, m, int(n)) for l,m,n in zip(P.lat_final, P.med, P.nq)])
check('chute: diferenca no reencontro (limiar 0,30)', 0.076,
      float(P.depois[~ch].mean() - P.depois[ch].mean()), 0.02)
print(f'        marcados {ch.mean():.1%} | {len(P):,} casos', flush=True)

print('\n=== 4. GATE ===', flush=True)
dd = d.sort_values(['user','part','ts']).reset_index(drop=True)
c = dd.correct.values.astype(np.int64)
cs = np.concatenate([[0], np.cumsum(c)]); n = len(c); idx = np.arange(n)
fut = np.full(n, np.nan)
ok = idx + 11 <= n
fut[ok] = (cs[idx[ok]+11] - cs[idx[ok]+1]) / 10.0
gc = dd.groupby(['user','part']).cumcount().values
sz = dd.groupby(['user','part']).correct.transform('size').values
fut[(sz - gc - 1) < 10] = np.nan          # janela cruza a fronteira do grupo
dd['alvo'] = (fut < 0.5).astype(float)
dd['k'] = gc
dd['ac_top'] = dd.groupby(['user','part']).correct.cumsum() - dd['correct']
dd['kg'] = dd.groupby('user').cumcount()
dd['ac_glob'] = dd.groupby('user').correct.cumsum() - dd['correct']
Dg = dd[np.isfinite(fut) & ~dd.tr & (dd.k >= 5)].copy()
print(f'  {len(Dg):,} pontos de decisao | taxa base {Dg.alvo.mean():.1%}', flush=True)
Sg = Dg.sample(min(60000, len(Dg)), random_state=2)
seq = {k: v.correct.values.astype(bool) for k, v in dd[~dd.tr].groupby(['user','part'])}
sc = np.array([GAT.risco(seq[(u,p)][:int(k)], ag/max(kg,1), qd, media_global=G).score
               for u,p,k,ag,kg,qd in zip(Sg.user, Sg.part, Sg.k, Sg.ac_glob, Sg.kg, Sg.q_dif)])
yg = Sg.alvo.values
check('gate: AUC da forma combinada', 0.732, roc_auc_score(yg, sc), 0.025)
k10 = sc >= np.quantile(sc, .90)
check('gate: precisao no ponto de 10%', 0.339, float(yg[k10].mean()), 0.05)
print(f'        lift {yg[k10].mean()/yg.mean():.1f}x | base {yg.mean():.1%}', flush=True)

print('\n' + '='*66)
if FALHAS:
    print(f'{len(FALHAS)} FALHAS:')
    for n_,a,m_ in FALHAS: print(f'   {n_}: afirmado {a:.3f}, medido {m_:.3f}')
else:
    print('tudo confere')
