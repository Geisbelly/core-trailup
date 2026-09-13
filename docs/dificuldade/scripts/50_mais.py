"""Busca de melhoria: dominio (terceiro termo), chute (limiar) e revisao
(nunca foi validada como PREDITOR - so a curva foi medida)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = d.loc[d.tr, 'correct'].mean()
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum'] + G*20) / (q['size'] + 20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.transform(lambda s: s.shift().expanding().mean())
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.transform(lambda s: s.shift().expanding().mean())
te = d[~d.tr & d.ac_top.notna() & d.ac_glob.notna()].copy()
y = te.correct.values
print(f'teste: {len(te):,} respostas\n')

print('=== DOMINIO: acrescentar o acerto GLOBAL do aluno ajuda? ===')
conf = te.k.values / (te.k.values + 3)
al_top = conf*te.ac_top.values + (1-conf)*G
cg = te.kg.values / (te.kg.values + 8)
al_glob = cg*te.ac_glob.values + (1-cg)*G
print(f'  {"formula":<52} {"AUC":>7}')
print(f'  {"0,7 questao + 0,3 topico (atual)":<52} {roc_auc_score(y, 0.7*te.q_dif.values+0.3*al_top):>7.3f}')
print(f'  {"0,7 questao + 0,3 global":<52} {roc_auc_score(y, 0.7*te.q_dif.values+0.3*al_glob):>7.3f}')
best=None
for wq in (0.60, 0.65, 0.70):
    for wt in (0.10, 0.15, 0.20, 0.30):
        wg = 1-wq-wt
        if wg < 0: continue
        s = wq*te.q_dif.values + wt*al_top + wg*al_glob
        a = roc_auc_score(y, s)
        if best is None or a > best[-1]: best = (wq, wt, wg, a)
        print(f'  {f"{wq:.2f} questao + {wt:.2f} topico + {wg:.2f} global":<52} {a:>7.3f}')
print(f'  -> melhor: questao {best[0]:.2f} / topico {best[1]:.2f} / global {best[2]:.2f} = {best[3]:.3f}')

print('\n=== CHUTE: o limiar de 30% da mediana e o melhor? ===')
ql = d[d.tr].groupby('qidx').lat_final.agg(['median','size'])
te2 = te[te.qidx.map(ql['size']).fillna(0) >= 50].copy()
te2['rel'] = te2.lat_final / te2.qidx.map(ql['median'])
# validacao: chute = aprendizado zero no reencontro
te2['kk'] = te2.groupby(['user','qidx']).cumcount()
pri = d[~d.tr].sort_values(['user','ts'])
pri['kk'] = pri.groupby(['user','qidx']).cumcount()
prox = pri[pri.kk==1].set_index(['user','qidx']).correct.rename('depois')
E = te2[(te2.kk==0)&(te2.correct==0)].set_index(['user','qidx']).join(prox, how='inner')
print(f'  {len(E):,} casos: errou na 1a vez e reencontrou')
print(f'  {"limiar":<12} {"% marcado":>10} {"acerta ao reencontrar":>22} {"vs nao-chute":>14}')
base = E.depois.mean()
for f in (0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
    m = E.rel < f
    if m.sum() < 200: continue
    lo, hi = wilson(int(E.depois[m].sum()), int(m.sum()))
    print(f'  {f:<12.2f} {m.mean():>10.1%} {E.depois[m].mean():>10.1%} [{lo:.1%},{hi:.1%}] {E.depois[~m].mean():>14.1%}')
print(f'  (base geral: {base:.1%})')

print('\n=== REVISAO: a curva prediz o reencontro? (nunca foi testada assim) ===')
R = pri.copy()
R['t_ant'] = R.groupby(['user','qidx']).ts.shift()
R['ac_ant'] = R.groupby(['user','qidx']).correct.shift()
RR = R[R.t_ant.notna()].copy()
RR['dias'] = (RR.ts - RR.t_ant) / 1000 / 86400
print(f'  {len(RR):,} reencontros')
_A = [(0.04,.901),(1,.907),(3,.873),(7,.859),(21,.835),(60,.819),(180,.831)]
_E = [(0.04,.836),(1,.793),(3,.715),(7,.661),(21,.614),(60,.581),(180,.575)]
def prev(dias, ac):
    t = _A if ac else _E
    if dias <= t[0][0]: return t[0][1]
    for (d0,r0),(d1,r1) in zip(t, t[1:]):
        if dias <= d1:
            w = (dias-d0)/(d1-d0); return r0 + w*(r1-r0)
    return t[-1][1]
p = np.array([prev(dd, a==1) for dd, a in zip(RR.dias.values, RR.ac_ant.values)])
yv = RR.correct.values
print(f'  {"preditor":<38} {"AUC":>7} {"ECE":>7}')
def ece(y,pp,b=10):
    e=0.0; i=np.digitize(pp,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-pp[m].mean())
    return e
print(f'  {"curva de esquecimento (modulo)":<38} {roc_auc_score(yv,p):>7.3f} {ece(yv,p):>7.3f}')
print(f'  {"so acertou antes (sim/nao)":<38} {roc_auc_score(yv,RR.ac_ant.values):>7.3f} {"-":>7}')
qac = d.groupby('qidx').correct.mean()
qd = RR.qidx.map(qac).values
print(f'  {"so a dificuldade da questao":<38} {roc_auc_score(yv,qd):>7.3f} {"-":>7}')
comb = 0.6*qd + 0.4*p
print(f'  {"curva + dificuldade da questao":<38} {roc_auc_score(yv,comb):>7.3f} {ece(yv,comb):>7.3f}')
