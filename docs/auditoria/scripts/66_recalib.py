"""Conserto do dominio: (a) recalibrar p, (b) confianca que meca algo.

(a) A media linear de duas probabilidades COMPRIME para o meio. Correcao
    padrao: expandir no espaco do logito, p' = sigmoid(a + b*logit(p)), com
    a e b ajustados no treino. Duas constantes, sem dependencia.

(b) |p - y| e plano porque e dominado pelo ruido de Bernoulli, nao pelo erro
    da estimativa. O que de fato encolhe com n e a INCERTEZA sobre a taxa do
    aluno. E isso que a confianca deveria reportar.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
from trailup_core import dominio as DOM
M2 = HERE.replace('/engaj', '/m2')
def ece(y, p, b=10):
    e = 0.0; i = np.digitize(p, np.linspace(0, 1, b+1)[1:-1])
    for k in range(b):
        m = i == k
        if m.sum(): e += m.mean()*abs(y[m].mean() - p[m].mean())
    return e
def logit(p): p = np.clip(p, 1e-4, 1-1e-4); return np.log(p/(1-p))
def sig(z): return 1/(1+np.exp(-np.clip(z, -30, 30)))
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','part','correct','ts']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.cumsum() - d['correct']
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.cumsum() - d['correct']
D = d[(d.k > 0) & (d.kg > 0)]
def prever(sub):
    return np.array([DOM.dominio(qd, int(a), int(k), media_global=G,
                                 acertos_totais=int(ag), respostas_totais=int(kg)).p
                     for qd,a,k,ag,kg in zip(sub.q_dif, sub.ac_top, sub.k, sub.ac_glob, sub.kg)])
TR = D[D.tr].sample(250000, random_state=2); TE = D[~D.tr].sample(250000, random_state=1)
ptr, pte = prever(TR), prever(TE)
ytr, yte = TR.correct.values.astype(float), TE.correct.values.astype(float)
# ajuste de Platt em 2 parametros, por Newton
X = np.c_[np.ones(len(ptr)), logit(ptr)]
w = np.zeros(2)
for _ in range(60):
    pr = sig(X @ w); W = np.clip(pr*(1-pr), 1e-6, None)
    H = X.T @ (X*W[:,None]) + 1e-6*np.eye(2)
    w = w + np.linalg.solve(H, X.T @ (ytr - pr))
print(f'recalibracao ajustada no TREINO: a = {w[0]:+.6f} | b = {w[1]:+.6f}')
pc = sig(w[0] + w[1]*logit(pte))
print(f'\n  {"":<14} {"AUC":>7} {"ECE":>8} {"faixa":>18}')
print(f'  {"antes":<14} {roc_auc_score(yte,pte):>7.3f} {ece(yte,pte):>8.4f} {f"{pte.min():.2f}-{pte.max():.2f}":>18}')
print(f'  {"recalibrado":<14} {roc_auc_score(yte,pc):>7.3f} {ece(yte,pc):>8.4f} {f"{pc.min():.2f}-{pc.max():.2f}":>18}')
print(f'\n  {"decil":<7} {"previsto":>10} {"observado":>11}')
b = pd.qcut(pc, 10, labels=False, duplicates='drop')
t = pd.DataFrame({'b':b,'p':pc,'y':yte}).groupby('b').agg(prev=('p','mean'), obs=('y','mean'))
for i, r in t.iterrows():
    print(f'  {int(i)+1:<7} {r.prev:>10.3f} {r.obs:>11.3f}' + ('  <-- fora' if abs(r.prev-r.obs)>0.05 else ''))

print('\n=== (b) o que DE FATO encolhe com n ===')
print(f'  {"n no topico":<14} {"erro |p-y|":>12} {"desvio da taxa do aluno":>25} {"casos":>10}')
for lo, hi in [(1,3),(4,8),(9,20),(21,50),(51,200),(201,10**9)]:
    m = (TE.k.values >= lo) & (TE.k.values < hi)
    if m.sum() < 500: continue
    err = float(np.mean(np.abs(pc[m] - yte[m])))
    nn = TE.k.values[m]
    taxa = TE.ac_top.values[m]/np.maximum(nn,1)
    sd = float(np.mean(np.sqrt(taxa*(1-taxa)/np.maximum(nn,1))))
    print(f'  {f"{lo}-{hi-1}":<14} {err:>12.3f} {sd:>25.3f} {int(m.sum()):>10,}')
print('\n  -> o erro e plano (ruido de Bernoulli); o desvio da taxa do aluno encolhe.')
print('     E o desvio que a confianca deve reportar.')
