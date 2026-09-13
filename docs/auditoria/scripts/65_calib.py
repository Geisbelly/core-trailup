"""Varredura: toda saida que afirma ser PROBABILIDADE esta calibrada?

dominio.dominio devolve p como "probabilidade de acertar a proxima" e a
calibracao nunca foi medida - so o AUC. E `confianca` e uma formula que nunca
foi conferida contra nada.
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
te = d[~d.tr & (d.k > 0) & (d.kg > 0)].sample(250000, random_state=1)
p = np.array([DOM.dominio(qd, int(a), int(k), media_global=G,
                          acertos_totais=int(ag), respostas_totais=int(kg)).p
              for qd,a,k,ag,kg in zip(te.q_dif, te.ac_top, te.k, te.ac_glob, te.kg)])
y = te.correct.values.astype(float)
print(f'{len(te):,} previsoes | acerto real {y.mean():.4f} | media prevista {p.mean():.4f}')
print(f'\n  AUC {roc_auc_score(y,p):.3f} | ECE {ece(y,p):.4f}')
print(f'\n  {"decil de p":<12} {"previsto":>10} {"observado":>11} {"n":>9}')
b = pd.qcut(p, 10, labels=False, duplicates='drop')
t = pd.DataFrame({'b':b,'p':p,'y':y}).groupby('b').agg(prev=('p','mean'), obs=('y','mean'), n=('y','size'))
for i, r in t.iterrows():
    marca = '  <-- fora' if abs(r.prev - r.obs) > 0.05 else ''
    print(f'  {int(i)+1:<12} {r.prev:>10.3f} {r.obs:>11.3f} {int(r.n):>9,}{marca}')
print(f'\n  faixa das previsoes: {p.min():.3f} a {p.max():.3f}')

print('\n=== a formula de `confianca` corresponde a alguma coisa? ===')
print('  confianca(n) = min(0,92; 0,35 + 0,57*(1-exp(-n/8)))')
print(f'\n  {"n no topico":<14} {"confianca":>10} {"erro real |p-y| medio":>23} {"n de casos":>12}')
for lo, hi in [(1,3),(4,8),(9,20),(21,50),(51,200),(201,10**9)]:
    m = (te.k.values >= lo) & (te.k.values < hi)
    if m.sum() < 500: continue
    c = DOM.confianca(int((lo+hi)/2) if hi < 10**9 else 300)
    err = float(np.mean(np.abs(p[m] - y[m])))
    print(f'  {f"{lo}-{hi-1}":<14} {c:>10.3f} {err:>23.3f} {int(m.sum()):>12,}')
print('\n  (se a confianca medisse algo, deveria subir quando o erro cai)')
