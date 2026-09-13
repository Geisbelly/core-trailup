"""A recalibracao mudou a ESCALA do p. O que depende de limiar sobre ele
continua valendo? E os limiares do tempo, que nunca foram validados?"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import wilson
from trailup_core import dominio as DOM, tempo as TMP
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.cumsum() - d['correct']
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.cumsum() - d['correct']
te = d[~d.tr & (d.k > 0) & (d.kg > 0)].sample(200000, random_state=1)
def obj(rec):
    return [DOM.dominio(qd, int(a), int(k), media_global=G, acertos_totais=int(ag),
                        respostas_totais=int(kg), recalibrar=rec)
            for qd,a,k,ag,kg in zip(te.q_dif, te.ac_top, te.k, te.ac_glob, te.kg)]
y = te.correct.values.astype(float)
print('=== precisa_reforco: o limiar 0,45 sobrevive a recalibracao? ===')
print(f'  {"versao":<16} {"dispara em":>11} {"acerto de quem dispara":>24} {"esperado":>10}')
for lab, rec in [('cru (antes)', False), ('recalibrado', True)]:
    O = obj(rec)
    f = np.array([DOM.precisa_reforco(o) for o in O])
    print(f'  {lab:<16} {f.mean():>11.1%} {y[f].mean():>24.1%} {"<0,45":>10}')
    if f.sum() > 100:
        lo, hi = wilson(int(y[f].sum()), int(f.sum()))
        print(f'  {"":<16} {"":<11} IC95 [{lo:.1%}, {hi:.1%}]  n={int(f.sum()):,}')
O = obj(True)
print('\n  varrendo o limiar no p recalibrado:')
print(f'  {"limiar":<10} {"dispara em":>11} {"acerto observado":>18} {"o limiar descreve?":>20}')
for lim in (0.30, 0.35, 0.40, 0.45, 0.50):
    f = np.array([DOM.precisa_reforco(o, limiar=lim) for o in O])
    if f.sum() < 100: 
        print(f'  {lim:<10.2f} {f.mean():>11.1%} {"-":>18}'); continue
    ac = y[f].mean()
    print(f'  {lim:<10.2f} {f.mean():>11.1%} {ac:>18.1%} {("sim" if ac <= lim else "NAO"):>20}')
print('  (se o p e probabilidade calibrada, quem dispara com p<L deve acertar <L)')

print('\n=== tempo: o limite de 3x (demorando) foi validado? ===')
ql = d[d.tr].groupby('qidx').lat_final.agg(['median','size']); ql = ql[ql['size'] >= 5]
T = d[~d.tr & d.qidx.isin(ql.index)].sample(300000, random_state=4).copy()
T['med'] = T.qidx.map(ql['median']); T['n'] = T.qidx.map(ql['size'])
rel = (T.lat_final/T.med).values; yt = T.correct.values.astype(float)
print(f'  {"faixa de lentidao":<20} {"acerto":>9} {"IC95":>18} {"respostas":>12}')
for lo_, hi_ in [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,10),(10,1e9)]:
    m = (rel >= lo_) & (rel < hi_)
    if m.sum() < 500: continue
    a = yt[m].mean(); l, h = wilson(int(yt[m].sum()), int(m.sum()))
    print(f'  {f"{lo_}x a {hi_}x":<20} {a:>9.1%} [{l:.1%}, {h:.1%}] {int(m.sum()):>12,}')
print(f'\n  afirmado no modulo: "acima de 3x o acerto cai de 67% para 54%"')
m3 = rel >= 3
print(f'  medido: acima de 3x -> {yt[m3].mean():.1%} | abaixo -> {yt[~m3].mean():.1%} | marca {m3.mean():.1%}')
