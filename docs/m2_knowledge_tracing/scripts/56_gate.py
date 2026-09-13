"""LACUNA: o gate nao esta no modulo. So existe como boosting (AUC 0,779).

Pergunta: existe forma fechada que bata a regra do TrailUp (0,703)?

Alvo do gate: a taxa de acerto do aluno nas PROXIMAS 10 respostas do mesmo
topico ficar abaixo de 50%. E a decisao de "vale abrir a LLM para este aluno
neste topico".
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = d.loc[d.tr, 'correct'].mean()
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
d['q_dif'] = d.qidx.map((q['sum'] + G*20)/(q['size']+20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.cumsum() - d['correct']
d['ac_top'] = np.where(d.k > 0, d.ac_top/d.k.clip(lower=1), G)
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.cumsum() - d['correct']
d['ac_glob'] = np.where(d.kg > 0, d.ac_glob/d.kg.clip(lower=1), G)
# dificuldade media das questoes JA vistas no topico
d['dif_top'] = d.groupby(['user','part']).q_dif.cumsum() - d['q_dif']
d['dif_top'] = np.where(d.k > 0, d.dif_top/d.k.clip(lower=1), G)
# regra do TrailUp: base +-0,08/-0,07, clamp
def regra(sub):
    out = np.zeros(len(sub)); p = 0.5; u = -1; pt = {}
    uu = sub.user.values; pp = sub.part.values; cc = sub.correct.values
    for i in range(len(sub)):
        key = (uu[i], pp[i])
        v = pt.get(key, 0.5); out[i] = v
        pt[key] = min(0.98, max(0.05, v + (0.08 if cc[i] else -0.07)))
    return out
d['regra'] = regra(d)
# ALVO: acerto nas proximas 10 do topico < 50%
fut = d.groupby(['user','part'], sort=False).correct.transform(
    lambda s: s.shift(-1).rolling(10, min_periods=10).mean().shift(-9))
d['alvo'] = (fut < 0.5).astype(float)
D = d[fut.notna() & ~d.tr].copy()
y = D.alvo.values
print(f'{len(D):,} pontos de decisao | taxa base {y.mean():.1%}\n')
print(f'  {"criterio":<50} {"AUC":>7} {"lift@10%":>9}')
def av(lab, s):
    s = np.asarray(s, float); m = np.isfinite(s)
    a = roc_auc_score(y[m], s[m]); k = s >= np.quantile(s[m], .9)
    print(f'  {lab:<50} {a:>7.3f} {y[k].mean()/y.mean():>8.1f}x')
    return a
av('regra do TrailUp (dominio < limiar)', -D.regra.values)
av('acerto acumulado no topico', -D.ac_top.values)
av('acerto global do aluno', -D.ac_glob.values)
av('dificuldade media do topico', -D.dif_top.values)
print()
best = None
for wt in (0.2, 0.35, 0.5, 0.65):
    for wg in (0.2, 0.35, 0.5):
        wd = 1 - wt - wg
        if wd < 0: continue
        s = -(wt*D.ac_top.values + wg*D.ac_glob.values + wd*D.dif_top.values)
        a = av(f'{wt:.2f} topico + {wg:.2f} global + {wd:.2f} dif. do topico', s)
        if best is None or a > best[1]: best = ((wt,wg,wd), a)
print(f'\n  melhor forma fechada: {best[0]} -> AUC {best[1]:.3f}')
print(f'  (boosting com 30 features: 0,779 | regra do TrailUp: ver acima)')

print('\n=== pontos de operacao: forma fechada vs regra ===')
s_f = -(0.50*D.ac_top.values + 0.35*D.ac_glob.values + 0.15*D.dif_top.values)
s_r = -D.regra.values
print(f'  {"dispara em":<12} {"fechada prec":>13} {"lift":>6} {"cob":>6} {"regra prec":>12} {"lift":>6} {"cob":>6}')
for taxa in (0.05, 0.10, 0.15, 0.20, 0.30):
    kf = s_f >= np.quantile(s_f, 1-taxa); kr = s_r >= np.quantile(s_r, 1-taxa)
    print(f'  {taxa:<12.0%} {y[kf].mean():>13.1%} {y[kf].mean()/y.mean():>5.1f}x {y[kf].sum()/y.sum():>5.0%}'
          f' {y[kr].mean():>12.1%} {y[kr].mean()/y.mean():>5.1f}x {y[kr].sum()/y.sum():>5.0%}')
print('\n=== e quanto vale so o acerto no topico (o mais simples possivel)? ===')
s_s = -D.ac_top.values
for taxa in (0.10, 0.20):
    k = s_s >= np.quantile(s_s, 1-taxa)
    print(f'  dispara {taxa:.0%}: precisao {y[k].mean():.1%} lift {y[k].mean()/y.mean():.1f}x cobertura {y[k].sum()/y.sum():.0%}')

print('\n=== a regra captura algo que a forma suave nao? combinar ===')
def z(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)
combos = [('so forma fechada', z(s_f)), ('so regra', z(s_r))]
for w in (0.3, 0.5, 0.7):
    combos.append((f'{w:.1f} fechada + {1-w:.1f} regra', w*z(s_f) + (1-w)*z(s_r)))
# a regra satura: marcar "esta no piso" como binario
piso = (D.regra.values <= 0.06).astype(float)
print(f'  alunos no piso da regra (<=0,06): {piso.mean():.1%} | taxa de alvo neles: {y[piso==1].mean():.1%}')
combos.append(('fechada + indicador de piso', z(s_f) + 1.2*piso))
print(f'\n  {"criterio":<34} {"AUC":>7} {"prec@5%":>9} {"prec@10%":>9} {"lift@10%":>9}')
for lab, s in combos:
    a = roc_auc_score(y, s)
    k5 = s >= np.quantile(s, .95); k10 = s >= np.quantile(s, .90)
    print(f'  {lab:<34} {a:>7.3f} {y[k5].mean():>9.1%} {y[k10].mean():>9.1%} {y[k10].mean()/y.mean():>8.1f}x')
