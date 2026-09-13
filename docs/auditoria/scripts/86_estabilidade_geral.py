"""Varredura completa, com o que NAO depende da semente computado uma vez so."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/caminho/para/o/repo')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score
from trailup_core import revisao as REV, chute as CHU, dificuldade as DIF
M2 = HERE.replace('/engaj', '/m2')
SEEDS = (7, 11, 23, 42, 101, 2026)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts']).reset_index(drop=True)
print(f'{len(d):,} respostas | pre-computando o que nao muda com a semente...', flush=True)
c = d.correct.values.astype(np.int64); cs = np.concatenate([[0], np.cumsum(c)])
n = len(c); idx = np.arange(n); fut = np.full(n, np.nan)
ok = idx + 11 <= n
fut[ok] = (cs[idx[ok]+11] - cs[idx[ok]+1]) / 10.0
gp = d.groupby(['user','part'])
gc = gp.cumcount().values; sz = gp.correct.transform('size').values
fut[(sz - gc - 1) < 10] = np.nan
ALVO = (fut < 0.5).astype(float)
AT = (gp.correct.cumsum() - d.correct).values / np.maximum(gc, 1)
gu = d.groupby('user'); kg = gu.cumcount().values
AG = (gu.correct.cumsum() - d.correct).values / np.maximum(kg, 1)
gq = d.groupby(['user','qidx'])
TA = gq.ts.shift().values; AA = gq.correct.shift().values
KQ = gq.cumcount().values; NAC = (gq.correct.cumsum() - d.correct).values
DIAS = (d.ts.values - TA) / 1000 / 86400
LOGLAT = np.log1p(d.lat_final.values)
USERS = d.user.values; QIDX = d.qidx.values; CORR = d.correct.values.astype(float)
print('pronto. rodando as particoes...', flush=True)
R = {}
def reg(k, v): R.setdefault(k, []).append(v)
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    us = np.unique(USERS).copy(); rng.shuffle(us)
    tr = np.zeros(USERS.max()+1, bool); tr[us[:int(.7*len(us))]] = True
    m_tr = tr[USERS]; G = float(CORR[m_tr].mean())
    q = pd.DataFrame({'q': QIDX[m_tr], 'c': CORR[m_tr]}).groupby('q').c.agg(['sum','size'])
    qd = pd.Series(QIDX).map((q['sum']+G*20)/(q['size']+20)).fillna(G).values
    sel = np.isfinite(fut) & (~m_tr) & (gc >= 5)
    reg('gate (acumulado linear)', roc_auc_score(ALVO[sel], -(0.50*AT[sel]+0.35*AG[sel]+0.15*qd[sel])))
    mr = np.isfinite(TA) & (~m_tr)
    ii = rng.choice(np.where(mr)[0], 200000, replace=False)
    yv = CORR[ii]
    # np.interp reproduz EXATAMENTE a interpolacao linear de revisao._na_tabela
    xa = np.array([x for x, _ in REV._ACERTOU]); ya = np.array([y for _, y in REV._ACERTOU])
    xe = np.array([x for x, _ in REV._ERROU]);   ye = np.array([y for _, y in REV._ERROU])
    dd_ = DIAS[ii]
    pa_ = np.interp(dd_, xa, ya); pe_ = np.interp(dd_, xe, ye)
    reg('revisao binaria', roc_auc_score(yv, np.where(AA[ii] == 1, pa_, pe_)))
    prop = NAC[ii] / np.maximum(KQ[ii], 1)
    reg('revisao interpolada', roc_auc_score(yv, pe_ + prop * (pa_ - pe_)))
    lt = pd.DataFrame({'q': QIDX[m_tr], 'l': d.lat_final.values[m_tr]}).groupby('q').l.agg(['median','size'])
    lt = lt[lt['size'] >= 5]
    mt = (~m_tr) & pd.Series(QIDX).isin(lt.index).values
    yl = LOGLAT[mt]
    med = pd.Series(QIDX).map(lt['median']).values[mt]
    reg('tempo R2 (mediana)', 1 - ((yl-np.log1p(med))**2).sum()/((yl-yl.mean())**2).sum())
    lg = pd.DataFrame({'q': QIDX[m_tr], 'l': LOGLAT[m_tr]}).groupby('q').l.mean()
    reg('tempo R2 (media do log)', 1 - ((yl-pd.Series(QIDX).map(lg).values[mt])**2).sum()/((yl-yl.mean())**2).sum())
    qq = lt[lt['size'] >= 50]
    m0 = (~m_tr) & (KQ == 0) & (CORR == 0) & pd.Series(QIDX).isin(qq.index).values
    m1 = (~m_tr) & (KQ == 1)
    dep = pd.Series(CORR[m1], index=pd.MultiIndex.from_arrays([USERS[m1], QIDX[m1]]))
    key = pd.MultiIndex.from_arrays([USERS[m0], QIDX[m0]])
    dv = dep.reindex(key).values
    okm = np.isfinite(dv)
    mm = pd.Series(QIDX[m0]).map(qq['median']).values
    ch = np.array([CHU.foi_chute(l, False, x, 60) for l, x in zip(d.lat_final.values[m0], mm)])
    reg('chute: gap no reencontro', float(np.nanmean(dv[okm & ~ch]) - np.nanmean(dv[okm & ch])))
    gg = pd.DataFrame({'q': QIDX, 'c': CORR, 'h': m_tr}).groupby(['q','h']).c.agg(['sum','size']).unstack().dropna()
    gg.columns = ['s0','s1','n0','n1']
    V = gg[(gg.n1 >= 30) & (gg.n0 >= 50)]
    al = (V.s0/V.n0).values
    cob = []
    for s1, n1, a, nb in zip(V.s1.values, V.n1.values, al, V.n0.values):
        f = DIF.estimar(int(s1), int(n1))
        sdp = (f.maximo - f.minimo) / (2*1.645)
        cob.append(abs(a - f.taxa) <= 1.645*np.sqrt(sdp**2 + a*(1-a)/nb))
    reg('dificuldade: cobertura', float(np.mean(cob)))
    print(f'  semente {seed} ok', flush=True)
AFIRM = {'gate (acumulado linear)': 0.746, 'revisao binaria': 0.665, 'revisao interpolada': 0.669,
         'tempo R2 (mediana)': 0.574, 'tempo R2 (media do log)': 0.578,
         'chute: gap no reencontro': 0.061, 'dificuldade: cobertura': 0.89}
print(f'\n  {"medida":<28} {"media":>8} {"desvio":>8} {"ampl.":>7} {"afirmado":>9} {"desvios":>8}')
for k, v in R.items():
    v = np.array(v); sd = v.std(ddof=1)
    a = AFIRM[k]; dd = abs(v.mean()-a)/sd if sd > 0 else 0
    print(f'  {k:<28} {v.mean():>8.4f} {sd:>8.4f} {v.max()-v.min():>7.4f} {a:>9.4f} {dd:>8.1f}'
          + ('  <- FORA' if dd > 2 else ''))
