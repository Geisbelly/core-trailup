"""O ganho de +0,004 do tempo (mediana -> media do log) foi afirmado sem teste
pareado, e o ruido de particao do tempo e 0,012 - tres vezes o ganho.

Teste pareado: MESMAS particoes, diferenca por particao.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
M2 = HERE.replace('/engaj', '/m2')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','lat_final'])
U = d.user.values; Q = d.qidx.values; L = d.lat_final.values
LOG = np.log1p(L)
print(f'{len(d):,} respostas | 20 particoes\n')
difs, r_med, r_log = [], [], []
for seed in range(20):
    rng = np.random.RandomState(seed)
    us = np.unique(U).copy(); rng.shuffle(us)
    tr = np.zeros(U.max()+1, bool); tr[us[:int(.7*len(us))]] = True
    m = tr[U]
    g = pd.DataFrame({'q': Q[m], 'l': L[m], 'lg': LOG[m]}).groupby('q').agg(
        med=('l','median'), mlog=('lg','mean'), n=('l','size'))
    g = g[g.n >= 5]
    mt = (~m) & pd.Series(Q).isin(g.index).values
    y = LOG[mt]; den = ((y - y.mean())**2).sum()
    p1 = np.log1p(pd.Series(Q).map(g['med']).values[mt])
    p2 = pd.Series(Q).map(g['mlog']).values[mt]
    a = 1 - ((y-p1)**2).sum()/den
    b = 1 - ((y-p2)**2).sum()/den
    r_med.append(a); r_log.append(b); difs.append(b - a)
difs = np.array(difs)
print(f'  {"":<22} {"media":>8} {"desvio":>8} {"min":>8} {"max":>8}')
print(f'  {"mediana bruta":<22} {np.mean(r_med):>8.4f} {np.std(r_med):>8.4f} {min(r_med):>8.4f} {max(r_med):>8.4f}')
print(f'  {"media do log":<22} {np.mean(r_log):>8.4f} {np.std(r_log):>8.4f} {min(r_log):>8.4f} {max(r_log):>8.4f}')
print(f'  {"diferenca pareada":<22} {difs.mean():>8.4f} {difs.std():>8.4f} {difs.min():>8.4f} {difs.max():>8.4f}')
lo, hi = np.percentile(difs, [2.5, 97.5])
print(f'\n  positiva em {(difs > 0).mean():.0%} das particoes | IC95 [{lo:+.4f}, {hi:+.4f}]')
print(f'  -> {"o ganho SOBREVIVE ao pareamento" if lo > 0 else "NAO distinguivel de zero"}')
print(f'\n  (o desvio ENTRE particoes e {np.std(r_med):.4f}; a diferenca PAREADA tem desvio')
print(f'   {difs.std():.4f} - e por isso que comparar medias soltas nao responde)')
