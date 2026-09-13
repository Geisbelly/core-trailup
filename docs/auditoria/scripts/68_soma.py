"""duracao_prevista soma MEDIANAS. Soma de medianas != mediana da soma.
Quanto isso erra numa atividade real? E perfil_chute, os percentis conferem?"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trailup_core import tempo as TMP, chute as CHU
M2 = HERE.replace('/engaj', '/m2')
rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
ql = d[d.tr].groupby('qidx').lat_final.agg(['median','size','mean'])
ql = ql[ql['size'] >= 30]
print('=== duracao_prevista: soma de medianas vs realidade ===')
print(f'  {"tamanho":<10} {"soma medianas":>15} {"mediana real":>14} {"media real":>12} {"erro":>9} {"casos":>9}')
te = d[~d.tr & d.qidx.isin(ql.index)]
for k in (5, 10, 20):
    tot_prev, tot_real = [], []
    for u, g in te.groupby('user'):
        v = g.head(k)
        if len(v) < k: continue
        tot_prev.append(TMP.duracao_prevista(list(v.qidx.map(ql['median']))))
        tot_real.append(float(v.lat_final.sum()))
        if len(tot_prev) >= 3000: break
    if len(tot_prev) < 200: continue
    tp = np.array(tot_prev); tr_ = np.array(tot_real)
    print(f'  {k:<10} {np.median(tp):>15.0f}s {np.median(tr_):>13.0f}s {tr_.mean():>11.0f}s '
          f'{(np.median(tp)/np.median(tr_)-1)*100:>+8.0f}% {len(tp):>9,}')
print('\n  cobertura do preditor: em que fracao dos casos a previsao fica')
print('  dentro de +-25% do real?')
for k in (5, 10, 20):
    tot_prev, tot_real = [], []
    for u, g in te.groupby('user'):
        v = g.head(k)
        if len(v) < k: continue
        tot_prev.append(TMP.duracao_prevista(list(v.qidx.map(ql['median']))))
        tot_real.append(float(v.lat_final.sum()))
        if len(tot_prev) >= 3000: break
    tp = np.array(tot_prev); tr_ = np.array(tot_real)
    dentro = np.mean(np.abs(tp/tr_ - 1) <= 0.25)
    print(f'    {k} questoes: {dentro:.1%}')
print('\n  e somando MEDIAS em vez de medianas (a soma de medias E a media da soma):')
for k in (10,):
    a, b_, c = [], [], []
    for u, g in te.groupby('user'):
        v = g.head(k)
        if len(v) < k: continue
        a.append(float(v.qidx.map(ql['median']).sum()))
        b_.append(float(v.qidx.map(ql['mean']).sum()))
        c.append(float(v.lat_final.sum()))
        if len(a) >= 3000: break
    a, b_, c = np.array(a), np.array(b_), np.array(c)
    print(f'    soma de medianas: mediana {np.median(a):.0f}s | erro medio {np.mean(np.abs(a/c-1)):.1%}')
    print(f'    soma de medias  : mediana {np.median(b_):.0f}s | erro medio {np.mean(np.abs(b_/c-1)):.1%}')
    print(f'    real            : mediana {np.median(c):.0f}s | media {c.mean():.0f}s')

print('\n=== perfil_chute: os percentis 90 e 99 conferem? ===')
qq = d.groupby('qidx').lat_final.agg(['median','size']); qq = qq[qq['size'] >= 50]
E = d[d.qidx.isin(qq.index)].copy()
E['med'] = E.qidx.map(qq['median'])
E['ch'] = [CHU.foi_chute(l, bool(c), m, 50) for l, c, m in zip(E.lat_final, E.correct, E.med)]
taxa = E.groupby('user').ch.mean()
taxa = taxa[E.groupby('user').size() >= 50]
print(f'  {len(taxa):,} alunos | mediana {taxa.median():.4f}')
print(f'  p90 medido: {taxa.quantile(.90):.4f}   (no modulo: 0,020)')
print(f'  p99 medido: {taxa.quantile(.99):.4f}   (no modulo: 0,079)')
for lab, lo in [('dentro do normal', 0), ('acima do normal', 0.020), ('muito acima', 0.079)]:
    m = taxa >= lo
    print(f'    {lab:<20} pega {m.mean():.1%} dos alunos')
