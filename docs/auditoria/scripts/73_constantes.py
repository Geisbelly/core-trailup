"""Constantes sem base medida. Mede cada uma.

  ritmo.FRONTEIRA_SEG = 40      define rapida/lenta para TODA questao
  chute.MIN_RESPOSTAS_QUESTAO=50
  discriminacao.MIN_RESPOSTAS=40
  tempo.MIN_RESPOSTAS = 5
  dominio.PRIOR_GLOBAL = 8
  engajamento.MIN_EVENTOS = 30
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, spearman
M2 = HERE.replace('/engaj', '/m2')
d = pd.read_parquet(f'{M2}/responses_v3.parquet', columns=['user','qidx','correct','lat_final','ts'])

print('=== 1. ritmo.FRONTEIRA_SEG = 40s: o dado apoia esse corte? ===')
q = d.groupby('qidx').agg(med=('lat_final','median'), n=('lat_final','size'))
q = q[q.n >= 50]
lat = np.log(q.med.values)
print(f'  {len(q):,} questoes | mediana da mediana: {np.exp(np.median(lat)):.1f}s')
print(f'  p10 {np.exp(np.quantile(lat,.1)):.0f}s | p25 {np.exp(np.quantile(lat,.25)):.0f}s | '
      f'p50 {np.exp(np.quantile(lat,.5)):.0f}s | p75 {np.exp(np.quantile(lat,.75)):.0f}s | p90 {np.exp(np.quantile(lat,.9)):.0f}s')
# procura o corte que MAIS separa em duas gaussianas no log (criterio de Otsu)
xs = np.sort(lat); best = None
for t in np.linspace(np.quantile(lat,.05), np.quantile(lat,.95), 200):
    a, b = lat[lat <= t], lat[lat > t]
    if len(a) < 100 or len(b) < 100: continue
    w = len(a)*len(b)/len(lat)**2
    inter = w*(a.mean()-b.mean())**2     # variancia entre classes (Otsu)
    if best is None or inter > best[1]: best = (t, inter)
print(f'\n  corte que mais separa (Otsu no log): {np.exp(best[0]):.1f}s   <- o modulo usa 40,0s')
# d de Cohen no corte de 40 e no otimo
for lab, t in [('40s (modulo)', np.log(40)), (f'{np.exp(best[0]):.0f}s (otimo)', best[0])]:
    a, b = lat[lat <= t], lat[lat > t]
    dc = abs(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
    print(f'    {lab:<18} d de Cohen {dc:.2f} | {len(a)/len(lat):.0%} rapidas')
print('  (o relatorio afirmava d = 3,30 para a separacao em dois grupos)')

print('\n=== 2. tempo.MIN_RESPOSTAS = 5: a mediana ja e estavel com 5? ===')
rng = np.random.RandomState(3)
g = d[d.qidx.isin(q.index)].groupby('qidx').lat_final
ref = g.median()
print(f'  {"n":<6} {"corr com a mediana de 50+":>26} {"erro relativo medio":>21}')
for n in (3, 5, 8, 15, 30):
    am = g.apply(lambda s, n=n: s.sample(min(n, len(s)), random_state=rng.randint(10**6)).median())
    c = spearman(am.values, ref.values)
    err = float(np.mean(np.abs(am.values/ref.values - 1)))
    print(f'  {n:<6} {c:>26.3f} {err:>20.1%}')

print('\n=== 3. chute.MIN_RESPOSTAS_QUESTAO = 50 e discriminacao.MIN_RESPOSTAS = 40 ===')
print('  a mediana da questao precisa de quantas respostas para o limiar de chute valer?')
print('  (mesma tabela acima: com 30 o erro ja e pequeno; 50 e conservador)')
