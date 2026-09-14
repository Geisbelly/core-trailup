"""Auditoria da faixa_percentual recem-adicionada. Os cortes 50 e 75 caem
exatamente nas notas 3,0 e 4,0, que sao 87% do corpus."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/Users/user/Documents/geis/core-trailup')
from trailup_core import pre_avaliacao as pa
d = pd.read_parquet('/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/texto/classex_grafo.parquet')
alvo = [c for c in d.columns if 'Content_AI' in c][0]
d = d.dropna(subset=['resp','gab',alvo])
print(f'{len(d)} respostas | alvo {alvo}')
res=[]
for t,(q,g,r) in enumerate(zip(d.q,d.gab,d.resp)):
    ref = pa.preparar(str(g))
    a = pa.avaliar(str(r), ref)
    res.append((a.nota, a.percentual_estimado, a.faixa_percentual))
R = pd.DataFrame(res, columns=['nota','pct','faixa'])
R['real']=pd.to_numeric(d[alvo].values, errors='coerce')
print('\nnota prevista: min %.2f max %.2f media %.2f desvio %.2f' % (R.nota.min(),R.nota.max(),R.nota.mean(),R.nota.std()))
print('\ndistribuicao das faixas:')
for f,n in R.faixa.value_counts().items(): print(f'  {f:<6} {n:>5} ({n/len(R):.1%})')
print('\nnota real media por faixa (o que a faixa deveria separar):')
print(R.groupby('faixa').real.agg(['mean','std','size']).to_string())
b=R[R.faixa!='baixo'] if (R.faixa=='baixo').sum()==0 else R
# quanto encosta no corte
for corte in (50.0,75.0):
    perto=((R.pct-corte).abs()<=1.0).mean()
    print(f'  a menos de 1 ponto do corte {corte:.0f}: {perto:.1%}')
print(f'\n  nota real: min {R.real.min():.1f} max {R.real.max():.1f} media {R.real.mean():.2f}')

print('\n--- o display de % promete demais? ---')
R['real_pct']=(R.real-1)/4*100
for lo,hi in [(0,50),(50,75),(75,90),(90,101)]:
    m=(R.pct>=lo)&(R.pct<hi)
    if m.sum()<5: continue
    print(f'  exibido {lo:>3}-{hi:<3}% (n={m.sum():>4}): real medio {R.real_pct[m].mean():>5.1f}% | desvio {R.real_pct[m].std():>4.1f}')
print(f'\n  exibido medio {R.pct.mean():.1f}% | real medio {R.real_pct.mean():.1f}% | vies {R.pct.mean()-R.real_pct.mean():+.1f} pontos')
m=R.pct>=90
print(f'  dos {m.sum()} exibidos >=90%, o real medio e {R.real_pct[m].mean():.1f}% -- exagero de {R.pct[m].mean()-R.real_pct[m].mean():+.1f} pontos')
from scipy.stats import spearmanr
print(f'\n  Spearman(exibido, real) = {spearmanr(R.pct,R.real).correlation:+.3f}  (ordenacao)')
print(f'  MAE em pontos percentuais = {(R.pct-R.real_pct).abs().mean():.1f}')
