"""pre_avaliacao devolve cobertura e divagacao como PORCENTAGEM, e o exemplo
exibe "cobre 70% do gabarito, divaga 22%". Isso significa alguma coisa?

Nunca foi validado - so a nota foi.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE.replace('/texto','/engaj'))
from _shim import spearman, wilson
import trailup_core.pre_avaliacao as PA
T = HERE
d = pd.read_parquet(f'{T}/classex_grafo.parquet').dropna(subset=['nota'])
corpus = list(d.q.astype(str)) + list(d.gab.astype(str)) + list(d.resp.astype(str))
stop = PA.stopwords(corpus)
linhas = []
for tarefa, g in d.groupby('Task'):
    ref = PA.preparar(str(g.gab.iloc[0]), enunciado=str(g.q.iloc[0]), stop=stop)
    for i, r in zip(g.index, g.resp.astype(str)):
        a = PA.avaliar(r, ref)
        linhas.append((i, a.nota, a.cobertura, a.divagacao, len(a.conceitos_faltando)))
P = pd.DataFrame(linhas, columns=['i','nota_prev','cobertura','divagacao','faltando']).set_index('i')
P['nota'] = d.nota
print(f'{len(P):,} respostas\n')
print('=== as porcentagens significam algo? correlacao com a nota humana ===')
for c in ('nota_prev','cobertura','divagacao','faltando'):
    print(f'  {c:<12} Spearman com a nota real: {spearman(P[c].values, P.nota.values):+.3f}')

print('\n=== a cobertura e interpretavel como "% do gabarito coberto"? ===')
print(f'  {"faixa de cobertura":<22} {"respostas":>10} {"nota real media":>17}')
P['fx'] = pd.cut(P.cobertura, [-.01,.1,.2,.3,.5,1.01],
                 labels=['0-10%','10-20%','20-30%','30-50%','50%+'])
for f, g in P.groupby('fx', observed=True):
    print(f'  {str(f):<22} {len(g):>10,} {g.nota.mean():>17.2f}')
print(f'\n  distribuicao da cobertura: p10 {P.cobertura.quantile(.1):.2f} | '
      f'mediana {P.cobertura.median():.2f} | p90 {P.cobertura.quantile(.9):.2f}')
print(f'  distribuicao da divagacao: p10 {P.divagacao.quantile(.1):.2f} | '
      f'mediana {P.divagacao.median():.2f} | p90 {P.divagacao.quantile(.9):.2f}')

print('\n=== "faltando" aponta os conceitos certos? ===')
print('  (resposta com nota alta deveria ter POUCOS conceitos centrais faltando)')
P['fn'] = pd.cut(P.faltando, [-1,2,4,6,8,100], labels=['0-2','3-4','5-6','7-8','9+'])
for f, g in P.groupby('fn', observed=True):
    if len(g) < 20: continue
    print(f'  {str(f):<10} faltando: {len(g):>5,} respostas | nota real media {g.nota.mean():.2f}')
