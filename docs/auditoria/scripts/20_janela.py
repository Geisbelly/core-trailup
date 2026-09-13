"""pre_avaliacao: JANELA=4, DF_STOPWORD=0,5 e MIN_DOCS=10 foram escolhidos sem
medida. Varre cada um, chamando o modulo ponta a ponta."""
import os, sys, numpy as np, pandas as pd, importlib
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE.replace('/texto','/engaj'))
from _shim import spearman
import trailup_core.pre_avaliacao as PA
d = pd.read_parquet(f'{HERE}/classex_grafo.parquet').dropna(subset=['nota'])
y = d.nota.values.astype(float)
corpus = list(d.q.astype(str)) + list(d.gab.astype(str)) + list(d.resp.astype(str))

def rodar(janela=None, df_stop=None):
    if janela is not None: PA.JANELA = janela
    stop = PA.stopwords(corpus, df_max=df_stop if df_stop is not None else PA.DF_STOPWORD)
    notas = {}
    for tarefa, g in d.groupby('Task'):
        ref = PA.preparar(str(g.gab.iloc[0]), enunciado=str(g.q.iloc[0]), stop=stop)
        for i, r in zip(g.index, g.resp.astype(str)):
            notas[i] = PA.avaliar(r, ref).nota
    P = pd.Series(notas).reindex(d.index).values
    i10 = np.argsort(P)[:10]
    return spearman(P, y), float(y[i10].mean()), len(stop)

print(f'{len(d):,} respostas | padrao do modulo: JANELA={PA.JANELA} DF_STOPWORD={PA.DF_STOPWORD}\n')
print('=== JANELA de coocorrencia ===')
print(f'  {"janela":<10} {"Spearman":>10} {"10 piores (nota real)":>23}')
orig = PA.JANELA
for j in (2, 3, 4, 5, 6, 8, 12):
    rho, tri, _ = rodar(janela=j)
    marca = '   <- padrao' if j == orig else ''
    print(f'  {j:<10} {rho:>10.3f} {tri:>23.2f}{marca}')
PA.JANELA = orig
print('\n=== DF_STOPWORD (fracao de documentos acima da qual vira stopword) ===')
print(f'  {"df_max":<10} {"stopwords":>11} {"Spearman":>10} {"10 piores":>12}')
for df in (0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.01):
    rho, tri, ns = rodar(df_stop=df)
    marca = '   <- padrao' if abs(df-0.5) < 1e-9 else ''
    print(f'  {df:<10.2f} {ns:>11} {rho:>10.3f} {tri:>12.2f}{marca}')

print('\n=== a diferenca entre os valores e maior que o ruido? ===')
print('  bootstrap por ALUNO, 300 reamostras')
import numpy as _np
def boot_rho(P, y, grupos, B=300, seed=1):
    r = _np.random.RandomState(seed); us = _np.unique(grupos); out = []
    for _ in range(B):
        sel = r.choice(us, len(us), replace=True)
        idx = _np.concatenate([_np.where(grupos == u)[0] for u in sel])
        out.append(spearman(P[idx], y[idx]))
    return _np.percentile(out, [2.5, 97.5])
g = d.aluno.values
orig_j, orig_df = PA.JANELA, PA.DF_STOPWORD
def preds(janela, df_stop):
    PA.JANELA = janela
    stop = PA.stopwords(corpus, df_max=df_stop)
    notas = {}
    for tarefa, gg in d.groupby('Task'):
        ref = PA.preparar(str(gg.gab.iloc[0]), enunciado=str(gg.q.iloc[0]), stop=stop)
        for i, rr in zip(gg.index, gg.resp.astype(str)):
            notas[i] = PA.avaliar(rr, ref).nota
    return pd.Series(notas).reindex(d.index).values
print(f'  {"config":<24} {"Spearman":>10} {"IC95":>18}')
for lab, j, df in [('JANELA=4, df=0,5 (atual)', 4, 0.5), ('JANELA=4, df=0,4', 4, 0.4),
                   ('JANELA=3, df=0,5', 3, 0.5), ('JANELA=5, df=0,5', 5, 0.5),
                   ('JANELA=8, df=0,5', 8, 0.5)]:
    P = preds(j, df); rho = spearman(P, y); lo, hi = boot_rho(P, y, g)
    print(f'  {lab:<24} {rho:>10.3f} [{lo:.3f}, {hi:.3f}]')
PA.JANELA, PA.DF_STOPWORD = orig_j, orig_df
