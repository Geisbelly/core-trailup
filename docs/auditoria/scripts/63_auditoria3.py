"""Auditoria parte 3: pre_avaliacao END-TO-END e engajamento.

O ponto critico do pre_avaliacao: os 0,463 foram medidos sobre features
extraidas pelo pipeline sklearn (01_grafos.py). O MODULO reimplementa o grafo
em Python puro. Se as duas implementacoes divergirem, o numero publicado nao
descreve o que o codigo faz.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, spearman
from trailup_core import pre_avaliacao as PA, engajamento as ENG
FALHAS = []
def check(nome, afirmado, medido, tol):
    ok = abs(afirmado - medido) <= tol
    print(f'  [{"OK " if ok else "FALHA"}] {nome:<48} afirmado {afirmado:>7.3f}  medido {medido:>7.3f}', flush=True)
    if not ok: FALHAS.append((nome, afirmado, medido))

print('=== 7. PRE_AVALIACAO, ponta a ponta (chamando preparar/avaliar) ===', flush=True)
T = HERE.replace('/engaj','/texto')
d = pd.read_parquet(f'{T}/classex_grafo.parquet').dropna(subset=['nota'])
print(f'  {len(d):,} respostas | {d.aluno.nunique()} alunos | {d.Task.nunique()} tarefas', flush=True)
# corpus para stopwords: todos os enunciados, gabaritos e respostas
corpus = list(d.q.astype(str)) + list(d.gab.astype(str)) + list(d.resp.astype(str))
stop = PA.stopwords(corpus)
print(f'  stopwords derivadas do corpus: {len(stop)}', flush=True)
notas = []
for tarefa, g in d.groupby('Task'):
    gab = str(g.gab.iloc[0]); enun = str(g.q.iloc[0])
    ref = PA.preparar(gab, enunciado=enun, stop=stop)
    for i, r in zip(g.index, g.resp.astype(str)):
        notas.append((i, PA.avaliar(r, ref).nota))
P = pd.Series(dict(notas)).reindex(d.index)
y = d.nota.values.astype(float)
rho = spearman(P.values, y)
check('pre_avaliacao: Spearman (modulo, ponta a ponta)', 0.463, rho, 0.06)
i10 = np.argsort(P.values)[:10]
print(f'        triagem: 10 piores previstas -> nota real media {y[i10].mean():.2f} (geral {y.mean():.2f})', flush=True)
print(f'        faixa das previsoes: {P.min():.2f} a {P.max():.2f}', flush=True)

print('\n=== 8. ENGAJAMENTO (chamando ordenar/calibrar/trajetoria) ===', flush=True)
for nome, arq in [('EdNet','eixos.parquet'), ('OULAD','oulad_eixos.parquet')]:
    E = pd.read_parquet(f'{HERE}/{arq}')
    if 'rec' not in E.columns:
        print(f'  {nome}: sem sub-janelas, pulando'); continue
    coorte = [{'dias_recentes': int(r.rec), 'dias_ativos': int(round(r.freq*30)),
               'voltou': int(r.ret)} for r in E.itertuples()]
    cal = ENG.calibrar(coorte)
    sc = np.array([ENG.ordenar(c['dias_recentes'], c['dias_ativos'], cal) for c in coorte])
    a = roc_auc_score(E.ret.values, sc)
    alvo = 0.856 if nome == 'EdNet' else 0.862
    check(f'engajamento: ordenar() no {nome}', alvo, a, 0.03)
    # a tabela de referencia bate a retencao observada?
    obs = E.groupby(E.rec.astype(int)).ret.mean()
    tab = ENG.REFERENCIA[nome]
    dif = max(abs(tab[k] - obs.get(k, tab[k])) for k in tab if k in obs.index)
    print(f'        REFERENCIA[{nome}] vs observado: maior diferenca {dif:.3f}', flush=True)
    if dif > 0.02: FALHAS.append((f'REFERENCIA[{nome}]', 0.0, dif))
    # risco() dispara perto da taxa pedida?
    for taxa in (0.10, 0.20):
        disp = np.mean([ENG.risco(c['dias_recentes'], c['dias_ativos'], cal, taxa) for c in coorte])
        print(f'        risco(taxa={taxa:.0%}) dispara em {disp:.1%}', flush=True)

print('\n' + '='*64)
if FALHAS:
    print(f'{len(FALHAS)} FALHAS:')
    for n_,a,m_ in FALHAS: print(f'   {n_}: afirmado {a:.3f}, medido {m_:.3f}')
else:
    print('tudo confere')
