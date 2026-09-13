"""O ganho de 8 para 10 features no pre_avaliacao foi +0,012, com n=1.167 e
IC de largura 0,09. Isso sobrevive a reamostragem?

Teste PAREADO: mesmas dobras para os dois conjuntos, e olhar a distribuicao da
DIFERENCA. Comparar dois IC que se sobrepoem nao responde nada; comparar a
diferenca pareada, sim.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(f'{HERE}/classex_grafo.parquet').dropna(subset=['nota'])
y = d.nota.values.astype(float); g = d.aluno.values
BASE = ['no_peso','no_cobertura','cob_conceitos_centrais','divagacao',
        'razao_tam','aresta_cobertura','aresta_peso','log_len']
NOVO = BASE + ['n_nos','n_arestas']
def rho(F, seed):
    rng = np.random.RandomState(seed)
    us = np.unique(g); rng.shuffle(us)
    ordem = {u: i for i, u in enumerate(us)}
    grp = np.array([ordem[u] for u in g])       # embaralha a atribuicao de dobras
    P = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(d, y, groups=grp):
        X = d[F].fillna(0).values
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        m = Ridge(alpha=1.0).fit((X[tr]-mu)/sd, y[tr]); P[te] = m.predict((X[te]-mu)/sd)
    return spearmanr(y, P).correlation
print(f'{len(d):,} respostas | {len(np.unique(g))} alunos | 40 reparticoes\n')
difs, r8s, r10s = [], [], []
for seed in range(40):
    a, b = rho(BASE, seed), rho(NOVO, seed)
    r8s.append(a); r10s.append(b); difs.append(b - a)
difs = np.array(difs)
print(f'  {"":<22} {"media":>8} {"desvio":>8} {"min":>8} {"max":>8}')
print(f'  {"8 features":<22} {np.mean(r8s):>8.4f} {np.std(r8s):>8.4f} {min(r8s):>8.4f} {max(r8s):>8.4f}')
print(f'  {"10 features":<22} {np.mean(r10s):>8.4f} {np.std(r10s):>8.4f} {min(r10s):>8.4f} {max(r10s):>8.4f}')
print(f'  {"diferenca (pareada)":<22} {difs.mean():>8.4f} {difs.std():>8.4f} {difs.min():>8.4f} {difs.max():>8.4f}')
print(f'\n  o ganho e positivo em {(difs > 0).mean():.0%} das reparticoes')
lo, hi = np.percentile(difs, [2.5, 97.5])
print(f'  IC95 da diferenca pareada: [{lo:+.4f}, {hi:+.4f}]')
print(f'  -> {"o ganho SOBREVIVE" if lo > 0 else "o ganho NAO e distinguivel de zero"}')
print(f'\n  (afirmei no modulo: 0,451 -> 0,463, ganho de +0,012)')

print('\n=== o mesmo teste para DF_STOPWORD, que eu rejeitei olhando IC sobrepostos ===')
sys.path.insert(0, '/caminho/para/o/repo')
import trailup_core.pre_avaliacao as PA
corpus = list(d.q.astype(str)) + list(d.gab.astype(str)) + list(d.resp.astype(str))
def rho_modulo(df_stop, janela, seed):
    PA.JANELA = janela
    stop = PA.stopwords(corpus, df_max=df_stop)
    notas = {}
    for tarefa, gg in d.groupby('Task'):
        ref = PA.preparar(str(gg.gab.iloc[0]), enunciado=str(gg.q.iloc[0]), stop=stop)
        for i, r in zip(gg.index, gg.resp.astype(str)):
            notas[i] = PA.avaliar(r, ref).nota
    P = pd.Series(notas).reindex(d.index).values
    # reamostra ALUNOS para pareamento
    rng = np.random.RandomState(seed)
    us = np.unique(g); sel = rng.choice(us, len(us), replace=True)
    idx = np.concatenate([np.where(g == u)[0] for u in sel])
    return spearmanr(y[idx], P[idx]).correlation
orig_j, orig_df = PA.JANELA, PA.DF_STOPWORD
difs = []
for seed in range(40):
    a = rho_modulo(0.5, 4, seed); b = rho_modulo(0.4, 4, seed)
    difs.append(b - a)
PA.JANELA, PA.DF_STOPWORD = orig_j, orig_df
difs = np.array(difs)
lo, hi = np.percentile(difs, [2.5, 97.5])
print(f'  diferenca pareada (df 0,4 menos df 0,5): media {difs.mean():+.4f} | desvio {difs.std():.4f}')
print(f'  positiva em {(difs > 0).mean():.0%} das reamostras | IC95 [{lo:+.4f}, {hi:+.4f}]')
print(f'  -> {"o ganho SOBREVIVE - eu rejeitei errado" if lo > 0 else "nao distinguivel de zero - rejeitar estava certo"}')
