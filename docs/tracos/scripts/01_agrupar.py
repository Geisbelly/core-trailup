"""Agrupar ALUNOS - a metade que nao tentei.

Questoes nao tinham estrutura (dificuldade unidimensional). Alunos podem ter:
o mesmo residuo, transposto. E ha indicio - a taxa de chute e traco com
confiabilidade 0,844.

Usa metodo que DESCOBRE a quantidade de grupos.
"""
import os, numpy as np, pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import silhouette_score
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(f'{M2}/responses_v3.parquet')
q=df.groupby('qidx').agg(qlat=('lat_final','median'),qac=('correct','mean'),qn=('correct','size'))
q=q[q.qn>=50]
df=df[df.qidx.isin(q.index)]
df['lat_rel']=df.lat_final/df.qidx.map(q.qlat)
df['chute']=((df.lat_rel<0.30)&(df.correct==0)).astype(int)
df['dif_q']=1-df.qidx.map(q.qac)
df=df.sort_values(['user','ts'])
# sessao: intervalo > 30 min entre respostas do mesmo aluno
dt=df.groupby('user').ts.diff()/1000.0
df['sessao']=((dt.isna())|(dt>1800)).groupby(df.user).cumsum()
g=df.groupby('user')
A=pd.DataFrame({
    'n':g.size(),
    'acerto':g.correct.mean(),
    'chute':g.chute.mean(),
    'lat_rel':g.lat_rel.median(),                  # rapido ou lento em relacao aos pares
    'lat_var':g.lat_rel.std(),                     # ritmo constante ou erratico
    'trocas':g.n_trocas.apply(lambda s:(s>0).mean()),
    'dif_enfrentada':g.dif_q.mean(),               # busca questao dificil?
    'expl':g.expl_sec.median(),                    # le explicacao?
    'sessoes':g.sessao.max(),
    'por_sessao':g.size()/g.sessao.max().clip(lower=1),
})
A=A[A.n>=100]
print(f'{len(A):,} alunos com >=100 respostas\n')
F=['acerto','chute','lat_rel','lat_var','trocas','dif_enfrentada','expl','por_sessao']
X=A[F].copy(); X['chute']=np.log1p(X.chute*100); X['expl']=np.log1p(X.expl)
Xs=QuantileTransformer(output_distribution='normal',random_state=0).fit_transform(X)
print('=== HDBSCAN sobre os alunos (descobre a quantidade) ===')
for mcs in (50,100,200,400):
    h=HDBSCAN(min_cluster_size=mcs,min_samples=10).fit(Xs)
    lab=h.labels_; k=len(set(lab))-(1 if -1 in lab else 0)
    sil=silhouette_score(Xs[lab!=-1],lab[lab!=-1]) if k>1 else float('nan')
    print(f'  min_cluster_size={mcs:>3} -> {k} grupos | ruido {(lab==-1).mean():>5.1%} | '
          f'silhueta {sil:.3f} | tamanhos {sorted(np.bincount(lab[lab!=-1]),reverse=True)[:6]}')
h=HDBSCAN(min_cluster_size=200,min_samples=10).fit(Xs); A['g']=h.labels_
k=len(set(A.g))-(1 if -1 in A.g.values else 0)
if k>=2:
    print(f'\n=== perfil dos {k} grupos ===')
    p=A.groupby('g')[F+['n']].median()
    p['alunos']=A.groupby('g').size()
    print(p.round(3).to_string())
    print('\n=== o que mais separa os grupos? ===')
    for c in F:
        gs=[A[c][A.g==i] for i in range(k)]
        if len(gs)<2: continue
        sp=(max(x.median() for x in gs)-min(x.median() for x in gs))/A[c].std()
        print(f'  {c:>16}: separacao {sp:>5.2f} desvios')
A.to_parquet(f'{HERE}/alunos.parquet')
print(f'\n-> alunos.parquet')
