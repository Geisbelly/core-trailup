"""EMPENHO por analise regressiva - exatamente o desenho descrito:
taxa de acerto, variacao da taxa, tempo de estudo e tempo de resposta.

empenho = RESIDUO do tempo investido, depois de descontar o que a questao
exige e o que a capacidade do aluno explica. Sem inventar rotulo.

O teste que decide: o ENGAJAMENTO nao prediz aprendizado (+-0,04). Se o
empenho predisser, mede algo que nenhum modulo atual mede.
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, spearman
M2=HERE.replace('/engaj','/m2'); DIA=86400*1000.0
d=pd.read_parquet(f'{M2}/responses_v3.parquet',
    columns=['user','qidx','correct','ts','lat_final','n_trocas','expl_sec','n_quits']).sort_values(['user','ts'])
print(f'{len(d):,} respostas | {d.user.nunique():,} alunos',flush=True)

# ---- insumos que ja existem no modulo ----
q=d.groupby('qidx').agg(qlat=('lat_final','median'),qac=('correct','mean'),qn=('correct','size'))
q=q[q.qn>=50]; d=d[d.qidx.isin(q.index)]
d['q_lat']=d.qidx.map(q.qlat)          # tempo esperado da questao  (tempo.py)
d['q_dif']=1-d.qidx.map(q.qac)         # dificuldade da questao     (dificuldade.py)
hab=d.groupby('user').correct.mean()
d['hab']=d.user.map(hab)               # capacidade do aluno        (dominio.py)
d['log_lat']=np.log1p(d.lat_final)
d['log_qlat']=np.log1p(d.q_lat)

# ---- REGRESSAO: o que explica o tempo investido numa resposta ----
X=np.c_[np.ones(len(d)),d.log_qlat.values,d.q_dif.values,d.hab.values]
y=d.log_lat.values
beta,*_=np.linalg.lstsq(X,y,rcond=None)
pred=X@beta
d['empenho_bruto']=y-pred              # residuo: tempo alem do previsivel
r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
print(f'\nregressao do tempo investido: R2 {r2:.3f}')
for n,b in zip(['intercepto','log tempo esperado da questao','dificuldade da questao','habilidade do aluno'],beta):
    print(f'  {n:<32} {b:+.4f}')
print('  -> o residuo e o que sobra depois de descontar questao e capacidade')

# ---- empenho como TRACO do aluno, e os outros sinais do desenho ----
d['sessao']=((d.groupby('user').ts.diff()/1000.0).isna()|((d.groupby('user').ts.diff()/1000.0)>1800)).groupby(d.user).cumsum()
g=d.groupby('user')
A=pd.DataFrame({
    'n':g.size(),
    'acerto':g.correct.mean(),                       # taxa de acerto
    'empenho':g.empenho_bruto.mean(),                # residuo medio
    'empenho_dp':g.empenho_bruto.std(),
    'trocas':g.n_trocas.mean(),                      # revisou a resposta
    'expl':g.expl_sec.median(),                      # leu a explicacao
    'quits':g.n_quits.mean(),                        # abandonou
    'tempo_estudo':g.lat_final.sum()/3600.0,         # horas somadas
    'sessoes':g.sessao.max(),
    'dias':(g.ts.max()-g.ts.min())/DIA,
})
A=A[A.n>=50]
# variacao da taxa de acerto ao longo do tempo (o "periodo de variacao")
d['bloco']=d.groupby('user').cumcount()//25
bl=d.groupby(['user','bloco']).correct.mean()
A['acerto_var']=bl.groupby('user').std().reindex(A.index)
A['acerto_tend']=bl.groupby('user').apply(lambda s: np.polyfit(range(len(s)),s.values,1)[0] if len(s)>=3 else np.nan).reindex(A.index)
print(f'\n{len(A):,} alunos com >=50 respostas')
print('\ndistribuicao do empenho (residuo medio de log-tempo):')
print('  p10 %.3f | p25 %.3f | mediana %.3f | p75 %.3f | p90 %.3f'%tuple(A.empenho.quantile([.1,.25,.5,.75,.9])))

print('\n=== 1. O EMPENHO E CONFUNDIDO COM CAPACIDADE? ===')
for c in ['acerto','tempo_estudo','sessoes','trocas','expl','quits','acerto_var','acerto_tend']:
    print(f'  correlacao empenho x {c:<14} {spearman(A.empenho.values,A[c].values):+.3f}')

print('\n=== 2. E TRACO ESTAVEL? metades independentes das respostas ===')
rng=np.random.RandomState(5); d['h']=rng.randint(0,2,len(d))
e0=d[d.h==0].groupby('user').empenho_bruto.mean(); e1=d[d.h==1].groupby('user').empenho_bruto.mean()
c=e0.index.intersection(e1.index).intersection(A.index)
print(f'  correlacao A-B: {spearman(e0[c].values,e1[c].values):+.3f}  ({len(c):,} alunos)')
A.to_parquet(f'{HERE}/empenho.parquet'); d[['user','qidx','correct','ts','empenho_bruto','q_dif']].to_parquet(f'{HERE}/empenho_resp.parquet')
print('\n-> empenho.parquet')
