"""Nivel de engajamento - como comportamento ACUMULADO, nao estado inferido.

Quatro componentes observaveis:
  frequencia  - com que frequencia volta
  volume      - quanto faz por sessao
  persistencia- termina o que comeca (nao abandona)
  profundidade- le a explicacao depois de errar

Validacao obrigatoria: (a) e traco estavel? (b) prediz DESFECHO?
Sem as duas, e so um numero bonito.
"""
import os, numpy as np, pandas as pd
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{M2}/responses_v3.parquet').sort_values(['user','ts'])
dt=d.groupby('user').ts.diff()/1000.0
d['sessao']=((dt.isna())|(dt>1800)).groupby(d.user).cumsum()
q=d.groupby('qidx').agg(ql=('lat_final','median'),qn=('correct','size'))
q=q[q.qn>=50]; d=d[d.qidx.isin(q.index)]
d['lat_rel']=d.lat_final/d.qidx.map(q.ql)
d['chute']=((d.lat_rel<0.30)&(d.correct==0)).astype(int)
print(f'{len(d):,} respostas | {d.user.nunique():,} alunos')
g=d.groupby('user')
A=pd.DataFrame({
    'n':g.size(),
    'sessoes':g.sessao.max(),
    'dias':(g.ts.max()-g.ts.min())/1000/86400,
    'por_sessao':g.size()/g.sessao.max().clip(lower=1),
    'quits':g.n_quits.mean(),
    'expl':g.expl_sec.median(),
    'chute':g.chute.mean(),
    'acerto':g.correct.mean(),
})
A=A[(A.n>=50)&(A.dias>=1)]
A['frequencia']=A.sessoes/A.dias.clip(lower=1)
A['persistencia']=1-A.quits/(A.por_sessao+1)
A['profundidade']=np.log1p(A.expl)
print(f'{len(A):,} alunos com >=50 respostas e >=1 dia\n')
COMP=['frequencia','por_sessao','persistencia','profundidade']
# escore: media dos percentis dos 4 componentes (robusto a escala)
R=A[COMP].rank(pct=True)
A['engajamento']=R.mean(axis=1)
print('=== os componentes se correlacionam entre si? ===')
print(A[COMP].corr(method='spearman').round(2).to_string())
print('\n  (se fossem quase independentes, "engajamento" nao seria um construto so)')

print('\n=== (a) e TRACO? metade A vs metade B das respostas ===')
rng=np.random.RandomState(20260912); d['h']=rng.randint(0,2,len(d))
def esc(sub):
    gg=sub.groupby('user')
    B=pd.DataFrame({'n':gg.size(),'sessoes':gg.sessao.nunique(),
        'dias':(gg.ts.max()-gg.ts.min())/1000/86400,
        'por_sessao':gg.size()/gg.sessao.nunique().clip(lower=1),
        'quits':gg.n_quits.mean(),'expl':gg.expl_sec.median()})
    B=B[(B.n>=25)&(B.dias>=1)]
    B['frequencia']=B.sessoes/B.dias.clip(lower=1)
    B['persistencia']=1-B.quits/(B.por_sessao+1)
    B['profundidade']=np.log1p(B.expl)
    return B[COMP].rank(pct=True).mean(axis=1)
e0,e1=esc(d[d.h==0]),esc(d[d.h==1])
c=e0.index.intersection(e1.index)
print(f'  {len(c):,} alunos | correlacao A-B: {e0[c].corr(e1[c]):+.3f}')

print('\n=== (b) prediz DESFECHO? ===')
# desfecho 1: volta depois de 7 dias do ultimo dia observado? (censura: exclui quem parou no fim da base)
fim=d.ts.max()
ult=g.ts.max()
A['ult']=ult
obs=A[(fim-A.ult)/1000/86400>=7]          # so quem teve 7 dias de chance
# reconstroi: teve alguma resposta nos 7 dias seguintes a metade do periodo dele?
meio=(g.ts.min()+g.ts.max())/2
A['meio']=meio
vol=d.merge(A[['meio']],left_on='user',right_index=True)
depois=vol[vol.ts>vol.meio].groupby('user').size()
A['voltou']=(A.index.map(depois).fillna(0)>0).astype(int)
from sklearn.metrics import roc_auc_score
print(f'  voltou apos a metade do periodo: {A.voltou.mean():.1%}')
for c_ in COMP+['engajamento','acerto','n']:
    try: print(f'    {c_:>14}: AUC {roc_auc_score(A.voltou,A[c_]):.3f}')
    except Exception: pass
print('\n  desfecho 2: quantidade total de estudo (proxy de aderencia)')
for c_ in COMP+['engajamento']:
    print(f'    {c_:>14}: Spearman com n de respostas {A[c_].corr(np.log1p(A.n),method="spearman"):+.3f}')
A.to_parquet(f'{HERE}/engajamento.parquet')
