"""Chute: resposta rapida demais PARA AQUELA QUESTAO, e errada.

Distingue "nao sabe" de "nao tentou" - distincao que o agente_notificacao do
TrailUp deveria fazer e hoje nao faz (ele so ve erro).

Validacao: se foi chute, o aluno NAO aprendeu. Entao ao reencontrar a questao
deve ir pior do que quem errou trabalhando. E o teste que separa o conceito
de uma definicao arbitraria.
"""
import os, numpy as np, pandas as pd
M2='/caminho/para/os/intermediarios'
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','lat_final','n_trocas','ts'])
df=df.sort_values('ts')
print(f'{len(df):,} respostas')
# limiar relativo A CADA questao (questao de 60s e de 15s nao tem o mesmo "rapido")
q=df.groupby('qidx').lat_final.agg(['median','count'])
q=q[q['count']>=50]
df=df[df.qidx.isin(q.index)]
df['lat_rel']=df.lat_final/df.qidx.map(q['median'])
print(f'{len(df):,} respostas em {df.qidx.nunique():,} questoes com >=50 respostas\n')
for p in (0.20,0.30,0.40,0.50):
    m=(df.lat_rel<p)&(df.correct==0)
    print(f'  lat < {p:.0%} da mediana da questao E errada: {m.mean():>5.2%} das respostas')
P=0.30
df['chute']=((df.lat_rel<P)&(df.correct==0)).astype(int)
df['erro_trab']=((df.lat_rel>=P)&(df.correct==0)).astype(int)
print(f'\nusando limiar {P:.0%}: chute {df.chute.mean():.2%} | erro trabalhado {df.erro_trab.mean():.2%}')

print('\n=== o chute e aprendizado zero? (reencontros da MESMA questao) ===')
df['ordem']=df.groupby(['user','qidx']).cumcount()
prim=df[df.ordem==0].set_index(['user','qidx'])
seg =df[df.ordem==1].set_index(['user','qidx'])
j=prim[['chute','erro_trab','correct','lat_rel']].join(seg[['correct']],rsuffix='_2',how='inner')
print(f'  {len(j):,} pares (aluno, questao) com reencontro')
for c,lab in [('chute','errou CHUTANDO'),('erro_trab','errou TRABALHANDO')]:
    m=j[c]==1
    if m.sum()<100: continue
    print(f'  {lab:>22}: acerta no reencontro em {j.correct_2[m].mean():.1%}  ({m.sum():>6,} casos)')
m=j.correct==1
print(f'  {"tinha acertado":>22}: acerta no reencontro em {j.correct_2[m].mean():.1%}  ({m.sum():>6,} casos)')

print('\n=== chutar e traço do aluno ou da situacao? ===')
a=df.groupby('user').agg(chute=('chute','mean'),n=('chute','size'))
a=a[a.n>=50]
print(f'  {len(a):,} alunos com >=50 respostas')
print(f'  taxa de chute por aluno: mediana {a.chute.median():.1%} | p90 {a.chute.quantile(.9):.1%} | p99 {a.chute.quantile(.99):.1%}')
rng=np.random.RandomState(0); df['h']=rng.randint(0,2,len(df))
h=df[df.user.isin(a.index)].groupby(['user','h']).chute.mean().unstack()
h=h.dropna()
print(f'  correlacao entre duas metades do mesmo aluno: {h[0].corr(h[1]):+.3f}')
print(f'  -> {"e TRACO do aluno" if h[0].corr(h[1])>0.4 else "e mais da situacao que do aluno"}')

print('\n=== chute concentra em que tipo de questao? ===')
qc=df.groupby('qidx').agg(chute=('chute','mean'),acerto=('correct','mean'),lat=('lat_final','median'))
print(f'  correlacao chute x dificuldade da questao: {qc.chute.corr(1-qc.acerto):+.3f}')
print(f'  correlacao chute x latencia da questao   : {qc.chute.corr(np.log1p(qc.lat)):+.3f}')
