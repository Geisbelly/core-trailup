"""Dois vieses que a tabela por n pode esconder.

(1) SELECAO DE QUEM CHEGA PRIMEIRO: no EdNet as n primeiras respostas vem de
    quem entrou cedo no app. Numa turma, as n respostas SAO a turma inteira.
    Se quem responde primeiro difere, a tabela superestima.
(2) SOBREVIVENCIA DA QUESTAO: so questoes com >=n+30 respostas entram na linha
    de n. Questoes pouco respondidas podem ser sistematicamente diferentes.
"""
import os, numpy as np, pandas as pd
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','lat_final','ts']).sort_values('ts')
print('=== (1) quem responde primeiro e diferente de quem responde depois? ===')
g=df.groupby('qidx')
ok=g.size()>=130
sub=df[df.qidx.isin(ok[ok].index)]
prim=sub.groupby('qidx').head(30).groupby('qidx').agg(a=('correct','mean'),l=('lat_final','median'))
dep =sub.groupby('qidx').tail(-30).groupby('qidx').agg(a=('correct','mean'),l=('lat_final','median'))
j=prim.join(dep,lsuffix='_p',rsuffix='_d')
print(f'  {len(j):,} questoes')
print(f'  acerto  : primeiras 30 = {j.a_p.mean():.3f} | resto = {j.a_d.mean():.3f} | diferenca {j.a_p.mean()-j.a_d.mean():+.3f}')
print(f'  latencia: primeiras 30 = {j.l_p.median():.1f}s | resto = {j.l_d.median():.1f}s | razao {j.l_p.median()/j.l_d.median():.2f}')
print(f'  correlacao entre as duas metades: acerto {j.a_p.corr(j.a_d):.3f} | latencia {np.log1p(j.l_p).corr(np.log1p(j.l_d)):.3f}')
# comparacao justa: 30 aleatorias vs resto
rng=np.random.RandomState(0)
amostra=sub.groupby('qidx',group_keys=False).apply(lambda x: x.sample(min(30,len(x)),random_state=rng.randint(1<<30)),include_groups=False)
aa=amostra.groupby(amostra.index if amostra.index.name=='qidx' else 'qidx').correct.mean() if False else None
print('\n=== (2) questoes pouco respondidas sao diferentes? ===')
tot=df.groupby('qidx').agg(n=('correct','size'),acerto=('correct','mean'),lat=('lat_final','median'))
for lo,hi,lab in [(0,100,'<100'),(100,300,'100-300'),(300,1000,'300-1000'),(1000,10**9,'1000+')]:
    m=(tot.n>=lo)&(tot.n<hi)
    if m.sum()<50: continue
    print(f'  {lab:>9}: {m.sum():>6,} questoes | acerto {tot.acerto[m].mean():.3f} | latencia {tot.lat[m].median():>5.1f}s')
print('\n=== (3) o quanto o viés (1) afeta a tabela por n ===')
print('  comparando a estimativa das 30 PRIMEIRAS contra 30 ALEATORIAS:')
for N in (30,):
    e_prim=[]; e_alea=[]
    qs=ok[ok].index[:3000]
    for q in qs:
        s=sub[sub.qidx==q]
        if len(s)<N+30: continue
        v=s.correct.iloc[N:].mean()
        e_prim.append(abs(s.correct.iloc[:N].mean()-v))
        e_alea.append(abs(s.correct.sample(N,random_state=1).mean()-v))
    print(f'  erro medio usando as {N} primeiras : {np.mean(e_prim):.4f}')
    print(f'  erro medio usando {N} aleatorias   : {np.mean(e_alea):.4f}')
    print(f'  -> penalidade por pegar as primeiras: {np.mean(e_prim)-np.mean(e_alea):+.4f}')
