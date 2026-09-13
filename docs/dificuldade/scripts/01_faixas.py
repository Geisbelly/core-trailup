"""Classificador de faixa de dificuldade da questao: facil / media / dificil.

Duas perguntas operacionais:
 1. com n respostas observadas, com que frequencia acerto a faixa?
 2. quando devo dizer "indeterminado" em vez de chutar?

Verdade = taxa de acerto medida numa metade das respostas; estimativa = a outra
metade. Sem isso a avaliacao e circular.
"""
import os, numpy as np, pandas as pd
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct'])
print(f'{len(df):,} respostas finais | {df.qidx.nunique():,} questoes')
rng=np.random.RandomState(20260912)
df=df.sample(frac=1,random_state=20260912)
half=df.groupby('qidx').cumcount()%2            # divide as respostas de cada questao
A=df[half==0].groupby('qidx').correct.agg(['sum','size'])   # estimativa
B=df[half==1].groupby('qidx').correct.agg(['sum','size'])   # verdade
q=A.join(B,lsuffix='_a',rsuffix='_b')
q=q[(q.size_b>=30)]                              # verdade confiavel
q['verdade']=q.sum_b/q.size_b
G=float(df.correct.mean())
print(f'{len(q):,} questoes com >=30 respostas de verdade | acerto global {G:.3f}')
print('distribuicao da dificuldade real:', {f'{k:.0%}':round(v,3) for k,v in
      q.verdade.quantile([.1,.25,.5,.75,.9]).items()})

CORTES=[(0.50,0.80,'fixo 50/80'),(0.55,0.85,'fixo 55/85')]
tercis=q.verdade.quantile([1/3,2/3]).values
CORTES.append((tercis[0],tercis[1],f'tercis ({tercis[0]:.2f}/{tercis[1]:.2f})'))
def faixa(p,lo,hi): return np.where(p<lo,0,np.where(p<hi,1,2))
NOMES=['dificil','media','facil']
for lo,hi,nome in CORTES:
    v=faixa(q.verdade.values,lo,hi)
    print(f'\n{nome}: ' + ', '.join(f'{NOMES[i]} {(v==i).mean():.0%}' for i in range(3)))
lo,hi,nome=CORTES[0]
q['fx_verdade']=faixa(q.verdade.values,lo,hi)

print(f'\n=== acerto da faixa por n de respostas observadas (cortes {nome}) ===')
print(f"{'n':>6} {'questoes':>9} {'acerta':>8} {'erra p/ vizinha':>16} {'erra 2 faixas':>14}")
print('-'*60)
res=[]
for N in (5,10,15,21,30,50,100):
    sub=q[q.size_a>=N]
    if len(sub)<50: continue
    # estimativa com as PRIMEIRAS N respostas (aproximada pela taxa da metade A)
    est=(sub.sum_a/sub.size_a).values
    # ruido binomial de uma amostra de tamanho N
    amostra=rng.binomial(N,np.clip(est,0,1))/N
    sh=(amostra*N+G*15)/(N+15)                  # encolhido, como no doc de turma
    f=faixa(sh,lo,hi); vv=sub.fx_verdade.values
    d=np.abs(f-vv)
    print(f'{N:>6} {len(sub):>9,} {(d==0).mean():>7.0%} {(d==1).mean():>15.0%} {(d==2).mean():>13.0%}')
    res.append((N,(d==0).mean()))

print(f'\n=== e se abster quando o intervalo cruza um corte? ===')
print(f"{'n':>6} {'classifica':>11} {'acerto quando classifica':>25}")
print('-'*46)
for N in (5,10,15,21,30,50):
    sub=q[q.size_a>=N]
    if len(sub)<50: continue
    est=(sub.sum_a/sub.size_a).values
    amostra=rng.binomial(N,np.clip(est,0,1))/N
    sh=(amostra*N+G*15)/(N+15)
    se=np.sqrt(np.clip(sh*(1-sh),1e-6,None)/N)
    # abstem se o intervalo de 1 erro-padrao cruza qualquer corte
    cruza=((sh-se<lo)&(sh+se>lo))|((sh-se<hi)&(sh+se>hi))
    f=faixa(sh,lo,hi); vv=sub.fx_verdade.values
    ok=~cruza
    print(f'{N:>6} {ok.mean():>10.0%} {(f[ok]==vv[ok]).mean():>24.0%}')
q.to_parquet(f'{HERE}/questoes.parquet')
print(f'\n-> questoes.parquet ({len(q):,} questoes com verdade e estimativa)')
