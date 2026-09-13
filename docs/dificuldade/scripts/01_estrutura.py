"""Agrupar questoes pelo PADRAO DE QUEM ACERTA, nao pela taxa agregada.

Passo critico: remover habilidade do aluno e dificuldade da questao. Sem isso
tudo correlaciona com tudo (aluno bom acerta tudo) e o agrupamento so redescobre
a habilidade geral.

Pergunta de fundo: depois de remover isso, SOBRA estrutura? Se nao sobrar, a
dificuldade e unidimensional e agregar por taxa de acerto estava certo.
"""
import os, numpy as np, pandas as pd
M2='/caminho/para/os/intermediarios'
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct'])
df=df.drop_duplicates(['user','qidx'],keep='first')
pq=df.groupby('qidx').size()
top=pq.nlargest(1000).index
d=df[df.qidx.isin(top)]
ca=d.groupby('user').size(); d=d[d.user.isin(ca[ca>=150].index)]
print(f'bloco: {d.qidx.nunique()} questoes x {d.user.nunique():,} alunos | {len(d):,} respostas')
M=d.pivot_table(index='user',columns='qidx',values='correct')
obs=M.notna().values
Y=M.values
print(f'densidade {obs.mean():.1%}\n')

# --- remover habilidade do aluno e dificuldade da questao (aditivo, iterativo)
G=np.nanmean(Y)
theta=np.zeros(Y.shape[0]); beta=np.zeros(Y.shape[1])
for _ in range(30):
    theta=np.nanmean(Y-beta[None,:],axis=1)-G+G*0
    theta=np.nan_to_num(np.nanmean(Y-(G+beta[None,:]),axis=1))
    beta=np.nan_to_num(np.nanmean(Y-(G+theta[:,None]),axis=0))
P=G+theta[:,None]+beta[None,:]
R=np.where(obs,Y-P,np.nan)
print(f'residuo: media {np.nanmean(R):+.4f} | desvio {np.nanstd(R):.4f}')
print(f'  (variancia original {np.nanvar(Y):.4f} -> residual {np.nanvar(R):.4f}: '
      f'habilidade+dificuldade explicam {1-np.nanvar(R)/np.nanvar(Y):.1%})\n')

# --- correlacao entre questoes sobre o residuo
Rz=np.where(np.isnan(R),0,R); m=obs.astype(float)
num=Rz.T@Rz
cnt=m.T@m
s=np.sqrt(np.diag(num)/np.maximum(np.diag(cnt),1))
C=num/np.maximum(cnt,1)/np.outer(s,s)
C=np.clip(np.nan_to_num(C),-1,1); np.fill_diagonal(C,1.0)
fora=C[~np.eye(len(C),dtype=bool)]
print('=== sobrou estrutura no residuo? ===')
print(f'  correlacao entre questoes: media {fora.mean():+.4f} | desvio {fora.std():.4f} | '
      f'|r|>0.1 em {(np.abs(fora)>0.1).mean():.1%}')
ev=np.linalg.eigvalsh(C)[::-1]
print(f'  maiores autovalores: {ev[:8].round(2)}')
print(f'  1o autovalor explica {ev[0]/ev.sum():.1%} | primeiros 5: {ev[:5].sum()/ev.sum():.1%}')
# referencia: quanto daria se nao houvesse estrutura nenhuma
n_al=obs.sum(0).mean()
esp=1/np.sqrt(n_al)
print(f'  desvio esperado por ruido puro (~1/sqrt(n)): {esp:.4f}')
print(f'  -> {"HA estrutura" if fora.std()>esp*1.5 else "NAO ha estrutura alem do ruido"}')
np.save(f'{HERE}/C.npy',C); np.save(f'{HERE}/qids.npy',M.columns.values)
