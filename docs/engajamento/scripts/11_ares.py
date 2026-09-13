"""Os tres eixos no ARES - terceira realidade, e a mais pobre.

200 universitarios lendo textos em L2 por 6 semanas. Janela menor (14/14 dias)
porque o curso inteiro dura ~6 semanas.
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score
M1=HERE.replace('/engaj','/m1')
W1,W2=14,28
R=pd.read_parquet(f'{M1}/leituras.parquet')
R['t0']=pd.to_datetime(R.t0,utc=True)
ini=R.groupby('actor').t0.transform('min')
R['dia']=(R.t0-ini).dt.total_seconds()/86400
fim=R.t0.max()
R['chance']=(fim-ini).dt.total_seconds()/86400
R=R[R.chance>=W2]
a=R[R.dia<W1]; b=R[(R.dia>=W1)&(R.dia<W2)]
g=a.groupby('actor')
A=pd.DataFrame({
    'leituras':g.size(),
    'dias_ativos':g.dia.apply(lambda s:s.astype(int).nunique()),
    'sessoes':g.sessao.nunique(),
    'dur_med':g.dur.median(),
    'ativo':g.ativo_ratio.median(),
})
A=A[A.leituras>=3]
A['freq']=A.dias_ativos/W1
A['volume']=A.leituras/A.sessoes.clip(lower=1)
A['prof']=np.log1p(A.dur_med)      # tempo por leitura = profundidade aqui
EIXOS=['freq','volume','prof']
nb=b.groupby('actor').size()
A['ret']=(A.index.map(nb).fillna(0)>0).astype(int)
print(f'{len(A):,} alunos com >=3 leituras nos dias 0-13 e 28 dias de chance')
print(f'  retencao (leu nos dias 14-27): {A.ret.mean():.1%}')
if len(A)<40 or A.ret.nunique()<2:
    print('  amostra insuficiente para medir AUC - registrado como tal.'); raise SystemExit
print('\n  distribuicao:')
for nome,col in [('dias ativos em 14','dias_ativos'),('leituras por sessao','volume'),
                 ('duracao mediana da leitura (s)','dur_med')]:
    q=A[col].quantile([.25,.5,.75])
    print(f'    {nome:<32} p25 {q[.25]:8.2f} | mediana {q[.5]:8.2f} | p75 {q[.75]:8.2f}')
print('\n  correlacao entre eixos:')
print('   '+A[EIXOS].rank().corr().round(2).to_string().replace('\n','\n   '))
print('\n  AUC contra retencao:')
for c in EIXOS+['leituras']:
    try: print(f'    {c:>10}: {roc_auc_score(A.ret,A[c]):.3f}  (n={len(A)})')
    except Exception as e: print(f'    {c:>10}: -')
A.to_parquet(f'{HERE}/ares_eixos.parquet')
