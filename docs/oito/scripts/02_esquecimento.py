"""(4) Curva de esquecimento: quanto o aluno perde em funcao do intervalo.

Usa os reencontros da MESMA questao pelo MESMO aluno (11,6% das respostas).
Alimenta revisao espacada - decisao de trilha, nao de estado.
"""
import os, numpy as np, pandas as pd
M2='/caminho/para/os/intermediarios'
d=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','ts']).sort_values('ts')
d['k']=d.groupby(['user','qidx']).cumcount()
pri=d[d.k==0].set_index(['user','qidx'])
seg=d[d.k==1].set_index(['user','qidx'])
j=pri[['correct','ts']].join(seg[['correct','ts']],rsuffix='_2',how='inner')
j['dias']=(j.ts_2-j.ts)/1000/86400
j=j[(j.dias>=0)&(j.dias<180)]
print(f'{len(j):,} reencontros | intervalo: mediana {j.dias.median():.1f} dias, p90 {j.dias.quantile(.9):.0f}\n')
print('=== retencao em funcao do intervalo ===')
print(f"{'intervalo':>16} {'n':>8} {'acertou antes':>14} {'errou antes':>12}")
print('-'*56)
faixas=[(0,1/24,'< 1 hora'),(1/24,1,'1h - 1 dia'),(1,3,'1 - 3 dias'),(3,7,'3 - 7 dias'),
        (7,21,'1 - 3 semanas'),(21,60,'3 sem - 2 meses'),(60,180,'2 - 6 meses')]
for lo,hi,lab in faixas:
    m=(j.dias>=lo)&(j.dias<hi)
    if m.sum()<200: continue
    a=j[m&(j.correct==1)].correct_2.mean(); b=j[m&(j.correct==0)].correct_2.mean()
    print(f'{lab:>16} {m.sum():>8,} {a:>13.1%} {b:>12.1%}')
print('\n=== a queda e real? (controlando dificuldade da questao) ===')
q=d.groupby('qidx').correct.mean()
j['qd']=j.index.get_level_values('qidx').map(q)
ac=j[j.correct==1].copy()
ac['bin']=pd.cut(ac.dias,[0,1,7,30,180],labels=['<1d','1-7d','7-30d','30d+'])
t=ac.groupby('bin',observed=True).agg(n=('correct_2','size'),ret=('correct_2','mean'),dif=('qd','mean'))
print(t.round(3).to_string())
print('\n  se `dif` for parecido entre as faixas, a queda e do TEMPO, nao da questao')
