"""Constroi a tendencia nas duas bases e mede quanto ela adiciona."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
M2=HERE.replace('/engaj','/m2'); DIA=86400*1000.0
O_DIR='/caminho/para/os/datasets/open+university+learning+analytics+dataset'
W1=30

# ---------- EdNet ----------
E=pd.read_parquet(f'{HERE}/eixos.parquet')
d=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','ts']).sort_values(['user','ts'])
ini=d.groupby('user').ts.transform('min'); d['dia']=(d.ts-ini)/DIA
d=d[(d.dia<W1)&(d.user.isin(E.index))]
t1=d[d.dia<10].groupby('user').size(); t3=d[d.dia>=20].groupby('user').size()
E['tend']=np.log1p(E.index.map(t3).fillna(0))-np.log1p(E.index.map(t1).fillna(0))
E['dias_recentes']=d[d.dia>=20].groupby('user').dia.apply(lambda s:s.astype(int).nunique()).reindex(E.index).fillna(0)/10

# ---------- OULAD ----------
O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet')
v=pd.read_csv(f'{O_DIR}/studentVle.csv',usecols=['code_module','code_presentation','id_student','date','sum_click'])
v['aluno']=v.code_module+'-'+v.code_presentation+'|'+v.id_student.astype(str)
v=v[(v.date>=0)&(v.date<W1)&(v.aluno.isin(O.index))]
o1=v[v.date<10].groupby('aluno').sum_click.sum(); o3=v[v.date>=20].groupby('aluno').sum_click.sum()
O['tend']=np.log1p(O.index.map(o3).fillna(0))-np.log1p(O.index.map(o1).fillna(0))
O['dias_recentes']=v[v.date>=20].groupby('aluno').date.nunique().reindex(O.index).fillna(0)/10

EIXOS=['freq','volume','prof']
for nome,D in [('EdNet',E),('OULAD',O)]:
    print(f'\n=== {nome} (n={len(D):,}, retencao {D.ret.mean():.1%}) ===')
    print(f'  {"medida":<22} {"AUC":>7}')
    for c in EIXOS+['tend','dias_recentes']:
        print(f'  {c:<22} {roc_auc_score(D.ret,D[c]):>7.3f}')
    rng=np.random.RandomState(4); m=rng.rand(len(D))<0.7
    tr,te=D[m],D[~m]
    def auc(cols):
        X=tr[cols].replace([np.inf,-np.inf],0).fillna(0); Xe=te[cols].replace([np.inf,-np.inf],0).fillna(0)
        mu,sd=X.mean(),X.std().replace(0,1)
        w=logistic_fit(((X-mu)/sd).values,tr.ret.values)
        return roc_auc_score(te.ret.values,logistic_pred(w,((Xe-mu)/sd).values))
    print(f'\n  {"modelo":<34} {"AUC fora do treino":>19}')
    for lab,cols in [('so frequencia',['freq']),
                     ('os tres eixos',EIXOS),
                     ('so dias_recentes',['dias_recentes']),
                     ('frequencia + tendencia',['freq','tend']),
                     ('frequencia + dias_recentes',['freq','dias_recentes']),
                     ('tres eixos + tendencia',EIXOS+['tend']),
                     ('tres eixos + tend + dias_rec',EIXOS+['tend','dias_recentes'])]:
        print(f'  {lab:<34} {auc(cols):>19.3f}')
E.to_parquet(f'{HERE}/eixos.parquet'); O.to_parquet(f'{HERE}/oulad_eixos.parquet')
