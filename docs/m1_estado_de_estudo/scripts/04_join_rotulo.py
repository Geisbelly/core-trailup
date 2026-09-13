"""Atribui a nota a leitura pela posicao temporal: a estrela e clicada em
text.html sem assignmentId, entao vale o assignment corrente na sessao.
Arrastar a estrela emite varios eventos - vale o ULTIMO valor.
"""
import os, json, numpy as np, pandas as pd
A='/Users/user/Downloads/Inicio/8y3zp-osfstorage-archive'
HERE=os.path.dirname(os.path.abspath(__file__))
x=pd.read_excel(f'{A}/data/preprocessed/xapi_statement_student.xlsx').drop_duplicates()
x=x[x.actor.astype(str).str.fullmatch(r'[A-Za-z0-9]{4}')]
def ext(c):
    try: d=json.loads(c).get('extensions',{})
    except Exception: return {}
    return {k.rsplit('/',1)[-1]:v for k,v in d.items()}
E=x.context.map(ext)
for k in ['fragment','assignmentId','ratingrName','ratingValue']:
    x[k]=E.map(lambda d,k=k: d.get(k))
x['assignmentId']=pd.to_numeric(x.assignmentId,errors='coerce')
x['ts']=pd.to_datetime(x.timestamp,errors='coerce',utc=True)
x=x.dropna(subset=['ts']).sort_values(['actor','ts']).reset_index(drop=True)
dt=x.groupby('actor').ts.diff().dt.total_seconds()
x['sessao']=((dt.isna())|(dt>1800)).groupby(x.actor).cumsum()
x['asg']=x.groupby(['actor','sessao']).assignmentId.ffill()

r=x[x.ratingrName.notna()].copy()
r['val']=pd.to_numeric(r.ratingValue,errors='coerce')
print(f'eventos de estrela: {len(r):,} | com assignment resolvido: {r.asg.notna().mean():.1%}')
r=r.dropna(subset=['val','asg'])
fim=r.sort_values('ts').groupby(['actor','sessao','asg','ratingrName']).val.last().unstack()
fim.index.names=['actor','sessao','asg']
print(f'notas consolidadas: {len(fim):,} pares (aluno, sessao, texto)')
print(fim.describe().round(2).to_string())

R=pd.read_parquet(f'{HERE}/leituras.parquet')
R=R.merge(fim.reset_index().rename(columns={'asg':'assignment'}),
          on=['actor','sessao','assignment'], how='left', suffixes=('_old',''))
for c in ['difficulty','interestingness']:
    if c+'_old' in R.columns: R=R.drop(columns=[c+'_old'])
print(f'\nleituras: {len(R):,} | com dificuldade: {R.difficulty.notna().sum():,} | com interesse: {R.interestingness.notna().sum():,}')
print('distribuicao da dificuldade declarada:', R.difficulty.value_counts().sort_index().to_dict())
R.to_parquet(f'{HERE}/leituras_rotuladas.parquet',index=False)
print('-> leituras_rotuladas.parquet')
