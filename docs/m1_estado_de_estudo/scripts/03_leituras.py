"""Refaz na unidade certa: a LEITURA = (aluno, assignment, sessao).

O corte anterior quebrava o episodio a cada modal de explicacao de palavra,
gerando janelas de 25s que nao sao leitura nenhuma.
"""
import os, json, numpy as np, pandas as pd
A='/caminho/para/os/datasets/8y3zp-osfstorage-archive'
OUT=os.path.dirname(os.path.abspath(__file__))
GAP=1800

x=pd.read_excel(f'{A}/data/preprocessed/xapi_statement_student.xlsx').drop_duplicates()
x=x[x.actor.astype(str).str.fullmatch(r'[A-Za-z0-9]{4}')]
def ext(c):
    try: d=json.loads(c).get('extensions',{})
    except Exception: return {}
    return {k.rsplit('/',1)[-1]:v for k,v in d.items()}
E=x.context.map(ext)
for k in ['fragment','assignmentId','assignmentStatus','ratingrName','ratingValue','dialogDescription']:
    x[k]=E.map(lambda d,k=k: d.get(k))
x['assignmentId']=pd.to_numeric(x.assignmentId,errors='coerce')
x['ts']=pd.to_datetime(x.timestamp,errors='coerce',utc=True)
x=x.dropna(subset=['ts']).sort_values(['actor','ts']).reset_index(drop=True)
dt=x.groupby('actor').ts.diff().dt.total_seconds()
x['sessao']=((dt.isna())|(dt>GAP)).groupby(x.actor).cumsum()
print('fragmento durante modal de explicacao:',
      x[x.dialogDescription.astype(str).str.contains('Word explanation|Explain word',na=False)].fragment.value_counts().head(3).to_dict())

# preenche o assignment corrente dentro da sessao (modais nao carregam o id)
x['assign_ff']=x.groupby(['actor','sessao']).assignmentId.ffill()
leitura=(x.fragment=='text.html')|(x.dialogDescription.astype(str).str.contains('Word explanation|Explain word|token popover',na=False))
L=x[leitura & x.assign_ff.notna()].copy()

recs=[]
for (a,s,asg),g in L.groupby(['actor','sessao','assign_ff'],sort=False):
    if len(g)<5: continue
    dur=(g.ts.iloc[-1]-g.ts.iloc[0]).total_seconds()
    if dur<30 or dur>10800: continue
    gaps=g.ts.diff().dt.total_seconds().dropna()
    ativo=float(gaps[gaps<=30].sum())          # tempo "ativo": soma dos gaps curtos
    dlg=g.dialogDescription.astype(str)
    recs.append({'actor':a,'sessao':int(s),'assignment':int(asg),'t0':g.ts.iloc[0],'t1':g.ts.iloc[-1],
        'dur':dur,'ativo_sec':ativo,'ativo_ratio':ativo/dur if dur else 0,
        'n_ev':len(g),'ev_min':len(g)/(dur/60),
        'n_press':int((g.verb=='pressed').sum()),'n_focus':int((g.object=='txtView').sum()),
        'n_token':int((g.object=='token').sum()+dlg.str.contains('token popover',na=False).sum()),
        'n_explic':int(dlg.str.contains('Word explanation|Explain word',na=False).sum()),
        'n_dialog':int((g.object=='dialog').sum()),
        'gap_med':float(gaps.median()) if len(gaps) else 0,'gap_p90':float(gaps.quantile(.9)) if len(gaps) else 0,
        'gap_max':float(gaps.max()) if len(gaps) else 0,
        'frac_pausa':float((gaps>30).mean()) if len(gaps) else 0})
R=pd.DataFrame(recs)
print(f'\nleituras: {len(R):,} | alunos {R.actor.nunique()} | assignments {R.assignment.nunique()}')
print(f'duracao: mediana {R.dur.median():.0f}s p90 {R.dur.quantile(.9):.0f}s | ativo_ratio mediano {R.ativo_ratio.median():.2f}')

# revisita: quantas vezes o aluno ja leu este assignment antes
R=R.sort_values(['actor','assignment','t0'])
R['revisita']=R.groupby(['actor','assignment']).cumcount()

# rotulo: rating do MESMO assignment dado apos o fim da leitura (ate 30 min)
r=x[x.ratingrName.notna()].copy(); r['val']=pd.to_numeric(r.ratingValue,errors='coerce')
r['asg']=r.assign_ff
r=r.dropna(subset=['val','asg'])
for nome in ['difficulty','interestingness']:
    rr=r[r.ratingrName==nome]
    vals=[]
    idx={(a,int(g)):v for (a,g),v in rr.groupby(['actor','asg']).val.last().items()}
    for a,asg,t1 in zip(R.actor,R.assignment,R.t1):
        sub=rr[(rr.actor==a)&(rr.asg==asg)&(rr.ts>=t1)&(rr.ts<=t1+pd.Timedelta(minutes=30))]
        vals.append(sub.val.iloc[0] if len(sub) else np.nan)
    R[nome]=vals
    print(f'  {nome}: {R[nome].notna().sum():,} leituras rotuladas')
st=x[x.assignmentStatus.notna()].sort_values('ts').groupby(['actor','assign_ff']).assignmentStatus.last()
R['status']=list(pd.MultiIndex.from_arrays([R.actor,R.assignment.astype(float)]).map(st))
R.to_parquet(f'{OUT}/leituras.parquet',index=False)
print(f'\n-> leituras.parquet {R.shape} | status: {R.status.value_counts().to_dict()}')
