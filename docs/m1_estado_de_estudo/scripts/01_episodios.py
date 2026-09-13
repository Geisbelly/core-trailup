"""ARES xAPI -> episodios de leitura com rotulo declarado pelo aluno.

Um "episodio" e a janela em que o aluno esteve em text.html para um assignment.
E o analogo mais proximo do lote do TrailUp: uma janela de atencao continua
sobre um conteudo, fechada por saida de tela.
"""
import os, json, re, numpy as np, pandas as pd

A='/Users/user/Downloads/Inicio/8y3zp-osfstorage-archive'
OUT=os.path.dirname(os.path.abspath(__file__))
GAP_SESSAO=1800     # s; acima disso e outra sessao
JANELA_RATING=900   # s; rating aceito ate 15 min depois do fim do episodio

x=pd.read_excel(f'{A}/data/preprocessed/xapi_statement_student.xlsx')
print(f'bruto: {len(x):,}')
x=x.drop_duplicates()
print(f'sem duplicatas: {len(x):,}')
# so alunos: codigos de 4 caracteres alfanumericos (Teacher Ares, Admin Kibi, Yi... fora)
x=x[x.actor.astype(str).str.fullmatch(r'[A-Za-z0-9]{4}')]
print(f'so alunos: {len(x):,} | {x.actor.nunique()} atores')

def ext(c):
    try: d=json.loads(c).get('extensions',{})
    except Exception: return {}
    return {k.rsplit('/',1)[-1]: v for k,v in d.items()}
E=x.context.map(ext)
for k in ['fragment','assignmentId','assignmentStatus','ratingrName','ratingValue',
          'dialogDescription','btnDescription','tokenId','answerCorrect','questionId']:
    x[k]=E.map(lambda d,k=k: d.get(k))
x['assignmentId']=pd.to_numeric(x.assignmentId, errors='coerce')
x['ts']=pd.to_datetime(x.timestamp, errors='coerce', utc=True)
x=x.dropna(subset=['ts']).sort_values(['actor','ts']).reset_index(drop=True)
print(f'com timestamp: {len(x):,} | janela {x.ts.min().date()} a {x.ts.max().date()}')

# sessoes por gap
dt=x.groupby('actor').ts.diff().dt.total_seconds()
x['sessao']=((dt.isna())|(dt>GAP_SESSAO)).groupby(x.actor).cumsum()

# episodios: blocos contiguos em text.html com o mesmo assignment, dentro da sessao
em_texto=(x.fragment=='text.html')&x.assignmentId.notna()
chave=x.actor.astype(str)+'|'+x.sessao.astype(str)+'|'+x.assignmentId.astype('Int64').astype(str)
x['ep']=((chave!=chave.shift())|(~em_texto)).cumsum().where(em_texto)

eps=[]
for ep, g in x[x.ep.notna()].groupby('ep', sort=False):
    if len(g)<3: continue
    t0,t1=g.ts.iloc[0],g.ts.iloc[-1]
    dur=(t1-t0).total_seconds()
    if dur<5 or dur>7200: continue
    gaps=g.ts.diff().dt.total_seconds().dropna()
    dlg=g.dialogDescription.astype(str)
    eps.append({
        'actor':g.actor.iloc[0],'sessao':int(g.sessao.iloc[0]),
        'assignment':int(g.assignmentId.iloc[0]),'t0':t0,'t1':t1,'dur':dur,
        'n_ev':len(g), 'ev_min':len(g)/(dur/60),
        'n_press':(g.verb=='pressed').sum(), 'n_focus':(g.object=='txtView').sum(),
        'n_token':(g.object=='token').sum() + dlg.str.contains('token popover',na=False).sum(),
        'n_explic':dlg.str.contains('Word explanation|Explain word',na=False,regex=True).sum(),
        'n_dialog':(g.object=='dialog').sum(),
        'gap_med':float(gaps.median()) if len(gaps) else 0.0,
        'gap_max':float(gaps.max()) if len(gaps) else 0.0,
        'gap_p90':float(gaps.quantile(.9)) if len(gaps) else 0.0,
        'frac_longa':float((gaps>30).mean()) if len(gaps) else 0.0,
    })
ep=pd.DataFrame(eps)
print(f'\nepisodios de leitura: {len(ep):,} | alunos {ep.actor.nunique()} | assignments {ep.assignment.nunique()}')
print(f'duracao: mediana {ep.dur.median():.0f}s, p90 {ep.dur.quantile(.9):.0f}s')

# --- rotulo 1: dificuldade/interesse declarados logo apos o episodio
r=x[x.ratingrName.notna()].copy()
r['val']=pd.to_numeric(r.ratingValue, errors='coerce')
r=r.dropna(subset=['val'])
print(f'\nratings: {len(r):,} | {r.ratingrName.value_counts().to_dict()}')
ep=ep.sort_values('t1')
for nome in ['difficulty','interestingness']:
    rr=r[r.ratingrName==nome].sort_values('ts')
    m=pd.merge_asof(ep[['t1','actor','assignment']].reset_index(), rr[['ts','actor','assignmentId','val']],
                    left_on='t1', right_on='ts', by='actor', direction='forward',
                    tolerance=pd.Timedelta(seconds=JANELA_RATING))
    m=m[(m.assignment==m.assignmentId)|m.assignmentId.isna()]
    ep[nome]=np.nan
    ep.loc[m['index'].values, nome]=m.val.values
    print(f'  {nome}: {ep[nome].notna().sum():,} episodios rotulados')

# --- rotulo 2: o assignment acabou abandonado?
st=x[x.assignmentStatus.notna()].sort_values('ts').groupby(['actor','assignmentId']).assignmentStatus.last()
ep['status']=list(pd.MultiIndex.from_arrays([ep.actor,ep.assignment.astype(float)]).map(st))
print(f'\nstatus final por (aluno, assignment): {ep.status.notna().sum():,} episodios')
print('  ',ep.status.value_counts().to_dict())
ep.to_parquet(f'{OUT}/episodios.parquet', index=False)
print(f'\n-> episodios.parquet {ep.shape}')
