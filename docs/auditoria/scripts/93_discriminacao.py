"""discriminacao.confirmar: 25,0% de precisao e lift 7,1x, medidos numa
particao so. Reproduz o desenho original (tres particoes de alunos: p0 e p1
para marcar, p2 nunca vista para conferir) em 8 sementes."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0,'/Users/user/Documents/geis/core-trailup')
from trailup_core import discriminacao as DI
E='/Users/user/Downloads/Inicio/EdNet-KT3'
f=sorted(glob.glob(f'{E}/**/u*.csv',recursive=True))
rng=np.random.default_rng(0); sel=rng.choice(len(f),14000,replace=False)
L=[]
for i in sel:
    try: d=pd.read_csv(f[i],usecols=['item_id','user_answer'])
    except Exception: continue
    if len(d)<10: continue
    d['aluno']=os.path.basename(f[i]); L.append(d)
R=pd.concat(L,ignore_index=True)
q=pd.read_csv('/Users/user/Downloads/Inicio/EdNet-Contents/contents/questions.csv',usecols=['question_id','correct_answer'])
q['item_id']=q.question_id
R=R.merge(q,on='item_id'); R['ok']=(R.user_answer==R.correct_answer).astype(int)
R['hab']=R.groupby('aluno').ok.transform('mean')
print(f'respostas {len(R):,} | itens {R.item_id.nunique():,} | alunos {R.aluno.nunique():,}',flush=True)
alunos=R.aluno.unique()
import os as _os; MIN=int(_os.environ.get('MIN','30'))
def disc(d):
    """correlacao item-total por questao. Sem groupby.apply: as somas saem de
    colunas pre-computadas, senao o processo e morto por memoria."""
    t=d[['item_id']].copy()
    t['x']=d.ok.values.astype(float); t['y']=d.hab.values
    t['xy']=t.x*t.y; t['xx']=t.x*t.x; t['yy']=t.y*t.y
    a=t.groupby('item_id').agg(n=('x','size'),sx=('x','sum'),sy=('y','sum'),
                               sxy=('xy','sum'),sxx=('xx','sum'),syy=('yy','sum'))
    a=a[a.n>=MIN]
    num=a.n*a.sxy-a.sx*a.sy
    den=np.sqrt((a.n*a.sxx-a.sx**2).clip(lower=0)*(a.n*a.syy-a.sy**2).clip(lower=0))
    return (num/den.replace(0,np.nan))
print(f'\n{"semente":>8} {"itens":>7} {"marcados":>9} {"precisao":>9} {"base":>7} {"lift":>6}')
pr=[];li=[];mk=[];bs=[]
for s in [7,11,23,42,101,2026,314,999]:
    rg=np.random.default_rng(s); t=rg.integers(0,3,len(alunos))
    P=[R[R.aluno.isin(set(alunos[t==k]))] for k in (0,1,2)]
    d0,d1,d2=[disc(p) for p in P]
    j=pd.concat([d0.rename('a'),d1.rename('b'),d2.rename('v')],axis=1).dropna()
    if len(j)<200: continue
    marc=np.array([bool(DI.confirmar(a,b)) for a,b in zip(j.a,j.b)])
    ruim=(j.v<0).values; b=ruim.mean()
    if marc.sum()==0: print(f'{s:>8} {len(j):>7,} {0:>9}'); continue
    p=ruim[marc].mean(); pr.append(p); li.append(p/b); mk.append(marc.sum()); bs.append(b)
    print(f'{s:>8} {len(j):>7,} {marc.sum():>9} {p:>8.1%} {b:>6.1%} {p/b:>5.1f}x',flush=True)
pr,li=np.array(pr),np.array(li)
print(f'\n  precisao media {pr.mean():.1%} | desvio {pr.std(ddof=1):.1%} | afirmado 25,0% -> {abs(pr.mean()-0.250)/pr.std(ddof=1):.1f} desvios')
print(f'  lift     media {li.mean():.1f}x | desvio {li.std(ddof=1):.1f}x | afirmado 7,1x  -> {abs(li.mean()-7.1)/li.std(ddof=1):.1f} desvios')
print(f'  marcados media {np.mean(mk):.0f} questoes (afirmado 132)')
print(f'  BASE media {np.mean(bs):.2%} (o denominador do lift; afirmado 3,53%) | MIN={MIN}')
print(f'  MIN={MIN} | base media {BASE:.2%}' if False else '')
