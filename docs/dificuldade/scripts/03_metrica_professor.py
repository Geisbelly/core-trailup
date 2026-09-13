"""Metrica de dificuldade da turma PARA O PROFESSOR.

A pergunta nao e "prediz bem?" e sim "quando eu mostro, esta certo?".
Teste: estimo o desvio da turma na METADE A dos alunos e verifico se ele se
confirma na METADE B. Se nao replica, e ruido - e ruido exibido ao professor
vira decisao pedagogica errada.
"""
import os, numpy as np, pandas as pd
from scipy import stats
O='/caminho/para/os/datasets/open+university+learning+analytics+dataset'
a=pd.read_csv(f'{O}/assessments.csv'); sa=pd.read_csv(f'{O}/studentAssessment.csv')
sa['score']=pd.to_numeric(sa.score,errors='coerce'); sa=sa.dropna(subset=['score'])
a=a.sort_values(['code_module','code_presentation','date'])
a['ordem']=a.groupby(['code_module','code_presentation']).cumcount()+1
a['slot']=a.code_module+'|'+a.assessment_type+'|'+a.ordem.astype(str)
d=sa.merge(a[['id_assessment','code_module','code_presentation','slot']],on='id_assessment')
d['turma']=d.code_module+'-'+d.code_presentation
d['y']=(d.score>=80).astype(int); G=d.y.mean()
g=d.groupby(['slot','turma']).y.agg(['sum','size']).reset_index()
tot=g.groupby('slot')[['sum','size']].sum()
g['os']=g.slot.map(tot['sum'])-g['sum']; g['on']=g.slot.map(tot['size'])-g['size']
GLOB={(r.slot,r.turma):((r.os+G*20)/(r.on+20) if r.on>0 else G) for r in g.itertuples()}

rng=np.random.RandomState(20260912); L=[]
for (turma,slot),gr in d.groupby(['turma','slot']):
    if len(gr)<20: continue
    yy=gr.y.sample(frac=1,random_state=rng.randint(1<<30)).values
    h=len(yy)//2; A,B=yy[:h],yy[h:]
    glob=GLOB.get((slot,turma),G)
    se=np.sqrt(max(glob*(1-glob),1e-6)/len(A))
    L.append({'turma':turma,'slot':slot,'nA':len(A),'nB':len(B),
              'devA':A.mean()-glob,'devB':B.mean()-glob,'glob':glob,
              'z':(A.mean()-glob)/se if se>0 else 0.0})
L=pd.DataFrame(L)
print(f'{len(L)} pares (turma, avaliacao) com >=20 alunos\n')
print('=== o desvio da turma medido na metade A se confirma na metade B? ===')
for lo,hi,lab in [(10,24,'10-24'),(25,49,'25-49'),(50,199,'50-199'),(200,10**9,'200+')]:
    m=(L.nA>=lo)&(L.nA<=hi)
    if m.sum()<8: continue
    r=stats.pearsonr(L.devA[m],L.devB[m])
    conc=((L.devA[m]>0)==(L.devB[m]>0)).mean()
    print(f'  metade A com {lab:>7} alunos | {m.sum():>3} pares | correlacao A-B {r.statistic:+.3f} '
          f'(p={r.pvalue:.3f}) | mesmo sinal {conc:.0%}')
print('\n=== e se so exibir quando o desvio for estatisticamente forte? ===')
print(f"{'regra':38s} {'exibe em':>9} {'mesmo sinal em B':>17} {'|desvio| medio B':>17}")
print('-'*86)
for lab,m in [('sempre',np.ones(len(L),bool)),
              ('|z| >= 1.0',(L.z.abs()>=1.0).values),
              ('|z| >= 1.5',(L.z.abs()>=1.5).values),
              ('|z| >= 2.0',(L.z.abs()>=2.0).values),
              ('|z| >= 2.0 e >=20 alunos',((L.z.abs()>=2.0)&(L.nA>=20)).values),
              ('|z| >= 2.5 e >=20 alunos',((L.z.abs()>=2.5)&(L.nA>=20)).values)]:
    if m.sum()<5: continue
    conc=((L.devA[m]>0)==(L.devB[m]>0)).mean()
    print(f'{lab:38s} {m.mean():>8.0%} {conc:>16.0%} {L.devB[m].abs().mean():>17.3f}')
print('\n=== percentil dentro da turma (o ranking que o professor ve) ===')
mt=d.groupby(['turma','slot']).y.agg(['mean','size']).reset_index()
mt=mt[mt['size']>=20]
mt['pct']=mt.groupby('turma')['mean'].rank(pct=True)
print(f'  {len(mt)} avaliacoes com >=20 alunos, em {mt.turma.nunique()} turmas')
print(f'  avaliacoes por turma: mediana {mt.groupby("turma").size().median():.0f}')
ex=mt[mt.turma==mt.turma.value_counts().index[0]].sort_values('pct')
print(f'\n  exemplo - turma {ex.turma.iloc[0]}:')
for r in ex.head(3).itertuples(): print(f'    {r.slot:20s} acerto {r.mean:.0%} | percentil {r.pct:.0%} (mais dificil)')
for r in ex.tail(2).itertuples(): print(f'    {r.slot:20s} acerto {r.mean:.0%} | percentil {r.pct:.0%}')
