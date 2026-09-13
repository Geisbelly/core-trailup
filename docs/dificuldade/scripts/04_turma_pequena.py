"""Turma de escola tem ~30 alunos, nao 200. A metrica sobrevive?

Subamostra cada turma do OULAD para n alunos e refaz o teste de replicacao,
repetindo varias vezes para nao depender de um sorteio.
"""
import os, numpy as np, pandas as pd
from scipy import stats
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
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
# verdade de referencia: o desvio medido com TODOS os alunos
VERD={(t,s):gr.y.mean()-GLOB.get((s,t),G) for (t,s),gr in d.groupby(['turma','slot']) if len(gr)>=100}
print(f'{len(VERD)} pares (turma, avaliacao) com >=100 alunos servem de referencia\n')
print(f"{'alunos':>7} │ {'exibe':>6} {'acerta direcao':>15} {'corr c/ verdade':>16} │ {'sem regra:':>11} {'direcao':>8}")
print('─'*76)
rng=np.random.RandomState(20260912)
for N in (15,20,30,40,60,100):
    disp=[]; ac=[]; est=[]; ver=[]; ac0=[]
    for rep in range(30):
        for (t,s),gr in d.groupby(['turma','slot']):
            if (t,s) not in VERD or len(gr)<N: continue
            yy=gr.y.sample(N,random_state=rng.randint(1<<30)).values
            glob=GLOB.get((s,t),G); dev=yy.mean()-glob
            se=np.sqrt(max(glob*(1-glob),1e-6)/N); z=dev/se if se>0 else 0
            v=VERD[(t,s)]
            ac0.append((dev>0)==(v>0)); est.append(dev); ver.append(v)
            if abs(z)>=2.0:
                disp.append(1); ac.append((dev>0)==(v>0))
            else: disp.append(0)
    c=stats.pearsonr(est,ver).statistic
    print(f'{N:>7} │ {np.mean(disp):>5.0%} {np.mean(ac) if ac else float("nan"):>14.0%} {c:>16.2f} │ '
          f'{"100%":>11} {np.mean(ac0):>7.0%}')
print('\n  "exibe" = fracao dos pares em que |z|>=2 dispara')
print('  "acerta direcao" = entre os exibidos, quantos apontam o mesmo lado que a verdade')
print('  "sem regra" = exibir sempre, e a taxa de acerto de direcao correspondente')
