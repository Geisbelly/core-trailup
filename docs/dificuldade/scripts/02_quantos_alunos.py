"""Quantos alunos da turma precisam responder antes de a dificuldade DA TURMA
valer mais que a dificuldade global?

Para cada (turma, avaliacao), embaralha os alunos e caminha: o aluno na posicao
k e previsto usando apenas os k-1 anteriores DA MESMA TURMA. O global vem das
OUTRAS turmas. E o cenario real: a turma vai revelando a dificuldade conforme
responde.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
a=pd.read_csv(f'{O}/assessments.csv'); sa=pd.read_csv(f'{O}/studentAssessment.csv')
sa['score']=pd.to_numeric(sa.score,errors='coerce'); sa=sa.dropna(subset=['score'])
a=a.sort_values(['code_module','code_presentation','date'])
a['ordem']=a.groupby(['code_module','code_presentation']).cumcount()+1
a['slot']=a.code_module+'|'+a.assessment_type+'|'+a.ordem.astype(str)
d=sa.merge(a[['id_assessment','code_module','code_presentation','slot','date']],on='id_assessment')
d['turma']=d.code_module+'-'+d.code_presentation
d['y']=(d.score>=80).astype(int)
G=d.y.mean()
print(f'{len(d):,} notas | {d.turma.nunique()} turmas | {d.slot.nunique()} avaliacoes | positivos {G:.1%}')

# global por slot, deixando a propria turma de fora (leave-one-class-out)
g=d.groupby(['slot','turma']).y.agg(['sum','size']).reset_index()
tot=g.groupby('slot')[['sum','size']].sum()
g['out_s']=g.slot.map(tot['sum'])-g['sum']; g['out_n']=g.slot.map(tot['size'])-g['size']
GLOB={(r.slot,r.turma):((r.out_s+G*20)/(r.out_n+20) if r.out_n>0 else G) for r in g.itertuples()}

rng=np.random.RandomState(20260912)
linhas=[]
for (turma,slot),gr in d.groupby(['turma','slot']):
    if len(gr)<40: continue
    gr=gr.sample(frac=1,random_state=rng.randint(1<<30))  # ordem de chegada aleatoria
    yy=gr.y.values; glob=GLOB.get((slot,turma),G)
    cs=0
    for k,v in enumerate(yy):
        if k>=5:   # so avalia quando ja ha pelo menos 5 alunos antes
            turma_p=cs/k
            for prior in (5,15,40):
                pass
            linhas.append((turma,slot,k,v,glob,turma_p,
                           (cs+glob*5)/(k+5),(cs+glob*15)/(k+15),(cs+glob*40)/(k+40)))
        cs+=v
L=pd.DataFrame(linhas,columns=['turma','slot','k','y','glob','turma_cru','hier5','hier15','hier40'])
print(f'{len(L):,} previsoes avaliadas\n')
def av(nome,p,m=None):
    s=L if m is None else L[m]
    pp=np.clip(p if m is None else p[m],1e-6,1-1e-6)
    return {'estimador':nome,'AUC':roc_auc_score(s.y,pp),'logloss':log_loss(s.y,pp),'brier':brier_score_loss(s.y,pp)}
print('=== geral ===')
print(pd.DataFrame([av('dificuldade GLOBAL (outras turmas)',L.glob.values),
                    av('dificuldade DA TURMA (cru)',L.turma_cru.values),
                    av('hierarquico prior=5',L.hier5.values),
                    av('hierarquico prior=15',L.hier15.values),
                    av('hierarquico prior=40',L.hier40.values)]).set_index('estimador').round(4).to_string())
print('\n=== por quantos alunos da turma ja responderam ===')
print(f"{'alunos antes':>13} {'n':>8} {'GLOBAL':>8} {'TURMA cru':>10} {'hier15':>8}")
print('-'*52)
for lo,hi,lab in [(5,9,'5-9'),(10,19,'10-19'),(20,49,'20-49'),(50,99,'50-99'),(100,10**9,'100+')]:
    m=((L.k>=lo)&(L.k<=hi)).values
    if m.sum()<300: continue
    print(f'{lab:>13} {m.sum():>8,} {roc_auc_score(L.y[m],L.glob[m]):>8.4f} '
          f'{roc_auc_score(L.y[m],L.turma_cru[m]):>10.4f} {roc_auc_score(L.y[m],L.hier15[m]):>8.4f}')
print('\n(AUC aqui e baixo de proposito: e so a dificuldade, sem nada do aluno.')
print(' O que interessa e a COMPARACAO entre as colunas.)')
