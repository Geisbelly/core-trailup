"""Se o comportamento nao explica a dificuldade declarada, o que explica?

Duas hipoteses concorrentes:
  (a) e do TEXTO   - uns textos sao dificeis para todo mundo
  (b) e do ALUNO   - uns alunos acham tudo dificil
Ambas sao estimaveis SEM inferir estado emocional de ninguem.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
d=R[R.difficulty.notna()].copy(); y=(d.difficulty>=4).astype(int).values
d['y']=y
print(f'{len(d):,} leituras | {d.actor.nunique()} alunos | {d.assignment.nunique()} textos | positivos {y.mean():.1%}')

# (a) media do TEXTO, estimada sem o proprio aluno (leave-one-student-out)
soma_t=d.groupby('assignment').y.sum(); n_t=d.groupby('assignment').y.size()
g=y.mean()
p_texto=[(soma_t[a]-yy+g*10)/(n_t[a]-1+10) for a,yy in zip(d.assignment,d.y)]
# (b) media do ALUNO, estimada sem a propria leitura
soma_a=d.groupby('actor').y.sum(); n_a=d.groupby('actor').y.size()
p_aluno=[(soma_a[a]-yy+g*5)/(n_a[a]-1+5) for a,yy in zip(d.actor,d.y)]
d['p_texto']=p_texto; d['p_aluno']=p_aluno

print('\n=== quem prediz a dificuldade declarada? (leave-one-out) ===')
for c,lab in [('p_texto','dificuldade media do TEXTO (outros alunos)'),
              ('p_aluno','severidade media do ALUNO (outras leituras)')]:
    print(f'  {lab:46s} AUC {roc_auc_score(y,d[c]):.4f}')
comb=0.5*(d.p_texto+d.p_aluno)
print(f'  {"os dois combinados":46s} AUC {roc_auc_score(y,comb):.4f}')
print(f'  {"comportamento (melhor modelo da tabela anterior)":46s} AUC ~0.53')

print('\n=== variancia: quanto cada nivel explica ===')
print(f'  desvio entre TEXTOS  (media por texto):  {d.groupby("assignment").y.mean().std():.3f}')
print(f'  desvio entre ALUNOS  (media por aluno):  {d.groupby("actor").y.mean().std():.3f}')
print(f'  desvio total das leituras:               {d.y.std():.3f}')

print('\n=== os textos se separam? ===')
t=d.groupby('assignment').agg(n=('y','size'),dificil=('y','mean')).query('n>=20').sort_values('dificil')
print(f'  {len(t)} textos com >=20 leituras | dificil em {t.dificil.min():.0%} a {t.dificil.max():.0%} das leituras')
print('  mais faceis:',t.head(3).dificil.round(2).to_dict(),'| mais dificeis:',t.tail(3).dificil.round(2).to_dict())
