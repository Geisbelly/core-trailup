"""(7) Roteador de intencao. Primeiro: o rotulo e recuperavel por template?"""
import ast, glob, re, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import classification_report, accuracy_score
B='/caminho/para/os/datasets/19232684'
L=[]
for f in sorted(glob.glob(f'{B}/SB_LogData_*.xlsx')):
    d=pd.read_excel(f); d['pais']=f.split('_')[-2]
    for r in d.itertuples():
        try: msgs=ast.literal_eval(r.messages)
        except Exception: continue
        u=[m.get('content','') for m in msgs if m.get('role')=='user']
        if not u: continue
        L.append({'pais':r.pais,'user':r.user_id,'tool':r.tool,'primeira':u[0],
                  'n_turnos':len(u),'todas':' '.join(u)})
D=pd.DataFrame(L)
print(f'{len(D)} sessoes | {D.user.nunique()} alunos | ferramentas: {D.tool.nunique()}')
print(D.tool.value_counts().to_dict())

print('\n=== o rotulo vem de um TEMPLATE na mensagem? ===')
for t,g in D[D.tool!='none'].groupby('tool'):
    if len(g)<5: continue
    pref=Counter(x[:45] for x in g.primeira)
    top,c=pref.most_common(1)[0]
    print(f'  {t:>30} n={len(g):>3} | prefixo mais comum em {c/len(g):>4.0%}: {top[:45]!r}')
comum=sum(Counter(x[:45] for x in g.primeira).most_common(1)[0][1]
          for t,g in D[D.tool!='none'].groupby('tool') if len(g)>=5)
tot=sum(len(g) for t,g in D[D.tool!='none'].groupby('tool') if len(g)>=5)
print(f'\n  -> {comum/tot:.0%} das sessoes com ferramenta comecam com o prefixo dominante da sua ferramenta')
print('     ou seja: o rotulo e em boa parte o TEMPLATE, nao a intencao do aluno')

print('\n=== e se tirar o template? (usar so o que o aluno escreveu depois) ===')
D['sem_template']=[re.sub(r'^.{0,120}?(?:θέμα|μάθημα|topic|about)\s*','',x,flags=re.I)[:400] for x in D.primeira]
keep=D[D.tool.isin(D.tool.value_counts()[lambda s:s>=15].index)].copy()
print(f'  {len(keep)} sessoes nas {keep.tool.nunique()} ferramentas com n>=15: {keep.tool.value_counts().to_dict()}')
for col,lab in [('primeira','mensagem inteira (com template)'),('sem_template','sem o prefixo')]:
    m=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True),
                    LogisticRegression(max_iter=2000,class_weight='balanced'))
    p=cross_val_predict(m,keep[col],keep.tool,groups=keep.user,cv=GroupKFold(4))
    print(f'  {lab:>34}: acuracia {accuracy_score(keep.tool,p):.0%} '
          f'(maioria = {keep.tool.value_counts(normalize=True).iloc[0]:.0%})')
