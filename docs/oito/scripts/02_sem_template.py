"""Teste honesto: prever a ferramenta usando SO os turnos seguintes ao primeiro.

O primeiro turno carrega o template da ferramenta (62% comecam com o prefixo
dominante). Os turnos seguintes o aluno escreve livre - e neles que se ve se a
INTENCAO e recuperavel, ou se o modelo so estava lendo o template.
"""
import ast, glob, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
B='/Users/user/Downloads/Inicio/19232684'
L=[]
for f in sorted(glob.glob(f'{B}/SB_LogData_*.xlsx')):
    d=pd.read_excel(f)
    for r in d.itertuples():
        try: msgs=ast.literal_eval(r.messages)
        except Exception: continue
        u=[m.get('content','') for m in msgs if m.get('role')=='user']
        if len(u)<2: continue
        L.append({'user':r.user_id,'tool':r.tool,'seguintes':' '.join(u[1:]),'n':len(u)})
D=pd.DataFrame(L)
keep=D[D.tool.isin(D.tool.value_counts()[lambda s:s>=12].index)]
print(f'{len(D)} sessoes com >=2 turnos | usando {len(keep)} nas ferramentas com n>=12')
print(f'  {keep.tool.value_counts().to_dict()}')
maioria=keep.tool.value_counts(normalize=True).iloc[0]
m=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=2,sublinear_tf=True),
                LogisticRegression(max_iter=2000,class_weight='balanced'))
p=cross_val_predict(m,keep.seguintes,keep.tool,groups=keep.user,cv=GroupKFold(4))
acc=accuracy_score(keep.tool,p)
print(f'\n=== so os turnos SEGUINTES (sem template) ===')
print(f'  acuracia {acc:.0%} | chutar a maioria {maioria:.0%} | ganho {acc-maioria:+.0%}')
cm=pd.DataFrame(confusion_matrix(keep.tool,p,labels=sorted(keep.tool.unique())),
                index=sorted(keep.tool.unique()),columns=sorted(keep.tool.unique()))
print('\n',cm.to_string())
print(f'\n  -> {"a intencao e recuperavel do texto livre" if acc-maioria>0.15 else "sem o template, quase nao sobra sinal"}')
