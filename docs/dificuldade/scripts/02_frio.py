"""Dá para prever a faixa de dificuldade ANTES de qualquer aluno responder?

E o caso do TrailUp: professor cadastra a questao, banco vazio, zero respostas.
Se metadado prediz, a questao ja nasce classificada. Se nao, so resta coletar.
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix
HERE=os.path.dirname(os.path.abspath(__file__))
BASE='/caminho/para/os/datasets'
q=pd.read_parquet(f'{HERE}/questoes.parquet').reset_index()
meta=pd.read_csv(f'{BASE}/EdNet-Contents/contents/questions.csv')
meta['qidx']=meta.question_id.str[1:].astype(np.int32)
meta['n_tags']=meta.tags.astype(str).apply(lambda t:0 if t=='-1' else t.count(';')+1)
bsz=meta.groupby('bundle_id').size().rename('tam_bundle')
meta=meta.join(bsz,on='bundle_id')
meta['pos_bundle']=meta.groupby('bundle_id').cumcount()+1
meta['tem_expl']=meta.explanation_id.notna().astype(int)
meta['deploy']=pd.to_numeric(meta.deployed_at,errors='coerce')
d=q.merge(meta[['qidx','part','n_tags','tam_bundle','pos_bundle','tem_expl','deploy','tags','bundle_id']],on='qidx')
print(f'{len(d):,} questoes com verdade e metadado')
lo,hi=0.50,0.80
NOMES=['dificil','media','facil']
y=d.fx_verdade.values
print('faixas reais:',{NOMES[i]:f'{(y==i).mean():.0%}' for i in range(3)})

# tags como multi-hot esparso (as 60 mais comuns)
from collections import Counter
cnt=Counter(t for s in d.tags.astype(str) for t in s.split(';') if t not in ('-1',''))
TOP=[t for t,_ in cnt.most_common(60)]
for t in TOP: d[f'tag_{t}']=d.tags.astype(str).str.split(';').apply(lambda L,t=t: int(t in L))
TAGC=[f'tag_{t}' for t in TOP]
F=['part','n_tags','tam_bundle','pos_bundle','tem_expl','deploy']
# split por BUNDLE: questoes do mesmo bundle sao quase a mesma questao
g=d.bundle_id.values
def roda(F,nome):
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_depth=4,
        min_samples_leaf=30,random_state=0,categorical_features=[F.index('part')] if 'part' in F else None)
    p=cross_val_predict(m,d[F],y,groups=g,cv=GroupKFold(5))
    return {'features':nome,'n':len(F),'acerto':accuracy_score(y,p),
            'erra_2_faixas':float((np.abs(p-y)==2).mean())}
base=max((y==i).mean() for i in range(3))
rows=[{'features':'chutar a faixa mais comum','n':0,'acerto':base,'erra_2_faixas':np.nan}]
rows.append(roda(F,'metadado basico (part, tags, bundle)'))
rows.append(roda(F+TAGC,'+ 60 tags de assunto'))
print('\n=== prever a faixa SEM nenhuma resposta (validacao por bundle) ===')
print(pd.DataFrame(rows).set_index('features').round(4).to_string())

m=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.06,max_depth=4,
    min_samples_leaf=30,random_state=0,categorical_features=[0])
p=cross_val_predict(m,d[F+TAGC],y,groups=g,cv=GroupKFold(5))
print('\nmatriz de confusao (linha = real, coluna = previsto):')
cm=pd.DataFrame(confusion_matrix(y,p),index=[f'real {n}' for n in NOMES],columns=[f'prev {n}' for n in NOMES])
print(cm.to_string())
print('\n=== quanto vale o metadado contra 1 resposta observada? ===')
G=0.668
for N in (0,5,10,21,30):
    if N==0:
        print(f'  sem resposta, so metadado : {accuracy_score(y,p):.0%}')
    else:
        rng=np.random.RandomState(7)
        est=d.verdade.values
        am=rng.binomial(N,np.clip(est,0,1))/N
        sh=(am*N+G*15)/(N+15)
        f=np.where(sh<lo,0,np.where(sh<hi,1,2))
        print(f'  {N:>2} respostas observadas  : {accuracy_score(y,f):.0%}')
