"""Enriquecer o grafo de REFERENCIA melhora a avaliacao?

Hoje a referencia e so o gabarito (mediana 466 chars). A proposta e construir o
grafo do DOMINIO a partir de conteudo didatico + questoes do mesmo assunto.

Teste sem raspar nada: usar as respostas BEM AVALIADAS DE OUTROS ALUNOS da mesma
tarefa como corpus de dominio. Se enriquecer a referencia melhora, a arquitetura
se paga; se nao, o gabarito ja bastava.

Cuidado com vazamento: o aluno avaliado nunca entra na propria referencia, e as
respostas usadas vem so do conjunto de TREINO de cada fold.
"""
import os, re, numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_grafo.parquet').reset_index(drop=True)
d['aluno']=d.aluno.fillna(pd.Series([f'anon_{i}' for i in range(len(d))]))
print(f'{len(d)} respostas | {d.Task.nunique()} tarefas | ~{len(d)//d.Task.nunique()} por tarefa')
TOK=re.compile(r'[A-Za-zÄÖÜäöüß]{3,}')
todos=Counter(w for t in pd.concat([d.resp,d.gab,d.q]) for w in TOK.findall(str(t).lower()))
STOP={w for w,_ in todos.most_common(60)}
def grafo(t,jan=4):
    ws=[w for w in (x.lower() for x in TOK.findall(str(t))) if w not in STOP]
    nos=Counter(ws); ar=Counter()
    for i,a in enumerate(ws):
        for b in ws[i+1:i+jan]:
            if a!=b: ar[(min(a,b),max(a,b))]+=1
    return nos,ar
def soma(gs):
    N,A=Counter(),Counter()
    for n,a in gs: N.update(n); A.update(a)
    return N,A
def feats(r,ref_n,ref_a,nq):
    nr,ar=grafo(r); sr,sg=set(nr),set(ref_n); er,eg=set(ar),set(ref_a)
    ni,ei=sr&sg,er&eg
    peso=sum(ref_n[w] for w in ni)/max(sum(ref_n.values()),1)
    pe=sum(min(ar[e],ref_a[e]) for e in ei)/max(sum(ref_a.values()),1)
    grau=defaultdict(int)
    for (a,b),w in ref_a.items(): grau[a]+=w; grau[b]+=w
    top=[w for w,_ in sorted(grau.items(),key=lambda x:-x[1])[:15]]
    return {'no_jac':len(ni)/max(len(sr|sg),1),'no_cob':len(ni)/max(len(sg),1),'no_peso':peso,
            'ar_jac':len(ei)/max(len(er|eg),1),'ar_cob':len(ei)/max(len(eg),1),'ar_peso':pe,
            'cob_central':float(np.mean([w in sr for w in top])) if top else 0.0,
            'divag':len(sr-sg-nq)/max(len(sr),1),'razao':len(sr)/max(len(sg),1),
            'n_nos':len(sr),'n_ar':len(er)}
y=d.nota.values; g=d.aluno.values
def qwk(a,b): return cohen_kappa_score(np.clip(np.round(a),1,5).astype(int),
                                       np.clip(np.round(b),1,5).astype(int),weights='quadratic')
GAB={t:grafo(gg) for t,gg in d.groupby('Task').gab.first().items()}
NQ={t:set(grafo(qq)[0]) for t,qq in d.groupby('Task').q.first().items()}
resultados={}
for K,nome in [(0,'so o gabarito (atual)'),(5,'gabarito + 5 melhores respostas'),
               (15,'gabarito + 15 melhores'),(40,'gabarito + 40 melhores'),
               (-1,'gabarito + TODAS do treino')]:
    P=np.zeros(len(d))
    for tr,te in GroupKFold(5).split(d,y,groups=g):
        T=d.iloc[tr]
        REF={}
        for t in d.Task.unique():
            sub=T[T.Task==t].sort_values('nota',ascending=False)
            if K>0: sub=sub.head(K)
            elif K==0: sub=sub.iloc[:0]
            gs=[GAB[t]]+[grafo(x) for x in sub.resp]
            REF[t]=soma(gs)
        X=[]
        for i in list(tr)+list(te):
            rn,ra=REF[d.Task.iloc[i]]
            X.append(feats(d.resp.iloc[i],rn,ra,NQ[d.Task.iloc[i]]))
        X=pd.DataFrame(X); X['log_len']=np.log1p(d.resp.astype(str).str.len().values[list(tr)+list(te)])
        ntr=len(tr)
        m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,max_depth=3,
            min_samples_leaf=30,l2_regularization=1.0,random_state=0).fit(X.iloc[:ntr],y[tr])
        P[te]=m.predict(X.iloc[ntr:])
    resultados[nome]={'Spearman':spearmanr(P,y).statistic,'QWK':qwk(P,y),'MAE':mean_absolute_error(y,P)}
    print(f'  {nome:>34}: Spearman {resultados[nome]["Spearman"]:.3f} | QWK {resultados[nome]["QWK"]:.3f}')
print('\n=== enriquecer a referencia ajuda? ===')
R=pd.DataFrame(resultados).T
print(R.round(3).to_string())
base=R.loc['so o gabarito (atual)','Spearman']
print(f'\nganho maximo sobre o gabarito sozinho: {R.Spearman.max()-base:+.3f} Spearman')
