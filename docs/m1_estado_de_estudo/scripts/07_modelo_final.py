"""M1 reformulado: em vez de inferir estado, estimar duas quantidades estaveis.

  dificuldade_percebida ~ base_do_aluno + dificuldade_do_conteudo (+ comportamento?)

Testa se o comportamento acrescenta ALGUMA COISA sobre as duas bases.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
d=R[R.difficulty.notna()].copy()
d['y']=(d.difficulty>=4).astype(int)
d['log_dur']=np.log1p(d.dur); d['log_gapmax']=np.log1p(d.gap_max)
d['press_min']=d.n_press/(d.dur/60); d['focus_min']=d.n_focus/(d.dur/60)
COMP=['log_dur','ativo_ratio','ev_min','press_min','focus_min','log_gapmax','frac_pausa','revisita']
G=d.y.mean()

def ece(y,p,b=10):
    e=0.0; i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e

# bases estimadas SO no treino de cada fold -> sem vazamento
res={k:np.zeros(len(d)) for k in ['base_aluno','base_texto','duas_bases','duas_mais_comp']}
for tr,te in GroupKFold(5).split(d,d.y,groups=d.actor):
    T,V=d.iloc[tr],d.iloc[te]
    # base do texto: estimavel no treino (outros alunos leram o mesmo texto)
    bt=T.groupby('assignment').y.agg(['sum','size'])
    p_txt=V.assignment.map(((bt['sum']+G*10)/(bt['size']+10))).fillna(G).values
    # base do aluno: o aluno do teste NAO esta no treino (split por aluno) ->
    # usa as leituras ANTERIORES do proprio aluno, acumuladas em ordem temporal
    V=V.sort_values(['actor','t0'])
    prev_s=V.groupby('actor').y.cumsum()-V.y; prev_n=V.groupby('actor').cumcount()
    p_alu=((prev_s+G*3)/(prev_n+3)).values
    ordem=V.index
    res['base_texto'][d.index.get_indexer(ordem)]=p_txt
    res['base_aluno'][d.index.get_indexer(ordem)]=p_alu
    res['duas_bases'][d.index.get_indexer(ordem)]=0.5*(p_txt+p_alu)
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=0.3))
    Xtr=T[COMP].copy(); Xte=V[COMP].copy()
    btr=T.groupby('assignment').y.agg(['sum','size'])
    Xtr['b_txt']=T.assignment.map((btr['sum']+G*10)/(btr['size']+10)).fillna(G).values
    Xtr['b_alu']=T.groupby('actor').y.transform(lambda s:(s.cumsum()-s+G*3)/(np.arange(len(s))+3)).values
    Xte['b_txt']=p_txt; Xte['b_alu']=p_alu
    m.fit(Xtr,T.y)
    res['duas_mais_comp'][d.index.get_indexer(ordem)]=m.predict_proba(Xte)[:,1]

y=d.y.values
print(f'{len(d):,} leituras | {d.actor.nunique()} alunos | {d.assignment.nunique()} textos | positivos {G:.1%}')
print('\n=== out-of-fold, split por aluno (o aluno do teste nunca aparece no treino) ===')
rows=[]
for k,lab in [('base_texto','so dificuldade do TEXTO'),
              ('base_aluno','so base do ALUNO (suas leituras anteriores)'),
              ('duas_bases','as duas bases, media simples'),
              ('duas_mais_comp','as duas bases + 8 features de comportamento')]:
    p=np.clip(res[k],1e-6,1-1e-6)
    rows.append({'modelo':lab,'AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p),
                 'Brier':brier_score_loss(y,p),'ECE':ece(y,p)})
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print('\nganho do comportamento sobre as duas bases: '
      f"{roc_auc_score(y,res['duas_mais_comp'])-roc_auc_score(y,res['duas_bases']):+.4f} AUC")
print('\n=== quantas leituras do proprio aluno bastam para a base pessoal? ===')
V=d.sort_values(['actor','t0']); prev_n=V.groupby('actor').cumcount().values
pa=res['base_aluno'][d.index.get_indexer(V.index)]; yv=V.y.values
for lo,hi,lab in [(0,0,'0 (primeira leitura)'),(1,2,'1-2'),(3,5,'3-5'),(6,100,'6+')]:
    m=(prev_n>=lo)&(prev_n<=hi)
    if m.sum()<40: continue
    try: a=roc_auc_score(yv[m],pa[m])
    except ValueError: a=float('nan')
    print(f'  leituras anteriores {lab:22s} {m.sum():>5,} casos | AUC {a:.4f}')
