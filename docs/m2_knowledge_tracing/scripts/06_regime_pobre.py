"""O regime do TrailUp: poucas respostas por (aluno, topico).

Com ~170 sessoes utilizaveis, nenhum aluno tem 20 respostas num topico. A
pergunta certa e: com 3, 5, 10 respostas, quem estima melhor o desempenho
futuro - a regra, a % de acerto passada, ou o modelo?
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE=os.path.dirname(os.path.abspath(__file__))
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features.parquet')
users=df.user.unique(); rng.shuffle(users)
n=len(users); tr_u=users[:int(.8*n)]; te_u=users[int(.9*n):]
s=pd.Series('test',index=users).to_dict(); s.update(dict.fromkeys(tr_u,'train'))
df['split']=df.user.map(s); tr,te=df[df.split=='train'],df[df.split=='test']
g=tr.groupby('qidx').y.agg(['mean','size']); G=tr.y.mean()
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
for d in (tr,te):
    d['q_dif']=d.qidx.map(diff).fillna(G).astype(np.float32)
    d['q_n']=np.log1p(d.qidx.map(g['size']).fillna(0)).astype(np.float32)
F=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
   'dt_log','dt_part_log','seen_before','part','q_dif','q_n','ntags','diagnosis']
m=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_depth=6,
    min_samples_leaf=200,l2_regularization=1.0,random_state=0,
    categorical_features=[F.index('part')]).fit(tr[F],tr.y)
te=te.copy(); te['p']=m.predict_proba(te[F])[:,1]

print('estimar o dominio com as PRIMEIRAS k respostas do aluno no topico e')
print('comparar com a taxa de acerto real nas 10 seguintes.\n')
print(f"{'k':>3} {'pares':>7} | {'REGRA':^17} | {'% acerto passada':^17} | {'MODELO':^17}")
print(f"{'':>3} {'':>7} | {'rho':>7} {'MAE':>8} | {'rho':>7} {'MAE':>8} | {'rho':>7} {'MAE':>8}")
te=te.sort_values(['user','part']).reset_index(drop=True)
grupos=list(te.groupby(['user','part'],sort=False))
for k in (3,5,10,20):
    lin=[]
    for (u,p),gr in grupos:
        if len(gr)<k+10: continue
        a,b=gr.iloc[:k],gr.iloc[k:k+10]
        lin.append({'regra':a.regra_trailup.iloc[-1],'acc':a.y.mean(),
                    'modelo':a.p.mean(),'futuro':b.y.mean()})
    L=pd.DataFrame(lin)
    if len(L)<300: continue
    out=[f"{k:>3} {len(L):>7,} |"]
    for c in ('regra','acc','modelo'):
        out.append(f" {spearmanr(L[c],L.futuro).statistic:>+7.4f} {abs(L[c]-L.futuro).mean():>8.4f} |")
    print(''.join(out).rstrip('|'))

print('\n=== gate no regime pobre (k=5): quem vai ficar abaixo de 50% nas proximas 10? ===')
lin=[]
for (u,p),gr in grupos:
    if len(gr)<15: continue
    a,b=gr.iloc[:5],gr.iloc[5:15]
    lin.append({'regra':a.regra_trailup.iloc[-1],'acc':a.y.mean(),'modelo':a.p.mean(),
                'futuro':b.y.mean()})
L=pd.DataFrame(lin); alvo=(L.futuro<0.5).astype(int)
print(f'  {len(L):,} pares | base: {alvo.mean():.1%} ficam abaixo de 50%\n')
for c,lab,thr in [('regra','regra do TrailUp',0.45),('acc','% acerto nas 5 primeiras',0.45),('modelo','modelo M2',0.45)]:
    auc=roc_auc_score(alvo,-L[c]); disp=L[c]<thr
    prec=alvo[disp].mean() if disp.sum() else float('nan')
    rec=(alvo[disp].sum()/alvo.sum()) if alvo.sum() else float('nan')
    print(f'  {lab:26s} AUC {auc:.4f} | dispara {disp.mean():5.1%} | precisao {prec:5.1%} | cobertura {rec:5.1%}')
