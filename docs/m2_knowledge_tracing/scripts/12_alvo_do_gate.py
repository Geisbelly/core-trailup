"""Treinar no alvo que o gate realmente decide.

O gate nao pergunta "acerta a proxima?" - pergunta "este aluno vai travar neste
topico?". Usar p_next_correct como proxy disso perde informacao: uma questao
dificil derruba p sem que o aluno esteja travado.

Alvo direto: a taxa de acerto do aluno nas PROXIMAS 10 respostas do topico < 50%.
"""
import os, gc, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE=os.path.dirname(os.path.abspath(__file__))
H=10   # horizonte
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features_v2.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict()
s.update(dict.fromkeys(u[:int(.8*n)],'train')); s.update(dict.fromkeys(u[int(.8*n):int(.9*n)],'val'))
df['split']=df.user.map(s)
m_tr=df.split=='train'
g=df.loc[m_tr].groupby('qidx').y.agg(['mean','size']); G=float(df.loc[m_tr,'y'].mean())
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
df['q_dif']=df.qidx.map(diff).fillna(G).astype(np.float32)
df['q_n']=np.log1p(df.qidx.map(g['size']).fillna(0)).astype(np.float32)

# alvo: media dos proximos H acertos NO MESMO (user, part)
df=df.sort_values(['user','part']).reset_index(drop=True)
gr=df.groupby(['user','part'],sort=False).y
fut=gr.transform(lambda s: s.shift(-1).rolling(H,min_periods=H).mean().shift(-(H-1)))
df['fut']=fut
d=df[df.fut.notna()].copy()
d['alvo']=(d.fut<0.5).astype(np.int8)
print(f'amostras com {H} respostas futuras no topico: {len(d):,} | positivos {d.alvo.mean():.1%}')
del df; gc.collect()

F=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
   'dt_log','dt_part_log','seen_before','ntags','diagnosis','part','q_dif','q_n',
   'acc5','acc10','n_sessao','pos_sessao','dt_rel','bundle_pos','acc_bundle',
   'tag_acc','tag_n','tag_acc_min','tag_acc_max','tag_ewma','tag_novo']
tr,va,te=(d[d.split==k] for k in ('train','val','test'))
print(f'treino {len(tr):,} | val {len(va):,} | teste {len(te):,}')
def ece(y,p,b=10):
    e=0.0;i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
best=None
for lr,it,dep,leaf in [(0.06,400,6,200),(0.04,700,7,150),(0.05,500,None,150)]:
    m=HistGradientBoostingClassifier(max_iter=it,learning_rate=lr,max_depth=dep,
        min_samples_leaf=leaf,l2_regularization=1.0,random_state=0,
        categorical_features=[F.index('part')]).fit(tr[F],tr.alvo)
    a=roc_auc_score(va.alvo,m.predict_proba(va[F])[:,1])
    print(f'  lr={lr} it={it} depth={dep} leaf={leaf} -> val AUC {a:.4f}')
    if best is None or a>best[0]: best=(a,m)
m=best[1]
y=te.alvo.values
p_direto=m.predict_proba(te[F])[:,1]
p_proxy=1-te.q_dif.values*0  # placeholder
# proxy: usar o modelo de proxima-resposta ja treinado
import pickle
try:
    M1=pickle.load(open(f'{HERE}/m2_v2_v2.pkl','rb'))
    p_proxy=1-M1['model'].predict_proba(te[M1['feats']])[:,1]
    proxy_ok=True
except Exception as e:
    p_proxy=1-te.ewma_part.values; proxy_ok=False
    print('  (modelo v2 nao encontrado, usando ewma como proxy)',e)
rows=[]
for nome,p in [('ALVO DIRETO: treinado em "vai travar"',p_direto),
               ('proxy: 1 - p(acerta a proxima)',p_proxy),
               ('regra do TrailUp (dominio<0.45)',1-te.regra_trailup.values),
               ('acerto acumulado no topico',1-te.acc_prev_part.values)]:
    p=np.clip(p,1e-6,1-1e-6)
    rows.append({'criterio':nome,'AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p),
                 'Brier':brier_score_loss(y,p) if 'DIRETO' in nome or 'proxy' in nome else np.nan,
                 'ECE':ece(y,p) if 'DIRETO' in nome else np.nan})
print(f'\n=== prever "o aluno vai ficar abaixo de 50% nas proximas {H}" (teste) ===')
print(f'base: {y.mean():.1%} dos casos\n')
print(pd.DataFrame(rows).set_index('criterio').round(4).to_string())

print(f'\n=== precisao do gate por taxa de disparo ===')
print(f'{"dispara em":>11} │ {"ALVO DIRETO":^22} │ {"proxy":^12} │ {"regra":^12}')
print(f'{"":>11} │ {"limiar":>8} {"precisao":>8} {"cob.":>5} │ {"precisao":>12} │ {"precisao":>12}')
print('─'*68)
for taxa in (0.05,0.10,0.15,0.20,0.30):
    t=np.quantile(p_direto,1-taxa); msk=p_direto>=t
    tp=np.quantile(p_proxy,1-taxa); mp=p_proxy>=tp
    tr_=np.quantile(1-te.regra_trailup.values,1-taxa); mr=(1-te.regra_trailup.values)>=tr_
    print(f'{taxa:>10.0%} │ {t:>8.2f} {y[msk].mean():>8.1%} {y[msk].sum()/y.sum():>5.0%} │ '
          f'{y[mp].mean():>12.1%} │ {y[mr].mean():>12.1%}')
pickle.dump({'model':m,'feats':F,'diff':diff,'global':G,'horizonte':H},
            open(f'{HERE}/m2_gate.pkl','wb'))
json.dump(rows,open(f'{HERE}/resultados_gate.json','w'),ensure_ascii=False,indent=1,default=str)
print(f"\nm2_gate.pkl salvo ({os.path.getsize(f'{HERE}/m2_gate.pkl')/1e6:.1f} MB)")
