"""Modelo x regra no MESMO dado corrigido. Taxa-base mudou, so lift compara."""
import os,pickle,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__))
M=pickle.load(open(f'{HERE}/m2_gate_v3.pkl','rb'))
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features_v3.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict()
s.update(dict.fromkeys(u[:int(.8*n)],'train')); s.update(dict.fromkeys(u[int(.8*n):int(.9*n)],'val'))
df['split']=df.user.map(s)
g=df[df.split=='train'].groupby('qidx').y.agg(['mean','size']); G=float(df.loc[df.split=='train','y'].mean())
diff=(g['mean']*g['size']+G*20)/(g['size']+20)
df['q_dif']=df.qidx.map(diff).fillna(G).astype(np.float32)
df['q_n']=np.log1p(df.qidx.map(g['size']).fillna(0)).astype(np.float32)
df=df.sort_values(['user','part']).reset_index(drop=True)
fut=df.groupby(['user','part'],sort=False).y.transform(lambda s:s.shift(-1).rolling(10,min_periods=10).mean().shift(-9))
df['t']=(fut<0.5).astype(float); df=df[fut.notna()]
te=df[df.split=='test']; y=te.t.values; B=y.mean()
p_mod=M['model'].predict_proba(te[M['feats']])[:,1]
print(f'teste {len(te):,} | taxa-base {B:.1%} (era 26,7% no dado contaminado)\n')
cands=[('MODELO do gate (v3)',p_mod),
       ('regra do TrailUp',1-te.regra_trailup.values),
       ('acerto acumulado no topico',1-te.acc_prev_part.values),
       ('EWMA no topico',1-te.ewma_part.values)]
print(f"{'criterio':30s} {'AUC':>7} {'AP':>7} {'lift@10%':>9}")
print('-'*57)
for nome,p in cands:
    k=p>=np.quantile(p,0.90)
    print(f'{nome:30s} {roc_auc_score(y,p):>7.4f} {average_precision_score(y,p):>7.4f} {y[k].mean()/B:>8.2f}x')
print(f'\n=== precisao por taxa de disparo (base {B:.1%}) ===')
print(f"{'dispara':>8} │ {'MODELO':^20} │ {'REGRA':^20}")
print(f"{'':>8} │ {'precisao':>9} {'lift':>5} {'cob':>4} │ {'precisao':>9} {'lift':>5} {'cob':>4}")
print('─'*56)
pr=1-te.regra_trailup.values
for taxa in (0.05,0.10,0.15,0.20):
    km=p_mod>=np.quantile(p_mod,1-taxa); kr=pr>=np.quantile(pr,1-taxa)
    print(f'{taxa:>7.0%} │ {y[km].mean():>9.1%} {y[km].mean()/B:>4.1f}x {y[km].sum()/y.sum():>4.0%} │ '
          f'{y[kr].mean():>9.1%} {y[kr].mean()/B:>4.1f}x {y[kr].sum()/y.sum():>4.0%}')
