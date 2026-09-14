"""O display exagera +5,9 pontos. Vale calibrar? Teste pareado com holdout."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/Users/user/Documents/geis/core-trailup')
from trailup_core import pre_avaliacao as pa
d = pd.read_parquet('/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/texto/classex_grafo.parquet')
alvo=[c for c in d.columns if 'Content_AI' in c][0]
d=d.dropna(subset=['resp','gab',alvo]).reset_index(drop=True)
p=np.array([pa.avaliar(str(r),pa.preparar(str(g))).percentual_estimado for r,g in zip(d.resp,d.gab)])
y=(pd.to_numeric(d[alvo],errors='coerce').values-1)/4*100
grp=d['Task'].astype(str).values if 'Task' in d else np.zeros(len(d))
print(f'{len(p)} respostas | {len(np.unique(grp))} tarefas')
from sklearn.model_selection import GroupKFold
gk=GroupKFold(n_splits=min(5,len(np.unique(grp))))
mae_cru=[];mae_cal=[];coefs=[]
for tr,te in gk.split(p,y,grp):
    A=np.polyfit(p[tr],y[tr],1); coefs.append(A)
    pc=np.clip(np.polyval(A,p[te]),0,100)
    mae_cru.append(np.abs(p[te]-y[te]).mean()); mae_cal.append(np.abs(pc-y[te]).mean())
mc,ml=np.array(mae_cru),np.array(mae_cal); dif=mc-ml
print(f'\n  MAE cru  {mc.mean():.2f} | MAE calibrado {ml.mean():.2f} | ganho {dif.mean():+.2f} pontos')
print(f'  melhora em {(dif>0).sum()}/{len(dif)} dobras')
b=np.array([np.random.default_rng(i).choice(dif,len(dif)).mean() for i in range(4000)])
print(f'  IC95 do ganho pareado [{np.percentile(b,2.5):+.2f}, {np.percentile(b,97.5):+.2f}]')
C=np.array(coefs); print(f'\n  inclinacao {C[:,0].mean():.4f} +- {C[:,0].std(ddof=1):.4f} | intercepto {C[:,1].mean():.2f} +- {C[:,1].std(ddof=1):.2f}')
A=np.polyfit(p,y,1); print(f'  ajuste completo: y = {A[0]:.4f}*x + {A[1]:.2f}')
pc=np.clip(np.polyval(A,p),0,100)
print(f'  faixa exibida: cru {p.min():.0f}-{p.max():.0f} -> calibrado {pc.min():.0f}-{pc.max():.0f} (real {y.min():.0f}-{y.max():.0f})')
print(f'  vies: cru {p.mean()-y.mean():+.1f} -> calibrado {pc.mean()-y.mean():+.1f}')
