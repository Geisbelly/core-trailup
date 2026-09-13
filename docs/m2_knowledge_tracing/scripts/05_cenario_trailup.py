"""Os dois testes que decidem o uso em produto:

1) ALUNO NOVO de verdade - primeiras respostas do aluno na base, nao "poucas
   respostas neste topico tendo centenas em outros".
2) A metrica do PRODUTO nao e acertar a proxima resposta: e o `dominio` por
   topico bater com o desempenho futuro do aluno naquele topico.
"""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.RandomState(20260912)
df = pd.read_parquet(f'{HERE}/features.parquet')
users = df.user.unique(); rng.shuffle(users)
n=len(users); tr_u, te_u = users[:int(.8*n)], users[int(.9*n):]
s = pd.Series('test', index=users).to_dict(); s.update(dict.fromkeys(tr_u,'train'))
df['split']=df.user.map(s); tr,te = df[df.split=='train'], df[df.split=='test']
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
y=te.y.values

print('=== 1) ALUNO NOVO: primeiras N respostas do aluno (n_prev global) ===')
print('    a regra parte de 0.5 e o modelo so tem a dificuldade da questao\n')
for lo,hi,lab in [(0,4,'1a a 5a'),(5,19,'6a a 20a'),(20,99,'21a a 100a'),(100,10**9,'101a+')]:
    k=((te.n_prev>=lo)&(te.n_prev<=hi)).values
    if k.sum()<500: continue
    print(f'  {lab:10s} {k.sum():7,} linhas | modelo {roc_auc_score(y[k],te.p.values[k]):.4f} | '
          f'regra {roc_auc_score(y[k],te.regra_trailup.values[k]):.4f} | '
          f'so dificuldade {roc_auc_score(y[k],te.q_dif.values[k]):.4f}')

print('\n=== 2) METRICA DE PRODUTO: dominio por (aluno, topico) prediz o futuro? ===')
print('    corta a serie de cada aluno na metade; estima o dominio com a 1a metade')
print('    e compara com a taxa de acerto real na 2a metade do MESMO topico\n')
te=te.sort_values(['user','part']).reset_index(drop=True)
lin=[]
for (u,p),gr in te.groupby(['user','part'],sort=False):
    if len(gr)<20: continue
    h=len(gr)//2; a,b=gr.iloc[:h],gr.iloc[h:]
    lin.append({'modelo':a.p.mean(),'regra':a.regra_trailup.iloc[-1],
                'acc_passada':a.y.mean(),'futuro':b.y.mean(),'n':len(gr)})
L=pd.DataFrame(lin)
print(f'  {len(L):,} pares (aluno, topico) com >=20 respostas')
for c,lab in [('modelo','dominio do MODELO (media de p)'),('regra','dominio da REGRA (ultimo valor)'),
              ('acc_passada','simplesmente a % de acerto passada')]:
    rho=spearmanr(L[c],L.futuro).statistic; mae=abs(L[c]-L.futuro).mean()
    print(f'  {lab:36s} Spearman {rho:+.4f} | erro absoluto medio {mae:.4f}')

print('\n=== 3) o gate: "dominio < 0.45" separa quem precisa de reforco? ===')
print('    alvo: aluno cujo desempenho FUTURO no topico fica abaixo de 50%\n')
alvo=(L.futuro<0.5).astype(int)
for c,lab in [('modelo','modelo'),('regra','regra do TrailUp')]:
    auc=roc_auc_score(alvo,-L[c])
    disp=(L[c]<0.45)
    prec=alvo[disp].mean() if disp.sum() else float('nan')
    print(f'  {lab:18s} AUC {auc:.4f} | dispara em {disp.mean():5.1%} dos casos | '
          f'precisao quando dispara {prec:.1%} | base {alvo.mean():.1%}')
