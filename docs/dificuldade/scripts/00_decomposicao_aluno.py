"""Dificuldade da questao POR ALUNO: e deslocamento ou e interacao?

A = so a questao        p = sigmoid(-b_q)                 (igual para todos)
B = aditivo (Rasch)     p = sigmoid(theta_aluno - b_q)    (desloca por habilidade)
C = habilidade por topico  theta varia por part
D = modelo completo     deixa o boosting achar interacao

Se D nao ganha de C, nao ha interacao para capturar e a conta simples basta.
Saida sempre em PROBABILIDADE, avaliada por calibracao, nao por acerto binario.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
HERE=os.path.dirname(os.path.abspath(__file__))
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{M2}/features_v3.parquet')
u=df.user.unique(); rng.shuffle(u); n=len(u)
s=pd.Series('test',index=u).to_dict()
s.update(dict.fromkeys(u[:int(.8*n)],'train')); s.update(dict.fromkeys(u[int(.8*n):int(.9*n)],'val'))
df['split']=df.user.map(s)
tr=df[df.split=='train']; te=df[df.split=='test'].copy()
G=float(tr.y.mean())
g=tr.groupby('qidx').y.agg(['sum','size'])
dif=((g['sum']+G*20)/(g['size']+20))          # dificuldade da questao (0-1), so do treino
NQ=g['size']                                   # quantas respostas sustentam a estimativa
print(f'treino {len(tr):,} | teste {len(te):,} | acerto global {G:.3f} | {len(dif):,} questoes')
L=lambda p: np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
te['b_q']=L(te.qidx.map(dif).fillna(G).values)                  # facilidade da questao (logit)
te['q_n']=np.log1p(te.qidx.map(NQ).fillna(0).values)
te['th']=L(te.acc_prev.values)                                   # habilidade geral (causal)
te['th_p']=L(te.acc_prev_part.values)                            # habilidade no topico (causal)
tr=tr.copy()
tr['b_q']=L(tr.qidx.map(dif).fillna(G).values); tr['th']=L(tr.acc_prev.values); tr['th_p']=L(tr.acc_prev_part.values)
tr['q_n']=np.log1p(tr.qidx.map(NQ).fillna(0).values)
def ece(y,p,b=10):
    e=0.0;i=np.digitize(p,np.linspace(0,1,b+1)[1:-1])
    for k in range(b):
        m=i==k
        if m.sum(): e+=m.mean()*abs(y[m].mean()-p[m].mean())
    return e
def av(nome,p):
    p=np.clip(p,1e-6,1-1e-6)
    return {'modelo':nome,'AUC':roc_auc_score(te.y,p),'logloss':log_loss(te.y,p),
            'brier':brier_score_loss(te.y,p),'ECE':ece(te.y.values,p)}
rows=[av('A  so a questao (igual para todos)',1/(1+np.exp(-te.b_q)))]
for F,nm in [(['b_q','th'],'B  aditivo: questao + habilidade'),
             (['b_q','th','th_p'],'C  + habilidade no topico'),
             (['b_q','th','th_p','n_prev','n_prev_part'],'C+ com quanta evidencia ha')]:
    m=LogisticRegression(max_iter=1000).fit(tr[F],tr.y)
    rows.append(av(nm,m.predict_proba(te[F])[:,1]))
    if nm.startswith('B'): coefB=dict(zip(F,m.coef_[0]))
FD=['b_q','th','th_p','n_prev','n_prev_part','ewma','ewma_part','streak','acc5','acc10',
    'dt_log','seen_before','part','q_n','pos_sessao','diagnosis','ntags']
md=HistGradientBoostingClassifier(max_iter=400,learning_rate=0.06,max_depth=6,
    min_samples_leaf=200,random_state=0,categorical_features=[FD.index('part')]).fit(tr[FD],tr.y)
p_d=md.predict_proba(te[FD])[:,1]
rows.append(av('D  modelo completo (acha interacao)',p_d))
print('\n=== probabilidade de acerto (teste, alunos nunca vistos) ===')
print(pd.DataFrame(rows).set_index('modelo').round(4).to_string())
print(f'\ncoeficientes do aditivo: {", ".join(f"{k}={v:+.3f}" for k,v in coefB.items())}')
print('  (se os dois forem proximos de 1, o modelo de Rasch se sustenta:')
print('   logit(p) = habilidade - dificuldade, sem mais nada)')

print('\n=== ha INTERACAO? a ordem de dificuldade muda entre alunos? ===')
# residuo do aditivo por (aluno, part): se houver interacao, o residuo e sistematico
mB=LogisticRegression(max_iter=1000).fit(tr[['b_q','th']],tr.y)
te['res']=te.y-mB.predict_proba(te[['b_q','th']])[:,1]
rp=te.groupby(['user','part']).res.agg(['mean','size'])
rp=rp[rp['size']>=30]
print(f'  {len(rp):,} pares (aluno, topico) com >=30 respostas')
print(f'  residuo medio por par: desvio {rp["mean"].std():.4f} (0 = aditivo explica tudo)')
h=te.groupby('user').res.agg(['mean','size']); h=h[h['size']>=100]
print(f'  residuo medio por ALUNO: desvio {h["mean"].std():.4f}')
print(f'  ganho de D sobre C em AUC: {rows[-1]["AUC"]-rows[2]["AUC"]:+.4f}')
