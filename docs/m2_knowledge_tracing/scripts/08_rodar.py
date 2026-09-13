"""Roda o M2 treinado sobre alunos reais do conjunto de teste.

Replica o que a API faria: a cada resposta, atualiza o estado do aluno no topico
e devolve dominio, confianca, tendencia e o nivel do gate. Ao lado, o que a
regra atual do TrailUp diria no mesmo instante.
"""
import os, pickle, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
M=pickle.load(open(f'{HERE}/m2_model.pkl','rb'))
mod, FEATS, DIFF, GLOB = M['model'], M['feats'], M['diff'], M['global']
print(f'modelo carregado: {len(FEATS)} features | dificuldade de {len(DIFF):,} questoes | base {GLOB:.3f}\n')

rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features.parquet')
u=df.user.unique(); rng.shuffle(u); te_u=set(u[int(.9*len(u)):])
te=df[df.user.isin(te_u)].copy()
te['q_dif']=te.qidx.map(DIFF).fillna(GLOB).astype(np.float32)
te['q_n']=np.log1p(te.qidx.map(pd.Series(1,index=DIFF.index)).fillna(0)).astype(np.float32)

def confianca(n):        # substitui o 0.66 fixo: cresce com observacao, satura
    return round(float(min(0.92, 0.35 + 0.57*(1-np.exp(-n/8)))), 2)

def faixa(p, tend):
    if p < 0.45: return 'RISCO'
    if p >= 0.72 and tend != 'caindo': return 'DOMINANDO'
    return 'ESTAVEL'

def nivel_gate(p, conf, delta, evento_perf):
    if p < 0.45 and conf >= 0.70: return 2, 'dominio baixo com confianca'
    if abs(delta) >= 0.05:        return 1, 'dominio mudou de faixa'
    if evento_perf:               return 1, 'houve resposta nova'
    return 0, 'sem mudanca -> reusa o ciclo anterior'

# escolhe 3 alunos com trajetorias diferentes no topico mais comum
cand=[]
for (usr,part),g in te.groupby(['user','part']):
    if len(g)<40: continue
    g=g.sort_index()
    cand.append((usr,part,g.y.iloc[:20].mean(),g.y.iloc[20:40].mean(),len(g)))
C=pd.DataFrame(cand,columns=['user','part','inicio','depois','n'])
alvos=[('em dificuldade', C[(C.inicio<0.35)].iloc[0]),
       ('melhorando',     C[(C.depois-C.inicio>0.25)].iloc[0]),
       ('consistente',    C[(C.inicio>0.75)&(C.depois>0.75)].iloc[0])]

for rotulo,row in alvos:
    usr,part=int(row.user),int(row.part)
    g=te[(te.user==usr)&(te.part==part)].sort_index().head(24)
    print(f'{"="*94}\nALUNO {usr} — tópico {part}  ({rotulo}: {row.inicio:.0%} de acerto nas 20 primeiras)')
    print(f'{"#":>3} {"resposta":>9} {"questao":>8} │ {"MODELO":^27} │ {"REGRA ATUAL":^17} │ gate')
    print(f'{"":>3} {"":>9} {"dific.":>8} │ {"p":>5} {"conf":>5} {"faixa":>10} {"":>3} │ {"dominio":>8} {"faixa":>8} │')
    print('─'*94)
    hist=[]; p_ant=None
    for i,(_,r) in enumerate(g.iterrows(),1):
        X=pd.DataFrame([r[FEATS].values],columns=FEATS)
        p=float(mod.predict_proba(X)[:,1][0])
        conf=confianca(r.n_prev_part)
        hist.append(p)
        tend='caindo' if len(hist)>=4 and hist[-1]<hist[-4]-0.03 else ('subindo' if len(hist)>=4 and hist[-1]>hist[-4]+0.03 else 'estavel')
        delta=0.0 if p_ant is None else p-p_ant
        nv,motivo=nivel_gate(p,conf,delta,True)
        rg=float(r.regra_trailup)
        rg_faixa='RISCO' if rg<0.45 else ('DOMINANDO' if rg>=0.72 else 'ESTAVEL')
        acertou='acertou' if r.y else 'ERROU'
        marca='' if nv==0 else ('·' if nv==1 else '‼')
        print(f'{i:>3} {acertou:>9} {r.q_dif:>8.2f} │ {p:>5.2f} {conf:>5.2f} {faixa(p,tend):>10} {marca:>3} │ '
              f'{rg:>8.2f} {rg_faixa:>8} │ {nv}')
        p_ant=p
    real=g.y.mean()
    print(f'{"":>3} acerto real nestas {len(g)}: {real:.0%} │ modelo médio {np.mean(hist):.2f} │ regra final {g.regra_trailup.iloc[-1]:.2f}')
    print(f'    erro do modelo: {abs(np.mean(hist)-real):.3f}  |  erro da regra: {abs(g.regra_trailup.iloc[-1]-real):.3f}\n')
