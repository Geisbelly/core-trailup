"""Efeito agregado no gate + dominio na definicao correta (dificuldade de referencia)."""
import os, pickle, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
M=pickle.load(open(f'{HERE}/m2_model.pkl','rb')); mod,FEATS,DIFF,GLOB=M['model'],M['feats'],M['diff'],M['global']
rng=np.random.RandomState(20260912)
df=pd.read_parquet(f'{HERE}/features.parquet')
u=df.user.unique(); rng.shuffle(u); te=df[df.user.isin(set(u[int(.9*len(u)):]))].copy()
te['q_dif']=te.qidx.map(DIFF).fillna(GLOB).astype(np.float32)
te['q_n']=np.log1p(te.qidx.map(pd.Series(1,index=DIFF.index)).fillna(0)).astype(np.float32)
te['p']=mod.predict_proba(te[FEATS])[:,1]

# DOMINIO = o que o modelo preveria se a questao fosse de dificuldade MEDIANA.
# Sem isso o numero oscila com a questao sorteada e nao e comparavel entre topicos.
ref=te.copy(); ref['q_dif']=float(np.median(DIFF.values)); ref['q_n']=float(te.q_n.median())
te['dominio']=mod.predict_proba(ref[FEATS])[:,1]
print(f'dificuldade de referencia usada: {np.median(DIFF.values):.3f}\n')
print('=== dominio (dificuldade de referencia) vs p bruto ===')
print(f'  p bruto   : media {te.p.mean():.3f} | desvio {te.p.std():.3f}')
print(f'  dominio   : media {te.dominio.mean():.3f} | desvio {te.dominio.std():.3f}')
print(f'  regra     : media {te.regra_trailup.mean():.3f} | desvio {te.regra_trailup.std():.3f} | '
      f'no teto {(te.regra_trailup>=.98).mean():.1%}')

print('\n=== o gate sobre todo o conjunto de teste ===')
te=te.sort_values(['user','part'])
te['dom_ant']=te.groupby(['user','part']).dominio.shift()
te['delta']=(te.dominio-te.dom_ant).abs().fillna(1.0)
def niveis(dom, delta, conf_ok):
    n2=(dom<0.45)&conf_ok
    n1=(~n2)&((delta>=0.05))
    n0=~(n1|n2)
    return n0,n1,n2
conf_ok=te.n_prev_part>=8
n0,n1,n2=niveis(te.dominio,te.delta,conf_ok)
r_n2=(te.regra_trailup<0.45); r_n1=(~r_n2)
print(f'  MODELO  nivel 0 (nada): {n0.mean():6.1%} | nivel 1 (sem LLM): {n1.mean():6.1%} | nivel 2 (LLM): {n2.mean():6.1%}')
print(f'  REGRA   equivalente ao nivel 2 (dominio<0.45): {r_n2.mean():6.1%}  — sem niveis 0/1, o grafo roda sempre')
print(f'\n  hoje: 100% dos lotes abrem o grafo (8-14 chamadas). Com o gate: {n2.mean():.1%} em nivel 2.')

# precisao do disparo: quem esta em nivel 2 realmente erra a proxima?
print(f'\n=== quando dispara nivel 2, o aluno erra mesmo? ===')
for nome,msk in [('MODELO (dominio<0.45 e conf ok)',n2),('REGRA (dominio<0.45)',r_n2)]:
    if msk.sum()<100: continue
    print(f'  {nome:34s} dispara {msk.mean():5.1%} | erro na proxima: {1-te.y[msk].mean():5.1%} '
          f'(base {1-te.y.mean():.1%})')
