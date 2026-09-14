"""SIGMA_COORTE=0,0153 entrega 87,7% num intervalo de 90%. Calibra em 4 sementes
e VALIDA em 4 sementes que nao participaram do ajuste."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0,'/Users/user/Documents/geis/core-trailup')
import trailup_core.dificuldade as D
E='/Users/user/Downloads/Inicio/EdNet-KT3'
f=sorted(glob.glob(f'{E}/**/u*.csv',recursive=True))
rng=np.random.default_rng(0); sel=rng.choice(len(f),12000,replace=False)
L=[]
for i in sel:
    try: d=pd.read_csv(f[i],usecols=['item_id','user_answer'])
    except Exception: continue
    if len(d)<10: continue
    d['aluno']=os.path.basename(f[i]); L.append(d)
R=pd.concat(L,ignore_index=True)
q=pd.read_csv('/Users/user/Downloads/Inicio/EdNet-Contents/contents/questions.csv',usecols=['question_id','correct_answer'])
q['item_id']=q.question_id
R=R.merge(q,on='item_id'); R['ok']=(R.user_answer==R.correct_answer).astype(int)
alunos=R.aluno.unique()
def particao(s):
    rg=np.random.default_rng(s); A=set(alunos[rg.random(len(alunos))<0.5])
    tr=R[R.aluno.isin(A)]; te=R[~R.aluno.isin(A)]
    a=tr.groupby('item_id').ok.agg(['sum','size']); b=te.groupby('item_id').ok.agg(['sum','size'])
    j=a.join(b,lsuffix='_t',rsuffix='_v',how='inner')
    return j[(j['size_t']>=5)&(j['size_v']>=30)]
P={s:particao(s) for s in [7,11,23,42,101,2026,314,999]}
def cobertura(j,sigma,nivel=0.90):
    old=D.SIGMA_COORTE; D.SIGMA_COORTE=sigma
    n=int(j['size_v'].median()); dentro=0
    for st,nt,sv,nv in zip(j['sum_t'],j['size_t'],j['sum_v'],j['size_v']):
        p=D.prever_turma(int(st),int(nt),n,nivel=nivel)
        if p.minimo<=sv/nv<=p.maximo: dentro+=1
    D.SIGMA_COORTE=old; return dentro/len(j)
AJ=[7,11,23,42]; VA=[101,2026,314,999]
print(f'{"sigma":>8} {"ajuste":>9} {"validacao":>10}')
melhor=None
for sig in [0.0153,0.020,0.025,0.030,0.035,0.040,0.045,0.050]:
    a=np.mean([cobertura(P[s],sig) for s in AJ])
    v=np.mean([cobertura(P[s],sig) for s in VA])
    print(f'{sig:>8.4f} {a:>9.1%} {v:>10.1%}',flush=True)
    if melhor is None or abs(a-0.90)<abs(melhor[1]-0.90): melhor=(sig,a,v)
print(f'\nescolhido pelo AJUSTE: sigma {melhor[0]:.4f} -> ajuste {melhor[1]:.1%} | validacao {melhor[2]:.1%}')
for nv in (0.50,0.80,0.95):
    a=np.mean([cobertura(P[s],melhor[0],nv) for s in VA])
    print(f'  nivel {nv:.0%} -> cobertura {a:.1%} (validacao)')
