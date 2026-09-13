"""Features v2: o que faltava na v1.

1. HABILIDADE POR TAG - 189 tags contra 7 parts. E a decomposicao que define
   knowledge tracing; a v1 usava so `part` e jogou fora a estrutura fina.
2. JANELA RECENTE - acerto nas ultimas 5 e 10. Media acumulada dilui o presente.
3. TEMPO RELATIVO - resposta lenta PARA ESTE ALUNO sinaliza esforco; o dt
   absoluto da v1 confunde aluno rapido com questao facil.
4. POSICAO NA SESSAO - fadiga dentro da sessao, e qual sessao e.
"""
import os, numpy as np, pandas as pd
from collections import defaultdict, deque

HERE=os.path.dirname(os.path.abspath(__file__))
BASE='/Users/user/Downloads/Inicio'
ALPHA=0.3; DT_CLIP=600.0; GAP_SESSAO=1800.0

q=pd.read_csv(f'{BASE}/EdNet-Contents/contents/questions.csv')
q['qidx']=q.question_id.str[1:].astype(np.int32)
TAGS={r.qidx:(tuple() if str(r.tags)=='-1' else tuple(int(t) for t in str(r.tags).split(';') if t.strip().lstrip('-').isdigit()))
      for r in q.itertuples()}
BUNDLE={r.qidx:r.bundle_id for r in q.itertuples()}
print(f'questoes {len(q):,} | tags distintas {len(set(t for v in TAGS.values() for t in v))}')

df=pd.read_parquet(f'{HERE}/responses.parquet')
user=df.user.values; part=df.part.values.astype(np.int64); qidx=df.qidx.values
ts=df.ts.values; y=df.correct.values.astype(np.float64); n=len(df)
print(f'respostas {n:,}')

cols={k:np.zeros(n,np.float32) for k in [
 'n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
 'dt_log','dt_part_log','seen_before','regra_trailup',
 'tag_acc','tag_n','tag_acc_min','tag_acc_max','tag_ewma','tag_novo',
 'acc5','acc10','n_sessao','pos_sessao','dt_rel','bundle_pos','acc_bundle']}

cur=-1
for i in range(n):
    u=user[i]
    if u!=cur:
        cur=u; cnt=0; hit=0; ew=0.5; strk=0.0; last=-1
        pc=defaultdict(int); ph=defaultdict(float); pe={}; pl={}; pm={}; seen=defaultdict(int)
        tc=defaultdict(int); th=defaultdict(float); tew={}
        ult=deque(maxlen=10); dts=deque(maxlen=50)
        sess=0; pos=0; bund=None; bpos=0; bhit=0
    p=part[i]; tg=TAGS.get(qidx[i],())
    # sessao
    gap=(ts[i]-last)/1000.0 if last>0 else -1
    if last<0 or gap>GAP_SESSAO: sess+=1; pos=0
    pos+=1
    # bundle
    b=BUNDLE.get(qidx[i])
    if b!=bund: bund=b; bpos=0; bhit=0
    bpos+=1
    # --- escreve estado ANTES de atualizar
    cols['n_prev'][i]=cnt; cols['acc_prev'][i]=hit/cnt if cnt else 0.5; cols['ewma'][i]=ew
    c_=pc[p]; cols['n_prev_part'][i]=c_; cols['acc_prev_part'][i]=ph[p]/c_ if c_ else 0.5
    cols['ewma_part'][i]=pe.get(p,0.5); cols['streak'][i]=strk
    cols['seen_before'][i]=seen[qidx[i]]; cols['regra_trailup'][i]=pm.get(p,0.5)
    cols['dt_log'][i]=np.log1p(min(gap,DT_CLIP)) if gap>0 else -1.0
    lp=pl.get(p,-1); cols['dt_part_log'][i]=np.log1p(min((ts[i]-lp)/1000.0,DT_CLIP)) if lp>0 else -1.0
    # tags
    if tg:
        accs=[th[t]/tc[t] for t in tg if tc[t]>0]
        cols['tag_n'][i]=float(np.mean([tc[t] for t in tg]))
        cols['tag_acc'][i]=float(np.mean(accs)) if accs else 0.5
        cols['tag_acc_min'][i]=float(min(accs)) if accs else 0.5
        cols['tag_acc_max'][i]=float(max(accs)) if accs else 0.5
        cols['tag_ewma'][i]=float(np.mean([tew.get(t,0.5) for t in tg]))
        cols['tag_novo'][i]=float(sum(1 for t in tg if tc[t]==0)/len(tg))
    else:
        for k in ('tag_acc','tag_acc_min','tag_acc_max','tag_ewma'): cols[k][i]=0.5
        cols['tag_novo'][i]=1.0
    cols['acc5'][i]=float(np.mean(list(ult)[-5:])) if ult else 0.5
    cols['acc10'][i]=float(np.mean(ult)) if ult else 0.5
    cols['n_sessao'][i]=sess; cols['pos_sessao'][i]=pos
    med=float(np.median(dts)) if len(dts)>=5 else np.nan
    cols['dt_rel'][i]=np.log1p(min(gap,DT_CLIP))-np.log1p(med) if (gap>0 and med==med) else 0.0
    cols['bundle_pos'][i]=bpos; cols['acc_bundle'][i]=bhit/(bpos-1) if bpos>1 else 0.5
    # --- atualiza
    c=y[i]; cnt+=1; hit+=c; ew=ALPHA*c+(1-ALPHA)*ew
    strk=(strk+1) if c else (strk-1 if strk<=0 else -1.0)
    if c and strk<0: strk=1.0
    pc[p]+=1; ph[p]+=c; pe[p]=ALPHA*c+(1-ALPHA)*pe.get(p,0.5); pl[p]=ts[i]
    pm[p]=min(0.98,max(0.05,pm.get(p,0.5)+(0.08 if c else -0.07)))
    for t in tg:
        tc[t]+=1; th[t]+=c; tew[t]=ALPHA*c+(1-ALPHA)*tew.get(t,0.5)
    seen[qidx[i]]+=1; ult.append(c); bhit+=c
    if gap>0: dts.append(min(gap,DT_CLIP))
    last=ts[i]

out=pd.DataFrame({'user':user,'qidx':qidx,'part':part.astype(np.int8),'y':y.astype(np.int8),
                  'ntags':df.ntags.values,'diagnosis':df.diagnosis.values,**cols})
out.to_parquet(f'{HERE}/features_v2.parquet',index=False)
print(f'-> features_v2.parquet {out.shape}')
print(out[['tag_acc','tag_n','tag_novo','acc5','dt_rel','bundle_pos','pos_sessao']].describe().round(3).to_string())
