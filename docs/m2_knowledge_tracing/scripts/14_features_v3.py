"""Features v3: acumulados recalculados sobre a RESPOSTA FINAL + sinais novos."""
import os, numpy as np, pandas as pd
from collections import defaultdict, deque
HERE=os.path.dirname(os.path.abspath(__file__))
A=0.3; CLIP=600.0; GS=1800.0
BASE='/Users/user/Downloads/Inicio'
q=pd.read_csv(f'{BASE}/EdNet-Contents/contents/questions.csv')
q['qidx']=q.question_id.str[1:].astype(np.int32)
BUN={r.qidx:r.bundle_id for r in q.itertuples()}
df=pd.read_parquet(f'{HERE}/responses_v3.parquet')
print(f'{len(df):,} respostas finais | {df.user.nunique():,} usuarios')
user=df.user.values; part=df.part.values.astype(np.int64); qidx=df.qidx.values
ts=df.ts.values; y=df.correct.values.astype(np.float64); n=len(df)
K=['n_prev','acc_prev','ewma','n_prev_part','acc_prev_part','ewma_part','streak',
   'dt_log','dt_part_log','seen_before','regra_trailup','acc5','acc10',
   'n_sessao','pos_sessao','lat_rel','trocas_prev','lat_med_prev']
c={k:np.zeros(n,np.float32) for k in K}
cur=-1
for i in range(n):
    u=user[i]
    if u!=cur:
        cur=u; cnt=0; hit=0; ew=0.5; st=0.0; last=-1
        pc=defaultdict(int); ph=defaultdict(float); pe={}; pl={}; pm={}; seen=defaultdict(int)
        ult=deque(maxlen=10); lats=deque(maxlen=50); trocas=deque(maxlen=20); sess=0; pos=0
    p=part[i]
    gap=(ts[i]-last)/1000.0 if last>0 else -1
    if last<0 or gap>GS: sess+=1; pos=0
    pos+=1
    c['n_prev'][i]=cnt; c['acc_prev'][i]=hit/cnt if cnt else 0.5; c['ewma'][i]=ew
    k=pc[p]; c['n_prev_part'][i]=k; c['acc_prev_part'][i]=ph[p]/k if k else 0.5
    c['ewma_part'][i]=pe.get(p,0.5); c['streak'][i]=st; c['seen_before'][i]=seen[qidx[i]]
    c['regra_trailup'][i]=pm.get(p,0.5)
    c['dt_log'][i]=np.log1p(min(gap,CLIP)) if gap>0 else -1.0
    lp=pl.get(p,-1); c['dt_part_log'][i]=np.log1p(min((ts[i]-lp)/1000.0,CLIP)) if lp>0 else -1.0
    c['acc5'][i]=float(np.mean(list(ult)[-5:])) if ult else 0.5
    c['acc10'][i]=float(np.mean(ult)) if ult else 0.5
    c['n_sessao'][i]=sess; c['pos_sessao'][i]=pos
    med=float(np.median(lats)) if len(lats)>=5 else np.nan
    c['lat_med_prev'][i]=med if med==med else -1.0
    c['lat_rel'][i]=np.log1p(df.lat_final.values[i])-np.log1p(med) if med==med else 0.0
    c['trocas_prev'][i]=float(np.mean(trocas)) if trocas else 0.0
    cc=y[i]; cnt+=1; hit+=cc; ew=A*cc+(1-A)*ew
    st=(st+1) if cc else (st-1 if st<=0 else -1.0)
    if cc and st<0: st=1.0
    pc[p]+=1; ph[p]+=cc; pe[p]=A*cc+(1-A)*pe.get(p,0.5); pl[p]=ts[i]
    pm[p]=min(0.98,max(0.05,pm.get(p,0.5)+(0.08 if cc else -0.07)))
    seen[qidx[i]]+=1; ult.append(cc); lats.append(df.lat_final.values[i])
    trocas.append(min(df.n_trocas.values[i],5)); last=ts[i]
out=pd.DataFrame({'user':user,'qidx':qidx,'part':part.astype(np.int8),'y':y.astype(np.int8),**c})
for col in ['ntags','diagnosis','lat_1a','lat_final','n_trocas','expl_sec','n_aulas','n_quits','mobile','q_no_bundle']:
    out[col]=df[col].values
out['log_lat']=np.log1p(out.lat_final); out['log_lat1']=np.log1p(out.lat_1a)
out['log_expl']=np.log1p(out.expl_sec)
out.to_parquet(f'{HERE}/features_v3.parquet',index=False)
print(f'-> features_v3.parquet {out.shape}')
print(out[['log_lat','lat_rel','n_trocas','trocas_prev','log_expl']].describe().round(3).to_string())
