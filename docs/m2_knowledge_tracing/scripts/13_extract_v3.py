"""Reextracao do stream COMPLETO do KT3.

Corrige e acrescenta:
 - RESPOSTA FINAL: a que vale e a ultima antes do submit (21,7% das linhas da
   v1 eram respostas trocadas contadas como independentes - rotulo contaminado)
 - n_trocas: trocar de alternativa antes de submeter = hesitacao medida
 - elapsed: tempo do enter do bundle ate a resposta (nao existia)
 - contexto: explicacao lida antes, aula assistida, quit recente, plataforma
"""
import glob, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

BASE='/Users/user/Downloads/Inicio'; OUT=os.path.dirname(os.path.abspath(__file__))
N=int(sys.argv[1]) if len(sys.argv)>1 else 120_000
MIN_R=20
q=pd.read_csv(f'{BASE}/EdNet-Contents/contents/questions.csv')
CORR=dict(zip(q.question_id,q.correct_answer)); PART=dict(zip(q.question_id,q.part.astype(np.int8)))
NTAG=dict(zip(q.question_id,q.tags.astype(str).apply(lambda t:0 if t=='-1' else t.count(';')+1)))

def one(path):
    try:
        d=pd.read_csv(path,usecols=['timestamp','action_type','item_id','source','user_answer','platform'])
    except Exception: return None
    if (d.action_type=='respond').sum()<MIN_R: return None
    uid=int(os.path.basename(path)[1:-4])
    recs=[]
    b_enter=None; cur={}         # item_id -> [primeiro_ts, ultimo_ts, ans, n]
    expl_sec=0.0; expl_ts=None; lect=0; quits=0; last_sub=None
    for t in d.itertuples():
        a=t.action_type; it=str(t.item_id); ts=int(t.timestamp)
        if a=='enter':
            if it.startswith('b'): b_enter=ts; cur={}
            elif it.startswith('e'): expl_ts=ts
            elif it.startswith('l'): lect+=1
        elif a=='respond':
            if it in cur: cur[it][1]=ts; cur[it][2]=t.user_answer; cur[it][3]+=1
            else: cur[it]=[ts,ts,t.user_answer,1]
        elif a=='quit':
            quits+=1
            if it.startswith('e') and expl_ts: expl_sec=(ts-expl_ts)/1000.0; expl_ts=None
        elif a=='submit':
            if b_enter is None: cur={}; continue
            for qi,(t0,t1,ans,nr) in cur.items():
                if qi not in CORR: continue
                recs.append((uid,t1,int(qi[1:]),
                    1 if ans==CORR[qi] else 0, PART[qi], NTAG[qi],
                    1 if t.source=='diagnosis' else 0,
                    min((t0-b_enter)/1000.0,600.0),      # ate a 1a resposta
                    min((t1-b_enter)/1000.0,600.0),      # ate a resposta final
                    nr-1,                                 # trocas de alternativa
                    min(expl_sec,900.0), lect, quits,
                    1 if t.platform=='mobile' else 0,
                    len(cur)))
            expl_sec=0.0; lect=0; quits=0; b_enter=None; cur={}
    if len(recs)<MIN_R: return None
    return pd.DataFrame(recs,columns=['user','ts','qidx','correct','part','ntags','diagnosis',
        'lat_1a','lat_final','n_trocas','expl_sec','n_aulas','n_quits','mobile','q_no_bundle'])

if __name__=='__main__':
    fs=sorted(glob.glob(f'{BASE}/EdNet-KT3/KT3/u*.csv'))
    random.Random(20260912).shuffle(fs); fs=fs[:N]
    print(f'lendo {len(fs):,} arquivos...',flush=True)
    parts=[]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i,r in enumerate(ex.map(one,fs,chunksize=200)):
            if r is not None: parts.append(r)
            if (i+1)%30000==0: print(f'  {i+1:,} | usuarios {len(parts):,}',flush=True)
    df=pd.concat(parts,ignore_index=True).sort_values(['user','ts'],kind='stable').reset_index(drop=True)
    for c,t in [('correct',np.int8),('part',np.int8),('ntags',np.int8),('diagnosis',np.int8),
                ('n_trocas',np.int16),('n_aulas',np.int16),('n_quits',np.int16),
                ('mobile',np.int8),('q_no_bundle',np.int8)]:
        df[c]=df[c].astype(t)
    for c in ['lat_1a','lat_final','expl_sec']: df[c]=df[c].astype(np.float32)
    df.to_parquet(f'{OUT}/responses_v3.parquet',index=False)
    print(f'\nusuarios {df.user.nunique():,} | respostas FINAIS {len(df):,} | acuracia {df.correct.mean():.4f}')
    print(f'trocou de alternativa: {(df.n_trocas>0).mean():.1%} das respostas')
    print(f'latencia ate a final: mediana {df.lat_final.median():.1f}s p90 {df.lat_final.quantile(.9):.1f}s')
    print(f'leu explicacao antes: {(df.expl_sec>0).mean():.1%} | mobile {df.mobile.mean():.1%}')
