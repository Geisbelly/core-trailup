"""Com n respostas, da para atribuir o grupo certo?

Verdade = grupo do k-means usando TODAS as respostas da questao.
Estimativa = perfil calculado com as primeiras n respostas, atribuido ao
centroide mais proximo. Mesmo protocolo dos experimentos anteriores.
"""
import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
M2='/private/tmp/claude-502/-Users-user-Downloads-Inicio/64768691-f3b8-40d6-8ff0-374843d9d6c4/scratchpad/m2'
HERE=os.path.dirname(os.path.abspath(__file__))
K=pickle.load(open(f'{HERE}/kmeans.pkl','rb'))
km,sc,F=K['kmeans'],K['scaler'],K['feats']
qa=pd.read_parquet(f'{HERE}/questoes_agrupadas.parquet').set_index('qidx')
df=pd.read_parquet(f'{M2}/responses_v3.parquet').sort_values('ts')
df=df[df.qidx.isin(qa.index)]
NOMES={0:'facil/rapida',1:'dificil/trabalhosa',2:'dificil/rapida',3:'facil/trabalhosa'}
ordem=[0,2,3,1]
def perfil(g):
    return pd.DataFrame({'acerto':g.correct.mean(),'lat_med':np.log1p(g.lat_final.median()),
                         'trocas':(g.n_trocas>0).mean(),'lat_1a':np.log1p(g.lat_1a.median())},index=[0])
print('=== acerto do grupo por n de respostas ===')
print(f"{'n':>5} {'questoes':>9} {'acerto':>8}   por grupo (verdadeiro -> acerto)")
print('-'*76)
res={}
for n in (10,21,30,50,100,200):
    sub=df.groupby('qidx').head(n)
    p=sub.groupby('qidx').agg(acerto=('correct','mean'),lat_med=('lat_final','median'),
        trocas=('n_trocas',lambda s:(s>0).mean()),lat_1a=('lat_1a','median'))
    cnt=sub.groupby('qidx').size()
    p=p[cnt>=n]                                # so questoes que de fato tem n
    X=p.copy(); X['lat_med']=np.log1p(X.lat_med); X['lat_1a']=np.log1p(X.lat_1a)
    pred=km.predict(sc.transform(X[F]))
    verd=qa.loc[p.index,'grupo'].values
    acc=accuracy_score(verd,pred); res[n]=(verd,pred)
    porgrupo=' '.join(f'{NOMES[g][:12]}:{(pred[verd==g]==g).mean():.0%}' for g in ordem)
    print(f'{n:>5} {len(p):>9,} {acc:>7.0%}   {porgrupo}')
print('\n=== matriz de confusao com n=30 ===')
verd,pred=res[30]
cm=pd.DataFrame(confusion_matrix(verd,pred,labels=ordem),
                index=[f'real {NOMES[g]}' for g in ordem],
                columns=[f'prev {NOMES[g][:10]}' for g in ordem])
print(cm.to_string())
print('\n=== o erro e entre grupos VIZINHOS? ===')
eixo={0:(1,1),2:(0,1),3:(1,0),1:(0,0)}   # (acerto alto?, rapido?)
verd,pred=res[30]
ac_ok=np.array([eixo[v][0]==eixo[p][0] for v,p in zip(verd,pred)])
tp_ok=np.array([eixo[v][1]==eixo[p][1] for v,p in zip(verd,pred)])
print(f'  acerta o eixo ACERTO (facil x dificil): {ac_ok.mean():.0%}')
print(f'  acerta o eixo TEMPO  (rapida x lenta) : {tp_ok.mean():.0%}')
print(f'  erra os dois eixos ao mesmo tempo     : {(~ac_ok & ~tp_ok).mean():.1%}')
