"""Calibracao cruzada e um quarto eixo.

(a) Os cortes absolutos do EdNet funcionam no OULAD?
(b) Cortes por PERCENTIL da propria coorte funcionam nos dois?
(c) TENDENCIA (ultimo terco vs primeiro terco da janela) melhora?
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, logistic_fit, logistic_pred
E=pd.read_parquet(f'{HERE}/eixos.parquet')       # EdNet
O=pd.read_parquet(f'{HERE}/oulad_eixos.parquet') # OULAD
EIXOS=['freq','volume','prof']
CORTES_EDNET={'freq':(0.100,0.233),'volume':(12.5,19.619),'prof':(2.175,2.865)}

print('='*72); print('(a) CORTES ABSOLUTOS DO EDNET APLICADOS AO OULAD'); print('='*72)
for e in EIXOS:
    lo,hi=CORTES_EDNET[e]
    for nome,D in [('EdNet',E),('OULAD',O)]:
        f=np.where(D[e]<=lo,'baixo',np.where(D[e]<=hi,'medio','alto'))
        d=pd.Series(f).value_counts(normalize=True).reindex(['baixo','medio','alto']).fillna(0)
        print(f'  {e:>7} em {nome:>5}: baixo {d["baixo"]:5.1%} | medio {d["medio"]:5.1%} | alto {d["alto"]:5.1%}')
    print()

print('='*72); print('(b) CORTES POR PERCENTIL DA PROPRIA COORTE (tercis)'); print('='*72)
print(f'{"eixo":>7} {"base":>6} {"baixo":>8} {"medio":>8} {"alto":>8} {"spread":>8}')
for nome,D in [('EdNet',E),('OULAD',O)]:
    print(f'  -- {nome} (retencao base {D.ret.mean():.1%}) --')
    for e in EIXOS:
        q=D[e].quantile([1/3,2/3]).values
        f=np.where(D[e]<=q[0],0,np.where(D[e]<=q[1],1,2))
        r=[D.ret[f==k].mean() for k in (0,1,2)]
        print(f'  {e:>7} {D.ret.mean():>6.1%} {r[0]:>8.1%} {r[1]:>8.1%} {r[2]:>8.1%} {r[2]-r[0]:>+8.1%}')

print('\n'+'='*72); print('(c) QUARTO EIXO: TENDENCIA'); print('='*72)
print('  tendencia = atividade no ultimo terco da janela / atividade no primeiro terco')
print('  (calculada nos MESMOS dados de janela 1 - nao usa nada do desfecho)\n')
