"""(6) O avaliador serve como PRE-FILTRO? e (8) o texto prediz dificuldade?"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_parquet(f'{HERE}/classex_pred.parquet')
y=d.nota.values; p=d.pred.values
print('=== (6) pre-filtro: quando da para NAO chamar a LLM? ===')
print('    "adequada" = nota >= 3,5 (60% da base)\n')
alvo=(y>=3.5).astype(int)
print(f"{'regra':>34} {'dispensa LLM em':>16} {'precisao':>10} {'erro':>8}")
print('-'*72)
for q in (0.5,0.6,0.7,0.8,0.9):
    lim=np.quantile(p,q); m=p>=lim
    print(f'{f"prev >= percentil {q:.0%} ({lim:.2f})":>34} {m.mean():>15.0%} {alvo[m].mean():>10.1%} {1-alvo[m].mean():>8.1%}')
print('\n    e no outro extremo - claramente precisa de feedback:')
for q in (0.1,0.2,0.3):
    lim=np.quantile(p,q); m=p<=lim
    print(f'{f"prev <= percentil {q:.0%} ({lim:.2f})":>34} {m.mean():>15.0%} {1-alvo[m].mean():>10.1%} (precisa msm)')

print('\n\n=== (8) o TEXTO do enunciado prediz a dificuldade da questao? ===')
print('    a lacuna que travou o cold start: questao nova, zero respostas\n')
q=d.groupby('Task').agg(nota=('nota','mean'),n=('nota','size'),
                        enun=('q','first'),gab=('gab','first'))
print(f'  {len(q)} tarefas | nota media por tarefa: {q.nota.round(2).to_dict()}')
print(f'  amplitude: {q.nota.min():.2f} a {q.nota.max():.2f}\n')
# features do enunciado e do gabarito (sem ver resposta nenhuma)
import re
TOK=re.compile(r'[A-Za-zÄÖÜäöüß]{3,}')
def f_txt(t):
    w=TOK.findall(str(t)); 
    return {'n_pal':len(w),'n_unic':len(set(w)),'tam_med':np.mean([len(x) for x in w]) if w else 0,
            'ttr':len(set(w))/max(len(w),1),'n_frases':str(t).count('.')+str(t).count('?')}
linhas=[]
for r in q.itertuples():
    lin={f'q_{k}':v for k,v in f_txt(r.enun).items()}
    lin.update({f'g_{k}':v for k,v in f_txt(r.gab).items()})
    linhas.append(lin)
X=pd.DataFrame(linhas,index=q.index)
print('  correlacao de cada feature do TEXTO com a nota media da tarefa:')
for c in X.columns:
    r=spearmanr(X[c],q.nota).statistic
    print(f'    {c:>12}: {r:+.3f}')
print(f'\n  -> com apenas {len(q)} tarefas nao da para treinar modelo; o que da para')
print('     dizer e se ALGUMA feature de texto tem relacao monotonica forte.')
mx=max(abs(spearmanr(X[c],q.nota).statistic) for c in X.columns)
print(f'     maior |correlacao| encontrada: {mx:.3f}')
print(f'     {"promissor - testar com mais tarefas" if mx>0.6 else "fraco neste recorte"}')
