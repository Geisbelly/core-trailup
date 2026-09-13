"""O teste que decide: o empenho prediz APRENDIZADO, onde o engajamento nao prediz?

Dois desfechos, os dois independentes do que entrou no empenho:
  (a) no reencontro da MESMA questao, o aluno acerta? (aprendizado local)
  (b) ganho de acerto entre a primeira e a segunda metade das respostas
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, spearman
A=pd.read_parquet(f'{HERE}/empenho.parquet')
R=pd.read_parquet(f'{HERE}/empenho_resp.parquet').sort_values(['user','qidx','ts'])
def wil(k,n,z=1.96):
    if n==0: return (float('nan'),)*2
    p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd; return c-h,c+h

print('='*74); print('(a) REENCONTRO: empenho na 1a vez prediz acertar na 2a?'); print('='*74)
R['k']=R.groupby(['user','qidx']).cumcount()
pri=R[R.k==0].set_index(['user','qidx'])[['correct','empenho_bruto','q_dif']]
seg=R[R.k==1].set_index(['user','qidx'])['correct'].rename('depois')
J=pri.join(seg,how='inner').dropna()
print(f'{len(J):,} reencontros')
# so quem ERROU na primeira: aprendeu depois?
E=J[J.correct==0].copy()
print(f'{len(E):,} casos em que errou na primeira vez')
E['fx']=pd.qcut(E.empenho_bruto,5,labels=False,duplicates='drop')
print(f'\n  {"quintil de empenho na 1a vez":<30} {"acerta ao reencontrar":>22} {"n":>9}')
for q_ in sorted(E.fx.dropna().unique()):
    s=E.depois[E.fx==q_]; lo,hi=wil(int(s.sum()),len(s))
    print(f'  {"q"+str(int(q_)+1):<30} {s.mean():>10.1%} [{lo:.1%},{hi:.1%}] {len(s):>9,}')
print(f'  AUC do empenho para acertar no reencontro: {roc_auc_score(E.depois.values,E.empenho_bruto.values):.3f}')
print('  (controle - dificuldade da questao: %.3f)'%roc_auc_score(E.depois.values,-E.q_dif.values))
# dentro de faixa de dificuldade, para nao confundir com "questao facil"
E['dq']=pd.qcut(E.q_dif,4,labels=False,duplicates='drop')
print('\n  dentro de cada quartil de dificuldade da questao:')
for q_ in sorted(E.dq.dropna().unique()):
    s=E[E.dq==q_]
    print(f'    dificuldade q{int(q_)+1}: AUC {roc_auc_score(s.depois.values,s.empenho_bruto.values):.3f} | n={len(s):,}')

print('\n'+'='*74); print('(b) GANHO DE ACERTO entre a 1a e a 2a metade das respostas'); print('='*74)
R2=R.sort_values(['user','ts']); R2['meta']=R2.groupby('user').cumcount()/R2.groupby('user').user.transform('size')
g1=R2[R2.meta<0.5].groupby('user').correct.mean(); g2=R2[R2.meta>=0.5].groupby('user').correct.mean()
G=(g2-g1).rename('ganho').to_frame().join(A,how='inner').dropna(subset=['ganho','empenho'])
print(f'{len(G):,} alunos | ganho medio {G.ganho.mean():+.4f}')
print(f'  {"medida":<22} {"Spearman com o ganho":>22}')
for c in ['empenho','acerto','tempo_estudo','sessoes','trocas','expl','acerto_var','n']:
    print(f'  {c:<22} {spearman(G[c].values,G.ganho.values):>+22.3f}')
print('\n  (referencia: no engajamento, todos os eixos ficaram entre -0,04 e +0,03)')
print('\n  controlando pelo acerto inicial (residuo do ganho sobre g1):')
g1a=G.index.map(g1); 
import numpy as _n
Xc=_n.c_[_n.ones(len(G)),_n.asarray(g1a,dtype=float)]
b,*_=_n.linalg.lstsq(Xc,G.ganho.values,rcond=None)
res=G.ganho.values-Xc@b
for c in ['empenho','tempo_estudo','trocas','expl']:
    print(f'    {c:<20} {spearman(G[c].values,res):>+8.3f}')
