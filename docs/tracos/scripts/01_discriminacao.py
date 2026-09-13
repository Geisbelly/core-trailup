"""Questao defeituosa: discriminacao (correlacao item-total).

Se os alunos que MAIS acertam no geral erram esta questao, algo esta errado:
gabarito trocado, enunciado ambiguo, ou distrator melhor que a resposta.

Teste que importa: a baixa discriminacao REPLICA numa metade independente de
alunos? Se nao replicar, e ruido - e ruido exibido ao professor vira retrabalho.
"""
import os, numpy as np, pandas as pd
M2='/caminho/para/os/intermediarios'
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_parquet(f'{M2}/responses_v3.parquet',columns=['user','qidx','correct','lat_final'])
df=df.drop_duplicates(['user','qidx'],keep='first')
hab=df.groupby('user').correct.agg(['sum','size'])
hab=hab[hab['size']>=30]
df=df[df.user.isin(hab.index)]
print(f'{len(df):,} respostas | {df.user.nunique():,} alunos com >=30 | {df.qidx.nunique():,} questoes')

# habilidade do aluno EXCLUINDO a questao avaliada (senao a propria questao infla)
s=df.user.map(hab['sum']); n=df.user.map(hab['size'])
df['hab_ex']=(s-df.correct)/(n-1)

rng=np.random.RandomState(20260912)
df['metade']=rng.randint(0,2,len(df))
def disc(g):
    if len(g)<40 or g.correct.nunique()<2: return np.nan
    return np.corrcoef(g.correct,g.hab_ex)[0,1]
print('\ncalculando discriminacao por metade...')
d0=df[df.metade==0].groupby('qidx').apply(disc,include_groups=False)
d1=df[df.metade==1].groupby('qidx').apply(disc,include_groups=False)
D=pd.DataFrame({'A':d0,'B':d1}).dropna()
tot=df.groupby('qidx').agg(n=('correct','size'),acerto=('correct','mean'))
D=D.join(tot)
print(f'{len(D):,} questoes com discriminacao estimavel nas duas metades\n')
print('=== distribuicao da discriminacao ===')
print(f'  media {D.A.mean():.3f} | mediana {D.A.median():.3f} | '
      f'p5 {D.A.quantile(.05):.3f} | p95 {D.A.quantile(.95):.3f}')
for lim,lab in [(-1,'NEGATIVA (aluno bom erra)'),(0.05,'quase nula (<0,05)'),(0.15,'baixa (<0,15)')]:
    m=D.A<lim if lim<0 else D.A<lim
    print(f'  {lab:>28}: {m.mean():>5.1%} ({m.sum():>4} questoes)')

print('\n=== replica? (metade A vs metade B, alunos independentes) ===')
r=np.corrcoef(D.A,D.B)[0,1]
print(f'  correlacao A-B: {r:+.3f}')
for lim in (0.0,0.05,0.10):
    sel=D.A<lim
    if sel.sum()<10: continue
    conf=(D.B[sel]<lim).mean()
    base=(D.B<lim).mean()
    print(f'  marcadas com disc<{lim:.2f} na metade A ({sel.sum():>4}): '
          f'{conf:.0%} confirmam em B | base {base:.0%} | lift {conf/base:.1f}x')

print('\n=== as marcadas sao diferentes? ===')
ruim=D.A<0.05; ok=D.A>=0.15
print(f'  {"":16} {"defeituosa":>12} {"normal":>10}')
for c,lab in [('acerto','taxa de acerto'),('n','respostas')]:
    print(f'  {lab:>16} {D[c][ruim].mean():>12.2f} {D[c][ok].mean():>10.2f}')
lat=df.groupby('qidx').lat_final.median()
D['lat']=lat
print(f'  {"latencia mediana":>16} {D.lat[ruim].median():>12.1f} {D.lat[ok].median():>10.1f}')
D.to_parquet(f'{HERE}/discriminacao.parquet')
print(f'\n-> discriminacao.parquet')
