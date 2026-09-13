"""Os tres eixos de engajamento no OULAD - segunda realidade.

EdNet: adulto coreano, autoestudo, TOEIC, sem turma.
OULAD: universitario britanico, EAD, com turma, matricula e ABANDONO FORMAL.

Mesmo desenho: eixos nos dias 0-29 do curso, desfecho nos dias 30-59.
Aqui o desfecho tem duas versoes - atividade (comparavel ao EdNet) e
desmatricula formal (mais forte, so existe aqui).
"""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from _shim import roc_auc_score, spearman
O='/Users/user/Downloads/Inicio/open+university+learning+analytics+dataset'
W1,W2=30,60

vle=pd.read_csv(f'{O}/vle.csv',usecols=['id_site','code_module','code_presentation','activity_type'])
# PROFUNDIDADE: material de conteudo (ler/estudar) vs navegacao (clicar para achar)
CONTEUDO={'oucontent','resource','url','forumng','ouwiki','glossary','dataplus','dualpane','htmlactivity'}
vle['conteudo']=vle.activity_type.isin(CONTEUDO).astype(int)
KEY=['code_module','code_presentation','id_site']
mapa=vle.set_index(KEY).conteudo

print('lendo studentVle (10,6 M linhas)...',flush=True)
v=pd.read_csv(f'{O}/studentVle.csv')
v['turma']=v.code_module+'-'+v.code_presentation
v['aluno']=v.turma+'|'+v.id_student.astype(str)
v['conteudo']=pd.MultiIndex.from_frame(v[KEY]).map(mapa).fillna(0).astype(int)
print(f'{len(v):,} cliques | {v.aluno.nunique():,} alunos-turma',flush=True)

reg=pd.read_csv(f'{O}/studentRegistration.csv')
reg['turma']=reg.code_module+'-'+reg.code_presentation
reg['aluno']=reg.turma+'|'+reg.id_student.astype(str)
info=pd.read_csv(f'{O}/studentInfo.csv')
info['turma']=info.code_module+'-'+info.code_presentation
info['aluno']=info.turma+'|'+info.id_student.astype(str)

a=v[(v.date>=0)&(v.date<W1)]
b=v[(v.date>=W1)&(v.date<W2)]
g=a.groupby('aluno')
A=pd.DataFrame({
    'cliques':g.sum_click.sum(),
    'dias_ativos':g.date.nunique(),
    'cliques_conteudo':a.assign(c=a.sum_click*a.conteudo).groupby('aluno').c.sum(),
})
A=A[A.cliques>=30]
A['freq']=A.dias_ativos/W1
A['volume']=A.cliques/A.dias_ativos            # cliques por dia ativo (dia = sessao aqui)
A['prof']=A.cliques_conteudo/A.cliques          # fracao do esforco em conteudo
EIXOS=['freq','volume','prof']

nb=b.groupby('aluno').sum_click.sum()
A['ret']=(A.index.map(nb).fillna(0)>0).astype(int)
A['vol_fut']=np.log1p(A.index.map(nb).fillna(0))
du=reg.set_index('aluno').date_unregistration
A['desmat']=pd.to_numeric(A.index.map(du),errors='coerce')
A['saiu_formal']=((A.desmat>=W1)&(A.desmat<W2)).astype(int)
A['resultado']=A.index.map(info.set_index('aluno').final_result)
print(f'\n{len(A):,} alunos com >=30 cliques nos dias 0-29')
print(f'  retencao (clicou nos dias 30-59): {A.ret.mean():.1%}')
print(f'  desmatricula formal nos dias 30-59: {A.saiu_formal.mean():.1%}')
print(f'  resultado final: {A.resultado.value_counts(normalize=True).round(3).to_dict()}')

print('\n=== distribuicao dos eixos (unidade crua) ===')
for nome,col in [('frequencia (dias ativos em 30)','dias_ativos'),
                 ('volume (cliques por dia ativo)','volume'),
                 ('profundidade (fracao em conteudo)','prof')]:
    q=A[col].quantile([.05,.25,.5,.75,.95])
    print(f'  {nome:<34} p5 {q[.05]:7.2f} | p25 {q[.25]:7.2f} | mediana {q[.5]:7.2f} | p75 {q[.75]:7.2f} | p95 {q[.95]:7.2f}')

print('\n=== correlacao entre os eixos (Spearman) ===')
print(A[EIXOS].rank().corr().round(2).to_string())

print('\n=== AUC contra os dois desfechos ===')
print(f'{"eixo":>10} {"retencao":>10} {"saiu formal":>13}')
for c in EIXOS+['cliques']:
    a1=roc_auc_score(A.ret,A[c]); a2=roc_auc_score(A.saiu_formal,-A[c])
    print(f'{c:>10} {a1:>10.3f} {a2:>13.3f}')
print('  (no desfecho "saiu formal" o sinal esta invertido: AUC calculado sobre -x,')
print('   entao >0,5 significa que MAIS do eixo -> MENOS abandono)')

print('\n=== taxa de retencao por tercil de cada eixo ===')
for e in EIXOS:
    q=A[e].quantile([1/3,2/3]).values
    f=np.where(A[e]<=q[0],'baixo',np.where(A[e]<=q[1],'medio','alto'))
    t=pd.DataFrame({'f':f,'r':A.ret,'s':A.saiu_formal}).groupby('f').agg(
        ret=('r','mean'),saiu=('s','mean'),n=('r','size')).reindex(['baixo','medio','alto'])
    print(f'  {e:>7} (cortes {q[0]:.3f} / {q[1]:.3f}):')
    for i,r in t.iterrows():
        print(f'      {i:>5}: retencao {r.ret:5.1%} | desmatricula {r.saiu:5.1%} | n={int(r.n):,}')
A.to_parquet(f'{HERE}/oulad_eixos.parquet')
print(f'\n-> oulad_eixos.parquet')
