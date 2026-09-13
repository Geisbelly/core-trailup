"""Roda o M1 na forma que sobreviveu: base do aluno + dificuldade do conteudo.

NAO e deteccao de estado. E previsao de como o aluno vai avaliar o proximo
texto - util para calibrar a escala e para ordenar conteudo, nao para dizer
que o aluno "esta travado".
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
R=pd.read_parquet(f'{HERE}/leituras_rotuladas.parquet')
d=R[R.difficulty.notna()].sort_values(['actor','t0']).copy()
d['y']=(d.difficulty>=4).astype(int)
G=d.y.mean()
# dificuldade do texto: dos OUTROS alunos
bt=d.groupby('assignment').y.agg(['sum','size'])
print(f'{len(d):,} leituras | {d.actor.nunique()} alunos | {d.assignment.nunique()} textos | base {G:.1%}\n')

print('=== escore por aluno, acumulando as proprias respostas (causal) ===')
for actor in ['owxO','BUE4','3Pjt']:
    g=d[d.actor==actor]
    if len(g)<8: continue
    print(f'\nALUNO {actor} — {len(g)} leituras')
    print(f'{"#":>3} {"texto":>6} {"dific.texto":>12} {"base.aluno":>11} {"previsto":>9} │ {"declarou":>9}')
    print('─'*62)
    s=n=0
    for i,(_,r) in enumerate(g.head(10).iterrows(),1):
        p_txt=(bt.loc[r.assignment,'sum']-r.y+G*10)/(bt.loc[r.assignment,'size']-1+10)
        p_alu=(s+G*3)/(n+3)
        p=0.5*(p_txt+p_alu)
        decl=f'{int(r.difficulty)}/5' + (' DIFICIL' if r.y else '')
        print(f'{i:>3} {int(r.assignment):>6} {p_txt:>12.2f} {p_alu:>11.2f} {p:>9.2f} │ {decl:>9}')
        s+=r.y; n+=1
    print(f'    (base do aluno apos {n} leituras: {(s+G*3)/(n+3):.2f} — ele marca dificil em {s}/{n})')

print('\n\n=== para que serve: ordenar conteudo para o aluno certo ===')
d2=d.copy()
d2['p_txt']=d2.assignment.map((bt['sum']+G*10)/(bt['size']+10))
t=d2.groupby('assignment').agg(n=('y','size'),p=('p_txt','first')).query('n>=20').sort_values('p')
print(f'{len(t)} textos com >=20 leituras, ordenados por dificuldade percebida:')
print('  mais faceis :',[f'texto {i} ({v:.0%})' for i,v in t.head(3).p.items()])
print('  mais dificeis:',[f'texto {i} ({v:.0%})' for i,v in t.tail(3).p.items()])
print('\n  -> e isto que o TrailUp pode usar: escolher o proximo conteudo pela')
print('     dificuldade que a TURMA reportou, e calibrar a escala pelo historico do aluno.')
print('     Nenhuma inferencia de estado emocional, nenhuma camera, nenhum toque rastreado.')
