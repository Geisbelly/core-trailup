"""COMPORTAMENTO APOS O ERRO prediz aprender a questao?

expl_sec[i] = tempo lendo explicacao ANTES da resposta i (acumulado desde o
submit anterior). Entao o que o aluno fez APOS errar a questao i esta na
linha i+1 do mesmo aluno.

  casos    = (aluno, questao) em que ERROU na 1a vez e reencontrou depois
  features = da linha SEGUINTE ao erro
  desfecho = acertou no reencontro

PLACEBO: a explicacao lida ANTES do erro (linha do proprio erro). Se o efeito
for do comportamento pos-erro, o placebo nao pode predizer igual.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson
M2 = HERE.replace('/engaj', '/m2')
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
    columns=['user','qidx','part','correct','ts','lat_final','expl_sec','n_aulas','n_quits','n_trocas'])
d = d.sort_values(['user','ts']).reset_index(drop=True)
print(f'{len(d):,} respostas | {d.user.nunique():,} alunos', flush=True)

for c in ['expl_sec','n_aulas','n_quits','ts','part']:
    d['prox_'+c] = d.groupby('user')[c].shift(-1)
d['tem_prox'] = d.prox_ts.notna()
d['gap_prox'] = (d.prox_ts - d.ts) / 1000.0
d['mesmo_topico'] = (d.prox_part == d.part).astype(float)

d['k'] = d.groupby(['user','qidx']).cumcount()
pri = d[(d.k == 0) & (d.correct == 0) & d.tem_prox].set_index(['user','qidx'])
seg = d[d.k == 1].set_index(['user','qidx'])['correct'].rename('depois')
J = pri.join(seg, how='inner').dropna(subset=['depois'])
q = d.groupby('qidx').correct.mean(); hab = d.groupby('user').correct.mean()
J['q_ac'] = J.index.get_level_values('qidx').map(q)
J['hab'] = J.index.get_level_values('user').map(hab)
print(f'{len(J):,} casos: errou na 1a vez, tem resposta seguinte, e reencontrou\n', flush=True)

J['log_expl_pos'] = np.log1p(J.prox_expl_sec)
J['log_expl_pre'] = np.log1p(J.expl_sec)          # PLACEBO
J['log_gap'] = np.log1p(J.gap_prox.clip(lower=0))
J['abandonou'] = (J.prox_n_quits > 0).astype(float)
J['viu_aula'] = (J.prox_n_aulas > 0).astype(float)

print('=== A. cada sinal sozinho, contra "acerta ao reencontrar" ===')
print(f'  {"sinal":<36} {"AUC":>7}')
for c, lab in [('log_expl_pos','explicacao lida APOS o erro'),
               ('log_expl_pre','explicacao lida ANTES (PLACEBO)'),
               ('viu_aula','assistiu aula logo apos'),
               ('abandonou','abandonou a sessao apos'),
               ('log_gap','tempo ate a proxima resposta'),
               ('mesmo_topico','continuou no mesmo topico'),
               ('q_ac','(controle) facilidade da questao'),
               ('hab','(controle) habilidade do aluno')]:
    v = J[c].values.astype(float); m = np.isfinite(v)
    print(f'  {lab:<36} {roc_auc_score(J.depois.values[m], v[m]):>7.3f}')

print('\n=== B. leu explicacao apos o erro: acerto no reencontro ===')
J['fx'] = pd.cut(J.prox_expl_sec, [-0.01, 0.01, 5, 15, 30, 60, 1e9],
                 labels=['nao leu','<5s','5-15s','15-30s','30-60s','60s+'])
for f, g in J.groupby('fx', observed=True):
    lo, hi = wilson(int(g.depois.sum()), len(g))
    print(f'  {str(f):>8}: acerta {g.depois.mean():>6.1%} [{lo:.1%},{hi:.1%}]  n={len(g):>8,}')

print('\n=== C. sobrevive a controlar dificuldade e habilidade? ===')
J['dq'] = pd.qcut(J.q_ac, 4, labels=False, duplicates='drop')
J['hq'] = pd.qcut(J.hab, 4, labels=False, duplicates='drop')
print('  AUC de "explicacao apos o erro", dentro de cada celula:')
print('  ' + ' ' * 11 + ' '.join(f'  hab q{i+1} ' for i in range(4)))
for dqi in range(4):
    row = []
    for hqi in range(4):
        s = J[(J.dq == dqi) & (J.hq == hqi)]
        row.append(f'{roc_auc_score(s.depois.values, s.log_expl_pos.values):.3f}'
                   if len(s) > 500 and s.depois.nunique() > 1 else '  -  ')
    print(f'  dific q{dqi+1}   ' + ' '.join(f'{r:>8}' for r in row))
J.reset_index()[['user','qidx','depois','log_expl_pos','log_expl_pre','viu_aula',
                 'abandonou','log_gap','mesmo_topico','q_ac','hab']].to_parquet(f'{HERE}/pos_erro.parquet')
print('\n-> pos_erro.parquet')
