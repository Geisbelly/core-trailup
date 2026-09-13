"""AUDITORIA: cada numero afirmado nos cabecalhos, reconferido chamando as
FUNCOES DO MODULO - nao reimplementacoes. E o que pega divergencia entre o
que o codigo faz e o que a documentacao diz."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/user/Documents/geis/core-trailup')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, spearman
from trailup_core import dificuldade as DIF, ritmo as RIT, tempo as TMP
from trailup_core import dominio as DOM, chute as CHU, revisao as REV
from trailup_core import discriminacao as DSC, gate as GAT
M2 = HERE.replace('/engaj', '/m2')
FALHAS = []
def check(nome, afirmado, medido, tol):
    ok = abs(afirmado - medido) <= tol
    print(f'  [{"OK " if ok else "FALHA"}] {nome:<52} afirmado {afirmado:>7.3f}  medido {medido:>7.3f}')
    if not ok: FALHAS.append(nome)

rng = np.random.RandomState(7)
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
    columns=['user','qidx','part','correct','ts','lat_final']).sort_values(['user','ts'])
us = d.user.unique(); rng.shuffle(us); tr_u = set(us[:int(.7*len(us))])
d['tr'] = d.user.isin(tr_u)
G = float(d.loc[d.tr,'correct'].mean())
print(f'{len(d):,} respostas | acerto global {G:.4f}\n')

print('=== 1. DIFICULDADE ===')
q = d[d.tr].groupby('qidx').correct.agg(['sum','size'])
qt = d[~d.tr].groupby('qidx').correct.agg(['sum','size'])
J = q.join(qt, lsuffix='_a', rsuffix='_b').dropna()
J = J[(J.size_a >= 50) & (J.size_b >= 50)]
alvo = J.sum_b / J.size_b
cob = []
for s0, n0, a, nb in zip(J.sum_a, J.size_a, alvo, J.size_b):
    f = DIF.estimar(int(s0), int(n0))
    sd_post = (f.maximo - f.minimo) / (2*1.645)
    sd = np.sqrt(sd_post**2 + a*(1-a)/nb)
    cob.append(abs(a - f.taxa) <= 1.645*sd)
check('cobertura do posterior (+ ruido do alvo)', 0.89, float(np.mean(cob)), 0.03)
p1 = np.array(alvo); m30 = rng.binomial(30, np.clip(p1,0,1))/30.0
c = [ (m >= DIF.prever_turma(int(s),int(n),30).minimo) and (m <= DIF.prever_turma(int(s),int(n),30).maximo)
      for s,n,m in zip(J.sum_a.values[:4000], J.size_a.values[:4000], m30[:4000]) ]
check('prever_turma(30) cobre a turma simulada', 0.888, float(np.mean(c)), 0.04)
pr = DIF.derivar_prior((q['sum']/q['size']).values, q['size'].values)
check('prior derivado: forca', 7.0, pr.forca, 1.5)

print('\n=== 2. RITMO e TEMPO ===')
ql = d[d.tr].groupby('qidx').lat_final.agg(['median','size'])
ql = ql[ql['size'] >= 5]
te_l = d[~d.tr & d.qidx.isin(ql.index)]
gl = float(d.loc[d.tr,'lat_final'].median())
yv = np.log1p(te_l.lat_final.values)
med = te_l.qidx.map(ql['median']).values
r2_med = 1 - ((yv-np.log1p(med))**2).sum()/((yv-yv.mean())**2).sum()
check('tempo: R2 da mediana da questao', 0.574, r2_med, 0.02)
lg = d[d.tr].assign(ll=np.log1p(d.lat_final)).groupby('qidx').ll.mean()
p_lg = te_l.qidx.map(lg).values
r2_lg = 1 - ((yv-p_lg)**2).sum()/((yv-yv.mean())**2).sum()
check('tempo: R2 da media do log', 0.578, r2_lg, 0.02)
# ritmo: concordancia entre a classificacao com 5 respostas e com todas
qa = d.groupby('qidx').lat_final.agg(['median','size']); qa = qa[qa['size']>=50]
amostra = d[d.qidx.isin(qa.index)].groupby('qidx').lat_final.apply(lambda s: s.head(5).median())
v_full = [RIT.ritmo(m, 50).valor for m in qa['median']]
v_5 = [RIT.ritmo(m, 5).valor for m in amostra.reindex(qa.index)]
check('ritmo: acerto com 5 respostas', 0.98, float(np.mean([a==b for a,b in zip(v_full,v_5)])), 0.03)

print('\n=== 3. DOMINIO (chamando dominio.dominio) ===')
d['q_dif'] = d.qidx.map((q['sum']+G*20)/(q['size']+20)).fillna(G)
d['k'] = d.groupby(['user','part']).cumcount()
d['ac_top'] = d.groupby(['user','part']).correct.cumsum() - d['correct']
d['kg'] = d.groupby('user').cumcount()
d['ac_glob'] = d.groupby('user').correct.cumsum() - d['correct']
te = d[~d.tr & (d.k > 0) & (d.kg > 0)].copy()
sub = te.sample(200000, random_state=1)
p2 = [DOM.dominio(qd, int(a), int(k), media_global=G).p
      for qd,a,k in zip(sub.q_dif, sub.ac_top, sub.k)]
p3 = [DOM.dominio(qd, int(a), int(k), media_global=G,
                  acertos_totais=int(ag), respostas_totais=int(kg)).p
      for qd,a,k,ag,kg in zip(sub.q_dif, sub.ac_top, sub.k, sub.ac_glob, sub.kg)]
check('dominio: 2 termos', 0.722, roc_auc_score(sub.correct.values, p2), 0.012)
check('dominio: 3 termos (com global)', 0.727, roc_auc_score(sub.correct.values, p3), 0.012)

print('\n=== 4. GATE (chamando gate.risco) ===')
fut = d.groupby(['user','part'], sort=False).correct.transform(
    lambda s: s.shift(-1).rolling(10, min_periods=10).mean().shift(-9))
d['alvo'] = (fut < 0.5).astype(float)
Dg = d[fut.notna() & ~d.tr & (d.k >= 5)].copy()
Sg = Dg.sample(120000, random_state=2)
seqs = {}
for (u,p_), gsub in d[~d.tr].groupby(['user','part']):
    seqs[(u,p_)] = gsub.correct.values
def score(row):
    s = seqs[(row.user, row.part)][:int(row.k)]
    return GAT.risco(s.astype(bool), row.ac_glob/max(row.kg,1),
                     row.q_dif, media_global=G).score
sc = Sg.apply(score, axis=1).values
yg = Sg.alvo.values
a_gate = roc_auc_score(yg, sc)
check('gate: AUC da forma combinada', 0.732, a_gate, 0.02)
k10 = sc >= np.quantile(sc, .90)
check('gate: precisao no ponto de 10%', 0.339, float(yg[k10].mean()), 0.04)
print(f'        (taxa base do alvo no recorte: {yg.mean():.1%})')

print('\n=== 5. REVISAO (chamando revisao.retencao) ===')
d['t_ant'] = d.groupby(['user','qidx']).ts.shift()
d['ac_ant'] = d.groupby(['user','qidx']).correct.shift()
d['kq'] = d.groupby(['user','qidx']).cumcount()
d['n_ac'] = d.groupby(['user','qidx']).correct.cumsum() - d['correct']
R = d[d.t_ant.notna() & ~d.tr].copy()
R['dias'] = (R.ts - R.t_ant)/1000/86400
Sr = R.sample(150000, random_state=3)
pb = [REV.retencao(x, bool(a)) for x,a in zip(Sr.dias, Sr.ac_ant)]
pi = [REV.retencao(x, bool(a), proporcao_acertos=float(n)/max(int(k),1))
      for x,a,n,k in zip(Sr.dias, Sr.ac_ant, Sr.n_ac, Sr.kq)]
check('revisao: binario (so o ultimo)', 0.665, roc_auc_score(Sr.correct.values, pb), 0.02)
check('revisao: interpolado pela proporcao', 0.669, roc_auc_score(Sr.correct.values, pi), 0.02)

print('\n=== 6. CHUTE (chamando chute.foi_chute) ===')
qq = d[d.tr].groupby('qidx').lat_final.agg(['median','size'])
qq = qq[qq['size'] >= 50]
E = d[~d.tr & d.qidx.isin(qq.index)].copy()
E['kq2'] = E.groupby(['user','qidx']).cumcount()
prox = E[E.kq2==1].set_index(['user','qidx']).correct.rename('depois')
P = E[(E.kq2==0)&(E.correct==0)].set_index(['user','qidx']).join(prox, how='inner')
P['med'] = P.index.get_level_values('qidx').map(qq['median'])
P['nq'] = P.index.get_level_values('qidx').map(qq['size'])
ch = [CHU.foi_chute(l, False, m, int(n)) for l,m,n in zip(P.lat_final, P.med, P.nq)]
ch = np.array(ch)
gap = P.depois[~ch].mean() - P.depois[ch].mean()
check('chute: diferenca no reencontro (limiar 0,30)', 0.076, float(gap), 0.02)
print(f'        marcados: {ch.mean():.1%}')

print('\n' + '='*70)
print(f'RESULTADO: {len(FALHAS)} falhas' + (f' -> {FALHAS}' if FALHAS else ' - tudo confere'))
