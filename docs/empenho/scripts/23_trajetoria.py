"""EMPENHO COMO TRAJETORIA DO PROPRIO ALUNO - caiu / constante / subiu.

Diferenca crucial em relacao ao teste anterior: aquele comparava ALUNOS entre
si (nivel de empenho), e o nivel nao predizia nada. Este compara o aluno com
ELE MESMO ao longo do tempo. Cada aluno e seu proprio controle, o que elimina
o confundimento de estilo que derrubou a "base pessoal" do M1.

Desenho:
  janela = blocos de 25 respostas consecutivas do aluno
  empenho_bloco = residuo medio de log(tempo) sobre o esperado da questao
  trajetoria = inclinacao do empenho nos blocos ANTERIORES (normalizada pelo
               desvio do proprio aluno -> "caiu", "constante", "subiu")
  desfechos (todos no FUTURO do que gerou a trajetoria):
    (a) o acerto do aluno cai no bloco seguinte?
    (b) o aluno para de responder (nao ha bloco seguinte)?
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from _shim import roc_auc_score, wilson, spearman
M2 = HERE.replace('/engaj', '/m2')
BLOCO = 25
d = pd.read_parquet(f'{M2}/responses_v3.parquet',
                    columns=['user','qidx','correct','ts','lat_final','expl_sec'])
d = d.sort_values(['user','ts']).reset_index(drop=True)

# --- residuo de esforco por resposta (mesma definicao do relatorio de empenho) ---
q = d.groupby('qidx').agg(qlat=('lat_final','median'), qac=('correct','mean'), qn=('correct','size'))
q = q[q.qn >= 50]; d = d[d.qidx.isin(q.index)].copy()
d['log_qlat'] = np.log1p(d.qidx.map(q.qlat)); d['q_dif'] = 1 - d.qidx.map(q.qac)
d['hab'] = d.user.map(d.groupby('user').correct.mean())
X = np.c_[np.ones(len(d)), d.log_qlat.values, d.q_dif.values, d.hab.values]
beta, *_ = np.linalg.lstsq(X, np.log1p(d.lat_final.values), rcond=None)
d['esf'] = np.log1p(d.lat_final.values) - X @ beta
print(f'{len(d):,} respostas | {d.user.nunique():,} alunos', flush=True)

# --- blocos ---
d['b'] = d.groupby('user').cumcount() // BLOCO
B = d.groupby(['user','b']).agg(esf=('esf','mean'), ac=('correct','mean'),
                                expl=('expl_sec','median'), n=('correct','size')).reset_index()
B = B[B.n == BLOCO]
B['nb'] = B.groupby('user').b.transform('size')
B = B[B.nb >= 5]
print(f'{len(B):,} blocos de {BLOCO} respostas | {B.user.nunique():,} alunos com >=5 blocos\n', flush=True)

# --- trajetoria: inclinacao dos 3 blocos anteriores, normalizada pelo proprio aluno ---
B = B.sort_values(['user','b']).reset_index(drop=True)
def incl(s):
    out = np.full(len(s), np.nan)
    v = s.values
    for i in range(3, len(v)):
        y = v[i-3:i]
        out[i] = np.polyfit(range(3), y, 1)[0]
    return pd.Series(out, index=s.index)
B['slope'] = B.groupby('user').esf.apply(incl).reset_index(level=0, drop=True)
B['dp_aluno'] = B.groupby('user').esf.transform('std')
B['traj'] = B.slope / B.dp_aluno.replace(0, np.nan)
B['ac_ant'] = B.groupby('user').ac.shift(1)
B['ac_prox'] = B.groupby('user').ac.shift(-1)
B['tem_prox'] = B.ac_prox.notna()
B['caiu_ac'] = ((B.ac_prox - B.ac) < 0).astype(float)
B['parou'] = (~B.tem_prox).astype(float)
V = B.dropna(subset=['traj']).copy()
print(f'{len(V):,} blocos com trajetoria calculavel\n')

# --- rotulo em 3 faixas, pelos tercis da distribuicao de trajetorias ---
lo, hi = V.traj.quantile([1/3, 2/3]).values
V['rot'] = np.where(V.traj <= lo, 'caiu', np.where(V.traj <= hi, 'constante', 'subiu'))
print(f'cortes da trajetoria: {lo:+.3f} / {hi:+.3f}\n')

print('=== (a) DESFECHO: o acerto cai no bloco seguinte? ===')
W = V[V.tem_prox]
print(f'  {len(W):,} blocos com bloco seguinte | base {W.caiu_ac.mean():.1%}')
for r in ['caiu','constante','subiu']:
    s = W.caiu_ac[W.rot == r]; l, h = wilson(int(s.sum()), len(s))
    print(f'    empenho {r:<10} acerto cai em {s.mean():>6.1%} [{l:.1%},{h:.1%}]  n={len(s):>8,}')
print(f'  AUC da trajetoria continua: {roc_auc_score(W.caiu_ac.values, -W.traj.values):.3f}')

print('\n=== (b) DESFECHO: o aluno para de responder? ===')
print(f'  base {V.parou.mean():.1%}')
for r in ['caiu','constante','subiu']:
    s = V.parou[V.rot == r]; l, h = wilson(int(s.sum()), len(s))
    print(f'    empenho {r:<10} para em {s.mean():>6.1%} [{l:.1%},{h:.1%}]  n={len(s):>8,}')
print(f'  AUC da trajetoria continua: {roc_auc_score(V.parou.values, -V.traj.values):.3f}')

print('\n=== CONTROLE: e so o nivel de empenho, ou e mesmo a mudanca? ===')
print(f'  {"medida":<34} {"AUC p/ acerto cair":>19} {"AUC p/ parar":>14}')
for c, lab, sinal in [('traj','trajetoria (mudanca)',-1),
                      ('esf','nivel de empenho no bloco',-1),
                      ('ac','acerto no bloco',-1),
                      ('expl','tempo na explicacao',-1)]:
    v = sinal * W[c].values.astype(float); m = np.isfinite(v)
    v2 = sinal * V[c].values.astype(float); m2 = np.isfinite(v2)
    print(f'  {lab:<34} {roc_auc_score(W.caiu_ac.values[m], v[m]):>19.3f} '
          f'{roc_auc_score(V.parou.values[m2], v2[m2]):>14.3f}')

print('\n=== O sinal e DENTRO do aluno? (AUC intra-aluno) ===')
def intra(df, alvo, score):
    g = df.dropna(subset=[alvo, score])
    c = d_ = 0
    for _, s in g.groupby('user'):
        y = s[alvo].values; x = s[score].values
        p = y == 1; n = y == 0
        if p.sum() == 0 or n.sum() == 0: continue
        c += (x[p][:, None] > x[n][None, :]).sum() + 0.5 * (x[p][:, None] == x[n][None, :]).sum()
        d_ += p.sum() * n.sum()
    return c / d_ if d_ else float('nan')
print(f'  trajetoria -> acerto cai : {intra(W, "caiu_ac", "traj"):.3f}  (invertido: quanto MENOR a traj, mais cai)')
print(f'  nivel      -> acerto cai : {intra(W, "caiu_ac", "esf"):.3f}')
V.to_parquet(f'{HERE}/trajetoria.parquet')
print('\n-> trajetoria.parquet')
