"""Features causais para M2: no passo i so entra o que aconteceu antes de i.

`part` (1-7 do TOEIC) faz o papel de topico: e nele que o TrailUp mantem
`aluno_topico_dominio`, entao todo acumulado existe em dois niveis - aluno e
aluno x part.
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.3          # ewma
DT_CLIP = 600.0      # segundos; acima disso e pausa, nao tempo de resposta

df = pd.read_parquet(f'{HERE}/responses.parquet')
print(f'entrada: {len(df):,} respostas de {df.user.nunique():,} usuarios')

user = df.user.values
part = df.part.values.astype(np.int64)
qidx = df.qidx.values
ts = df.ts.values
y = df.correct.values.astype(np.float64)
n = len(df)

# --- passada unica: tudo que e recursivo sai daqui
n_prev      = np.zeros(n, np.float32); acc_prev    = np.zeros(n, np.float32)
ewma        = np.zeros(n, np.float32); n_prev_p    = np.zeros(n, np.float32)
acc_prev_p  = np.zeros(n, np.float32); ewma_p      = np.zeros(n, np.float32)
streak      = np.zeros(n, np.float32); dt_log      = np.zeros(n, np.float32)
dt_p_log    = np.zeros(n, np.float32); seen_before = np.zeros(n, np.float32)
regra       = np.zeros(n, np.float32)   # baseline A: dominio da regra do TrailUp

cur_user = -1
for i in range(n):
    u = user[i]
    if u != cur_user:                      # troca de usuario: zera tudo
        cur_user = u
        cnt = 0; hit = 0; ew = 0.5; strk = 0.0; last_ts = -1
        p_cnt = {}; p_hit = {}; p_ew = {}; p_last = {}; p_mastery = {}; seen = {}
    p = part[i]
    n_prev[i] = cnt
    acc_prev[i] = hit / cnt if cnt else 0.5
    ewma[i] = ew
    pc = p_cnt.get(p, 0)
    n_prev_p[i] = pc
    acc_prev_p[i] = p_hit.get(p, 0) / pc if pc else 0.5
    ewma_p[i] = p_ew.get(p, 0.5)
    streak[i] = strk
    seen_before[i] = seen.get(qidx[i], 0)
    regra[i] = p_mastery.get(p, 0.5)       # dominio ANTES desta resposta
    dt_log[i] = np.log1p(min((ts[i] - last_ts) / 1000.0, DT_CLIP)) if last_ts > 0 else -1.0
    lp = p_last.get(p, -1)
    dt_p_log[i] = np.log1p(min((ts[i] - lp) / 1000.0, DT_CLIP)) if lp > 0 else -1.0

    c = y[i]                                # atualiza estado DEPOIS de registrar
    cnt += 1; hit += c
    ew = ALPHA * c + (1 - ALPHA) * ew
    strk = (strk + 1) if c else (strk - 1) if strk <= 0 else -1.0
    if c and strk < 0: strk = 1.0
    p_cnt[p] = pc + 1; p_hit[p] = p_hit.get(p, 0) + c
    p_ew[p] = ALPHA * c + (1 - ALPHA) * p_ew.get(p, 0.5)
    p_last[p] = ts[i]; last_ts = ts[i]
    seen[qidx[i]] = seen.get(qidx[i], 0) + 1
    # regra do TrailUp: base +0.08 por acerto, -0.07 por erro, clamp [0.05, 0.98]
    p_mastery[p] = min(0.98, max(0.05, p_mastery.get(p, 0.5) + (0.08 if c else -0.07)))

out = pd.DataFrame({
    'user': user, 'qidx': qidx, 'part': part.astype(np.int8), 'y': y.astype(np.int8),
    'n_prev': n_prev, 'acc_prev': acc_prev, 'ewma': ewma,
    'n_prev_part': n_prev_p, 'acc_prev_part': acc_prev_p, 'ewma_part': ewma_p,
    'streak': streak, 'dt_log': dt_log, 'dt_part_log': dt_p_log,
    'seen_before': seen_before, 'ntags': df.ntags.values, 'diagnosis': df.diagnosis.values,
    'regra_trailup': regra,
})
out.to_parquet(f'{HERE}/features.parquet', index=False)
print(f'saida: {out.shape} -> features.parquet')
print(out[['n_prev','acc_prev','ewma','streak','dt_log','regra_trailup']].describe().round(3).to_string())
