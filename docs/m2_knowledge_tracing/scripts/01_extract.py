"""EdNet KT3 -> parquet compacto de respostas (action_type == 'respond').

Mantem apenas o que M2 precisa: quem, quando, qual questao, acertou, qual part.
Nada de user_answer bruto depois do join - so o booleano.
"""
import glob, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

BASE = '/caminho/para/os/datasets'
OUT = os.path.dirname(os.path.abspath(__file__))
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 120_000
MIN_RESPONDS = 20

q = pd.read_csv(f'{BASE}/EdNet-Contents/contents/questions.csv')
q['qidx'] = q.question_id.str[1:].astype(np.int32)
CORRECT = dict(zip(q.question_id, q.correct_answer))
PART = dict(zip(q.question_id, q.part.astype(np.int8)))
NTAGS = dict(zip(q.question_id, q.tags.astype(str).apply(lambda t: 0 if t == '-1' else t.count(';') + 1)))

def one(path):
    try:
        d = pd.read_csv(path, usecols=['timestamp', 'action_type', 'item_id', 'user_answer', 'source'])
    except Exception:
        return None
    d = d[d.action_type == 'respond']
    if len(d) < MIN_RESPONDS:
        return None
    d = d[d.item_id.isin(CORRECT)]
    if len(d) < MIN_RESPONDS:
        return None
    uid = int(os.path.basename(path)[1:-4])
    return pd.DataFrame({
        'user': np.int32(uid),
        'ts': d.timestamp.astype(np.int64).values,
        'qidx': d.item_id.str[1:].astype(np.int32).values,
        'correct': (d.user_answer.values == d.item_id.map(CORRECT).values).astype(np.int8),
        'part': d.item_id.map(PART).astype(np.int8).values,
        'ntags': d.item_id.map(NTAGS).astype(np.int8).values,
        'diagnosis': (d.source.values == 'diagnosis').astype(np.int8),
    })

if __name__ == '__main__':
    files = sorted(glob.glob(f'{BASE}/EdNet-KT3/KT3/u*.csv'))
    random.Random(20260912).shuffle(files)
    files = files[:N_USERS]
    print(f'lendo {len(files)} arquivos...', flush=True)
    parts = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=200)):
            if r is not None:
                parts.append(r)
            if (i + 1) % 20000 == 0:
                print(f'  {i+1}/{len(files)} | usuarios mantidos: {len(parts)}', flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(['user', 'ts'], kind='stable').reset_index(drop=True)
    df.to_parquet(f'{OUT}/responses.parquet', index=False)
    print(f'\nusuarios: {df.user.nunique():,} | respostas: {len(df):,} | acuracia global: {df.correct.mean():.4f}')
    print(f'respostas por usuario: mediana {df.groupby("user").size().median():.0f}, media {df.groupby("user").size().mean():.0f}')
    print(df.part.value_counts().sort_index().to_dict())
