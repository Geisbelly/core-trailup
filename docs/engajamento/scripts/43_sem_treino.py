"""ETAPA 4 - o modelo, e a razao de ele ser medido em 12 particoes.

Um numero de uma particao so nao e o nivel do modelo, e o nivel daquela
particao. Nesta etapa treinamos 12 vezes, com 12 divisoes diferentes de alunos,
e reportamos media +- desvio. Foi assim que descobrimos que a primeira AUC que
tinhamos anotado estava quase dois desvios acima da media - era uma divisao
sortuda, nao um modelo melhor.

LINHA DE BASE: o eixo isolado que vai melhor. O modelo precisa ganhar DELE,
nao de 0,50 - qualquer coisa ganha de 0,50.

-- PROVENIENCIA ------------------------------------------------------------
Este script e a etapa 4 do pipeline do seminario, em `seminario/scripts/`,
que tem README e diario de decisoes. A copia aqui existe para o modelo
`engajamento` guardar, junto dos demais scripts de auditoria, o codigo que
produziu os numeros do relatorio.

Para RODAR o pipeline na ordem e com as instrucoes, use `seminario/scripts/`.
Aqui os caminhos foram normalizados como nos outros scripts de `docs/` -
ver docs/LEIA-ME_scripts.md.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from comum import SAIDA, SEMENTES
from importlib import import_module

z_por_coorte = import_module('03_vazamento').z_por_coorte
USADOS = ['recencia', 'frequencia']


def uma_particao(p, semente):
    rng = np.random.default_rng(semente)
    alunos = p.aluno.unique()
    treino = set(alunos[rng.random(len(alunos)) < 0.7])
    d = z_por_coorte(p, USADOS)
    tr, te = d[d.aluno.isin(treino)], d[~d.aluno.isin(treino)]
    m = LogisticRegression(max_iter=2000).fit(tr[USADOS], tr.voltou)
    fora = {}
    for c in sorted(p.coorte.unique()):
        s = te[te.coorte == c]
        fora[c] = roc_auc_score(s.voltou, m.predict_proba(s[USADOS])[:, 1])
    return fora, m.coef_[0], m.intercept_[0]


if __name__ == '__main__':
    p = pd.read_parquet(SAIDA / 'painel.parquet')
    coortes = sorted(p.coorte.unique())

    print('LINHA DE BASE - melhor eixo isolado, sem modelo nenhum')
    base = {}
    for c in coortes:
        d = p[p.coorte == c]
        base[c] = max(roc_auc_score(d.voltou, d[e]) for e in USADOS)
        print(f'  {c:<8} AUC {base[c]:.3f}')

    print(f'\nMODELO - 12 particoes independentes por aluno\n')
    print(f'{"semente":>8}' + ''.join(f'{c:>10}' for c in coortes))
    res = {c: [] for c in coortes}
    pesos, intercs = [], []
    for s in SEMENTES:
        fora, coef, interc = uma_particao(p, s)
        pesos.append(coef); intercs.append(interc)
        print(f'{s:>8}' + ''.join(f'{fora[c]:>10.3f}' for c in coortes))
        for c in coortes:
            res[c].append(fora[c])

    print()
    for c in coortes:
        v = np.array(res[c])
        print(f'  {c:<8} media {v.mean():.4f} | desvio {v.std(ddof=1):.4f} | '
              f'amplitude {v.max()-v.min():.4f} | ganho sobre a linha de base '
              f'{v.mean()-base[c]:+.4f}')

    P = np.array(pesos)
    print(f'\n  pesos medios: ' + ', '.join(
        f'{n} {P[:, i].mean():+.4f} +- {P[:, i].std(ddof=1):.4f}'
        for i, n in enumerate(USADOS)))
    print(f'  intercepto {np.mean(intercs):+.4f} +- {np.std(intercs, ddof=1):.4f}')
    print(f'  razao recencia/frequencia: {P[:, 0].mean()/P[:, 1].mean():.1f}x')

    json.dump({'features': USADOS,
               'pesos': [float(P[:, i].mean()) for i in range(len(USADOS))],
               'intercepto': float(np.mean(intercs)),
               'auc': {c: [float(np.mean(res[c])), float(np.std(res[c], ddof=1))]
                       for c in coortes},
               'linha_de_base': {c: float(base[c]) for c in coortes}},
              open(SAIDA / '04_modelo.json', 'w'), indent=2)
    print(f'\n  -> {SAIDA / "04_modelo.json"}')
