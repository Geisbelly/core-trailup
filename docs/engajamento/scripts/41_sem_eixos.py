"""ETAPA 2 - cada eixo, medido separadamente nas DUAS bases.

Esta e a etapa que decide o que entra no modelo. A pergunta nao e "qual eixo
tem a maior AUC no EdNet" - e "qual eixo se comporta igual nas duas bases".
Um eixo que funciona numa realidade e nao na outra e propriedade daquela base,
nao achado sobre engajamento.

METRICA: AUC (area sob a ROC). Le-se como "a chance de o eixo ordenar certo um
par de alunos escolhido ao acaso, um que voltou e um que nao". 0,50 e cara ou
coroa. NAO usamos acuracia - ver o porque na etapa 5.

-- PROVENIENCIA ------------------------------------------------------------
Este script e a etapa 2 do pipeline do seminario, em `seminario/scripts/`,
que tem README e diario de decisoes. A copia aqui existe para o modelo
`engajamento` guardar, junto dos demais scripts de auditoria, o codigo que
produziu os numeros do relatorio.

Para RODAR o pipeline na ordem e com as instrucoes, use `seminario/scripts/`.
Aqui os caminhos foram normalizados como nos outros scripts de `docs/` -
ver docs/LEIA-ME_scripts.md.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from comum import SAIDA

EIXOS = ['recencia', 'frequencia', 'profundidade', 'volume']


def ic_auc(y, x, n=400, semente=0):
    """IC95 por bootstrap. Reamostra alunos, nao linhas - aqui da no mesmo,
    porque o painel ja tem uma linha por aluno."""
    rng = np.random.default_rng(semente)
    y, x = np.asarray(y), np.asarray(x)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], x[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


if __name__ == '__main__':
    p = pd.read_parquet(SAIDA / 'painel.parquet')
    coortes = sorted(p.coorte.unique())
    print(f'{"eixo":<14}', end='')
    for c in coortes:
        print(f'{c:>28}', end='')
    print(f'{"replica?":>12}')

    linhas = []
    for eixo in EIXOS:
        print(f'{eixo:<14}', end='')
        aucs = {}
        for c in coortes:
            d = p[(p.coorte == c) & p[eixo].notna()]
            if len(d) == 0:
                aucs[c] = float('nan')
                print(f'{"nao medivel nesta base":>28}', end='')
                continue
            a = roc_auc_score(d.voltou, d[eixo])
            lo, hi = ic_auc(d.voltou, d[eixo])
            aucs[c] = a
            print(f'{a:>12.3f} [{lo:.3f}-{hi:.3f}]', end='')
        vals = [v for v in aucs.values() if v == v]
        if len(vals) < len(coortes):
            print(f'{"NAO DA":>12}')
            linhas.append(dict(eixo=eixo, veredito='NAO DA PARA TESTAR', **aucs))
            continue
        # "Replica" = fica do mesmo lado de 0,50 nas duas, e a distancia ao acaso
        # nao encolhe mais que a metade de uma base para a outra.
        forcas = [abs(v - 0.5) for v in vals]
        mesmo_lado = all(v > 0.5 for v in vals) or all(v < 0.5 for v in vals)
        veredito = ('NAO' if not mesmo_lado else
                    'SIM' if min(forcas) >= 0.5 * max(forcas) else 'FRACO')
        print(f'{veredito:>12}')
        linhas.append(dict(eixo=eixo, veredito=veredito, **aucs))

    pd.DataFrame(linhas).to_csv(SAIDA / '02_eixos.csv', index=False)
    print('\nLEITURA. Um eixo so entra no modelo se as DUAS bases concordarem')
    print('com ele. Tres jeitos de nao entrar:')
    print('  NAO    troca de lado entre as bases -> descreve a base, nao o aluno')
    print('  FRACO  mesmo sentido, mas a forca cai pela metade ou mais')
    print('  NAO DA uma das bases nao tem a informacao necessaria para medir')
    print('O `volume` cai no terceiro caso: o OULAD registra o dia, nao a hora,')
    print('e sem hora nao ha sessao. Nao da para defender no modelo um eixo que')
    print('metade das realidades-alvo nao consegue nem calcular.')
