"""ETAPA 3 - a armadilha de juntar as bases cruas.

Decisao do grupo: treinar UM modelo nas duas bases juntas, em vez de um por
base. Antes de fazer isso, um teste que quase nao se ve em tutorial:

    e possivel adivinhar DE QUAL BASE veio a linha, olhando so os eixos?

Se for, o modelo conjunto vai aprender a identificar a base e aplicar a taxa de
retencao dela. Isso sobe a metrica e nao mede engajamento nenhum.

O conserto e padronizar dentro de cada coorte (z = (x - media) / desvio, com
media e desvio DAQUELA base) antes de juntar.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from comum import SAIDA

USADOS = ['recencia', 'frequencia']


def z_por_coorte(d, colunas):
    """Padroniza dentro da coorte. E o passo que impede a adivinhacao."""
    s = d.copy()
    for c in colunas:
        g = s.groupby('coorte')[c]
        s[c] = (s[c] - g.transform('mean')) / g.transform('std').replace(0, 1)
    return s


if __name__ == '__main__':
    p = pd.read_parquet(SAIDA / 'painel.parquet')
    alvo = (p.coorte == p.coorte.iloc[0]).astype(int)

    print('Consigo adivinhar de qual base a linha veio?\n')
    for nome, dados in [('valor absoluto', p), ('z dentro da coorte', z_por_coorte(p, USADOS))]:
        m = LogisticRegression(max_iter=2000)
        pr = cross_val_predict(m, dados[USADOS], alvo, cv=5, method='predict_proba')[:, 1]
        a = roc_auc_score(alvo, pr)
        aviso = '  <- o modelo aprende a BASE, nao o aluno' if a > 0.7 else '  <- indistinguivel'
        print(f'  {nome:<24} AUC {a:.3f}{aviso}')

    print('\nE quanto isso custa na pratica?\n')
    # ATENCAO AO DESENHO DESTE TESTE. A primeira versao treinava numa base e
    # testava na outra, comparando cru contra z. Dava exatamente o mesmo numero
    # nos dois casos, e a razao e aritmetica: padronizar dentro da coorte e uma
    # transformacao MONOTONA dentro de cada coorte, e AUC so olha ordem. O
    # teste nao podia detectar diferenca nenhuma. Ver DIARIO, 13/09.
    #
    # O efeito aparece onde o dano realmente acontece: treinando JUNTO. Ai os
    # coeficientes sao ajustados com as duas bases no mesmo espaco, e no caso
    # cru eles se acomodam a distancia entre as bases em vez do sinal do aluno.
    rng = np.random.default_rng(7)
    alunos = p.aluno.unique()
    treino = set(alunos[rng.random(len(alunos)) < 0.7])
    for nome, dados in [('cru', p), ('z por coorte', z_por_coorte(p, USADOS))]:
        tr = dados[dados.aluno.isin(treino)]
        te = dados[~dados.aluno.isin(treino)]
        m = LogisticRegression(max_iter=2000).fit(tr[USADOS], tr.voltou)
        saida = []
        for c in sorted(p.coorte.unique()):
            s = te[te.coorte == c]
            a = roc_auc_score(s.voltou, m.predict_proba(s[USADOS])[:, 1])
            saida.append(f'{c} {a:.3f}')
        pesos = ', '.join(f'{n} {v:+.3f}' for n, v in zip(USADOS, m.coef_[0]))
        print(f'  treinado NAS DUAS ({nome:<12}) -> ' + ' | '.join(saida))
        print(f'      pesos aprendidos: {pesos}')
    print('\nLEITURA, E ELA E MAIS CONTIDA DO QUE ESPERAVAMOS.')
    print('A adivinhacao da base e real (AUC 0,92) e o efeito dela aparece nos')
    print('PESOS: no cru eles ficam ~15x maiores, porque a unica forma de a')
    print('logistica separar duas bases com escalas diferentes e esticar o')
    print('coeficiente ate a distancia entre elas virar decisao.')
    print()
    print('MAS o custo em AUC, NESTE painel, e pequeno - uma ou tres milesimas.')
    print('Nao vamos afirmar um estrago que nao medimos. O que se defende aqui e')
    print('que o modelo cru chega ao resultado por um caminho que nao e o')
    print('declarado, e que peso de 9,5 numa feature de duas nao e interpretavel.')
    print('Padronizar custa uma linha e tira as duas objecoes.')
