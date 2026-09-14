"""ETAPA 7 - segundo modelo: avaliador de resposta discursiva.

PROBLEMA. Corrigir texto aberto e caro. Mandar toda resposta para uma LLM custa
dinheiro e segundos; nao mandar nenhuma significa nao corrigir. A pergunta e se
da para decidir SOZINHO nas pontas obvias e mandar so o meio.

O QUE O MODELO FAZ. Compara o grafo de coocorrencia da resposta com o do
gabarito - quais conceitos aparecem, quais ligacoes entre conceitos, o que
falta. Dez features, todas operacoes de conjunto sobre tokens: nao ha rede
neural, nao ha embedding, nao ha dependencia externa.

A LINHA DE BASE QUE IMPORTA. Em avaliacao automatica de texto, o COMPRIMENTO da
resposta derruba a maioria dos trabalhos - resposta longa costuma ganhar nota
maior. Se contar palavras chega perto, o resto nao esta fazendo nada.

Saida: saida/07_discursivo.csv
"""
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from comum import SAIDA, exige

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trailup_core import pre_avaliacao as pa

AJUDA = ('classEx: dados do artigo de feedback entre pares em macroeconomia.\n'
         'Coloque o arquivo em dados/classex.parquet com as colunas\n'
         '  q (enunciado), gab (gabarito), resp (resposta do aluno),\n'
         '  Color (Pseudonym) (id do aluno), Task (id da tarefa),\n'
         '  Run1/2/3_Content_AI Evaluation (as tres notas de referencia).\n'
         'Licenca CC BY.')
CORTE = 3.5   # "precisa de feedback" = nota de referencia abaixo disto


def carregar(caminho):
    d = pd.read_parquet(exige(caminho, AJUDA))
    runs = [c for c in d.columns if 'Content_AI' in c]
    for c in runs:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['resp', 'gab'] + runs).reset_index(drop=True)
    # O ALVO E A MEDIA DAS TRES EXECUCOES. Uma execucao sozinha carrega o ruido
    # do proprio LLM e subestima qualquer modelo medido contra ela.
    return d, d[runs].mean(axis=1).values, runs


def prever(d):
    """Chama o modulo do jeito documentado: com enunciado E stopwords do corpus.

    Chamar so com o gabarito custa 0,066 de Spearman, em silencio - ver o
    docstring de `preparar`. Foi erro nosso na primeira medicao.
    """
    corpus = (list(d.gab.astype(str).unique()) + list(d.q.astype(str).unique())
              + list(d.resp.astype(str)))
    stop = pa.stopwords(corpus)
    ref = {(q, g): pa.preparar(g, enunciado=q, stop=stop)
           for (q, g), _ in d.groupby([d.q.astype(str), d.gab.astype(str)])}
    saidas = [pa.avaliar(str(r), ref[(str(q), str(g))])
              for r, g, q in zip(d.resp, d.gab, d.q)]
    return np.array([a.nota for a in saidas]), saidas, len(stop), len(ref)


def sobreposicao(r, g):
    a, b = set(str(r).lower().split()), set(str(g).lower().split())
    return len(a & b) / max(len(b), 1)


if __name__ == '__main__':
    d, y, runs = carregar(SAIDA.parent / 'dados' / 'classex.parquet')
    prev, saidas, n_stop, n_ref = prever(d)
    alunos = d['Color (Pseudonym)'].astype(str).values

    print(f'{len(d)} respostas | {d.Task.nunique()} tarefas | {len(np.unique(alunos))} alunos')
    print(f'nota de referencia: media {y.mean():.2f}, desvio {y.std():.2f}')
    print(f'{np.isin(np.round(y), [3, 4]).mean():.0%} das notas sao 3 ou 4 - o rotulo tem pouca')
    print(f'variacao, e isso COMPRIME qualquer correlacao medida contra ele.')
    print(f'stopwords derivadas do corpus: {n_stop} | referencias preparadas: {n_ref}\n')

    # O TETO NAO E 1,00. E o quanto o proprio LLM concorda consigo mesmo.
    tetos = [spearmanr(d[a], d[b]).correlation for a, b in itertools.combinations(runs, 2)]
    teto = float(np.mean(tetos))

    print('CONTRA AS LINHAS DE BASE (Spearman com a nota de referencia)\n')
    print(f'  {"o que e":<44} {"Spearman":>9} {"do teto":>8}')
    linhas = []
    for nome, x in [
        ('contar PALAVRAS da resposta', np.array([len(str(r).split()) for r in d.resp])),
        ('contar CARACTERES da resposta', np.array([len(str(r)) for r in d.resp])),
        ('sobreposicao crua de tokens com o gabarito',
         np.array([sobreposicao(r, g) for r, g in zip(d.resp, d.gab)])),
        ('cobertura do gabarito (1 feature do modulo)',
         np.array([a.cobertura for a in saidas])),
        ('conceitos faltando (1 feature)',
         -np.array([len(a.conceitos_faltando) for a in saidas])),
        ('O MODULO INTEIRO (10 features)', prev),
    ]:
        s = spearmanr(x, y).correlation
        print(f'  {nome:<44} {s:>+9.3f} {s/teto:>7.0%}')
        linhas.append(dict(criterio=nome, spearman=s))
    print(f'  {"TETO: duas execucoes do proprio LLM":<44} {teto:>+9.3f} {1.0:>7.0%}')
    print(f'      (as tres comparacoes: ' + ', '.join(f'{t:.3f}' for t in tetos) + ')')

    qwk = cohen_kappa_score(np.clip(np.round(prev), 1, 5).astype(int),
                            np.clip(np.round(y), 1, 5).astype(int), weights='quadratic')
    print(f'\n  QWK do modulo: {qwk:.3f}')

    # O modulo ganha da propria melhor feature? Teste PAREADO, reamostrando ALUNOS.
    cob = np.array([a.cobertura for a in saidas])
    sm, sc = spearmanr(prev, y).correlation, spearmanr(cob, y).correlation
    uni = np.unique(alunos)
    difs = []
    for i in range(2000):
        sel = np.random.default_rng(i).choice(uni, len(uni))
        m = np.isin(alunos, sel)
        if m.sum() < 50:
            continue
        difs.append(spearmanr(prev[m], y[m]).correlation - spearmanr(cob[m], y[m]).correlation)
    difs = np.array(difs)
    print(f'\n  O modulo inteiro ganha da `cobertura` sozinha?')
    print(f'    {sm:+.3f} contra {sc:+.3f} | diferenca {sm-sc:+.3f}')
    print(f'    IC95 pareado [{np.percentile(difs,2.5):+.3f}, {np.percentile(difs,97.5):+.3f}]'
          f' | positiva em {(difs>0).mean():.0%} das reamostras')

    pd.DataFrame(linhas).to_csv(SAIDA / '07_discursivo.csv', index=False)
    print(f'\n  -> {SAIDA / "07_discursivo.csv"}')
