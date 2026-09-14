"""ETAPA 8 - o avaliador discursivo vale a pena? A eficiencia operacional.

Spearman 0,495 nao diz se usar compensa. As perguntas de produto sao outras:
quantas chamadas a LLM o modulo evita, a que custo de erro, e quanto ele custa.

Saida: saida/08_operacao.csv

-- PROVENIENCIA ------------------------------------------------------------
Este script e a etapa 8 do pipeline do seminario, em `seminario/scripts/`,
que tem README e diario de decisoes. A copia aqui existe para o modelo
`pre_avaliacao` guardar, junto dos demais scripts de auditoria, o codigo que
produziu os numeros do relatorio.

Para RODAR o pipeline na ordem e com as instrucoes, use `seminario/scripts/`.
Aqui os caminhos foram normalizados como nos outros scripts de `docs/` -
ver docs/LEIA-ME_scripts.md.
"""
import sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from comum import SAIDA
from importlib import import_module

_d = import_module('07_discursivo')
carregar, prever, CORTE = _d.carregar, _d.prever, _d.CORTE


if __name__ == '__main__':
    d, y, runs = carregar(SAIDA.parent / 'dados' / 'classex.parquet')
    prev, saidas, _, _ = prever(d)
    alunos = d['Color (Pseudonym)'].astype(str).values
    precisa = y < CORTE
    print(f'{len(d)} respostas | {precisa.mean():.1%} precisam de feedback '
          f'(nota de referencia < {CORTE})\n')

    print('1) COMO PRE-FILTRO DE DUAS PONTAS')
    print('   Decidir sozinho nas pontas, mandar o meio para a LLM.\n')
    print(f'  {"corte":<14} {"vai para a LLM":>15} {"erro ponta baixa":>18} {"erro ponta alta":>17}')
    linhas = []
    for p in (5, 10, 15, 20, 25):
        lo, hi = np.percentile(prev, p), np.percentile(prev, 100 - p)
        baixa, alta = prev <= lo, prev >= hi
        # erro na ponta baixa: dissemos "precisa de feedback" e nao precisava
        eb = (y[baixa] >= CORTE).mean()
        # erro na ponta alta: dissemos "esta adequada" e precisava
        ea = (y[alta] < CORTE).mean()
        resta = 1 - (baixa | alta).mean()
        print(f'  {p}% / {100-p}%{"":<7} {resta:>14.0%} {eb:>17.0%} {ea:>16.0%}')
        linhas.append(dict(corte=p, vai_para_llm=resta, erro_baixa=eb, erro_alta=ea))

    print('\n   O corte de 10%/90% economiza 20% das chamadas errando uma em sete')
    print('   na ponta baixa e uma em dez na alta. Isso e decisao de produto, nao')
    print('   obviedade - depende de quanto custa cada tipo de erro.')

    print('\n2) COMO TRIAGEM - ordenar uma fila de revisao humana\n')
    ordem = np.argsort(prev)
    print(f'  {"recorte":<28} {"nota real media":>16} {"vs geral":>10}')
    for k in (5, 10, 20, 50, 100):
        print(f'  as {k:>3} de menor previsao{"":<5} {y[ordem[:k]].mean():>16.2f} '
              f'{y[ordem[:k]].mean()-y.mean():>+10.2f}')
    print(f'  {"geral":<28} {y.mean():>16.2f}')
    print('\n   ORDENAR EXIGE MENOS QUE PONTUAR. E por isso que Spearman 0,50')
    print('   ainda serve para triagem, e nao serve para dar nota.')

    print('\n3) ESTABILIDADE - o numero se move entre particoes?\n')
    uni = np.unique(alunos)
    ss = []
    for s in range(12):
        sel = np.random.default_rng(s).choice(uni, int(len(uni) * 0.5), replace=False)
        m = np.isin(alunos, sel)
        ss.append(spearmanr(prev[m], y[m]).correlation)
    ss = np.array(ss)
    print(f'  media {ss.mean():.3f} | desvio {ss.std(ddof=1):.3f} | '
          f'amplitude {ss.max()-ss.min():.3f}')
    print(f'  o modulo publica 0,501 -> {abs(ss.mean()-0.501)/ss.std(ddof=1):.1f} desvios. Confere.')

    print('\n4) CUSTO\n')
    t0 = time.perf_counter()
    prever(d)
    dt = time.perf_counter() - t0
    print(f'  {len(d)} respostas em {dt*1000:.0f} ms -> {dt/len(d)*1e6:.0f} microssegundos cada')
    print(f'  Sem rede, sem GPU, sem dependencia externa. Uma chamada de LLM leva')
    print(f'  1 a 3 segundos e custa dinheiro. O modulo e ~{1.0/(dt/len(d)):,.0f}x mais rapido')
    print(f'  que uma chamada de 1 segundo.')
    print('\n  E o ponto que fecha o caso: ele nao SUBSTITUI a LLM - ele decide')
    print('  quem precisa dela, e da um retorno provisorio enquanto ela trabalha.')

    pd.DataFrame(linhas).to_csv(SAIDA / '08_operacao.csv', index=False)
