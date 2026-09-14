"""ETAPA 5 - por que acuracia nao serve, e o que usar no lugar.

No OULAD 92,9% dos alunos voltam. Um modelo que nunca alerta ninguem acerta
92,9% das vezes. Acuracia alta, utilidade zero. Esta etapa mostra isso com
numero, e mostra o que usamos no lugar: PONTO DE OPERACAO - alertar os X% de
maior risco, com o limiar saindo dos dados daquela coorte.

Tambem mostra por que o limiar NAO transfere entre bases, mesmo com a ordem
transferindo bem.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from comum import SAIDA
from importlib import import_module

z_por_coorte = import_module('03_vazamento').z_por_coorte
USADOS = ['recencia', 'frequencia']


if __name__ == '__main__':
    p = pd.read_parquet(SAIDA / 'painel.parquet')
    d = z_por_coorte(p, USADOS)
    m = LogisticRegression(max_iter=2000).fit(d[USADOS], d.voltou)
    d = d.assign(risco=1 - m.predict_proba(d[USADOS])[:, 1])

    print('1) ACURACIA COM LIMIAR FIXO DE 0,5\n')
    print(f'  {"coorte":<8} {"retencao":>9} {"alerta em":>10} {"acuracia":>9} '
          f'{"acuracia de nunca alertar":>27} {"balanceada":>11}')
    for c in sorted(p.coorte.unique()):
        s = d[d.coorte == c]
        alerta = s.risco >= 0.5
        acerto = ((~alerta) == (s.voltou == 1)).mean()
        nunca = s.voltou.mean()
        bal = balanced_accuracy_score(s.voltou, (~alerta).astype(int))
        print(f'  {c:<8} {s.voltou.mean():>9.1%} {alerta.mean():>10.1%} '
              f'{acerto:>9.1%} {nunca:>27.1%} {bal:>11.1%}')

    print('\n2) PONTO DE OPERACAO - alertar os 10% de maior risco\n')
    print(f'  {"coorte":<8} {"limiar":>8} {"precisao":>9} {"na base":>8} {"lift":>6}')
    for c in sorted(p.coorte.unique()):
        s = d[d.coorte == c]
        lim = np.quantile(s.risco, 0.90)
        sel = s.risco >= lim
        prec = (s.voltou[sel] == 0).mean()
        b = (s.voltou == 0).mean()
        print(f'  {c:<8} {lim:>8.3f} {prec:>9.1%} {b:>8.1%} {prec/b:>5.1f}x')

    print('\n  Os limiares sao diferentes porque as bases sao diferentes. Exportar')
    print('  o limiar de uma para a outra troca completamente quem e alertado -')
    print('  e por isso que calibrar por coorte nao e detalhe, e requisito.')

    print('\n3) A ORDEM TRANSFERE, O NIVEL NAO\n')
    for c in sorted(p.coorte.unique()):
        outra = [x for x in p.coorte.unique() if x != c][0]
        tr, te = d[d.coorte == outra], d[d.coorte == c]
        mm = LogisticRegression(max_iter=2000).fit(tr[USADOS], tr.voltou)
        pr = mm.predict_proba(te[USADOS])[:, 1]
        ece = abs(pr.mean() - te.voltou.mean())
        print(f'  treinado em {outra:<7} -> {c:<7} AUC {roc_auc_score(te.voltou, pr):.3f} '
              f'(ordem, boa) | erro da media prevista {ece:.3f} (nivel, ruim)')
