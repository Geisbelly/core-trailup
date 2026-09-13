"""ETAPA 6 - exporta o modelo para o prototipo web.

O modelo e uma regressao logistica de duas features. Cabe num JSON e roda em
JavaScript sem biblioteca nenhuma - por isso o prototipo e uma pagina estatica
no GitHub Pages, sem servidor.

Exporta tambem a tabela de referencia (retencao observada por dias recentes em
cada base), que e o que a pagina usa para mostrar o aluno contra a coorte.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from comum import SAIDA
from importlib import import_module

z_por_coorte = import_module('03_vazamento').z_por_coorte
USADOS = ['recencia', 'frequencia']

if __name__ == '__main__':
    p = pd.read_parquet(SAIDA / 'painel.parquet')
    medido = json.load(open(SAIDA / '04_modelo.json'))
    d = z_por_coorte(p, USADOS)
    m = LogisticRegression(max_iter=2000).fit(d[USADOS], d.voltou)

    # RECALIBRACAO POR COORTE, E ELA NAO E OPCIONAL.
    # O modelo conjunto foi treinado com as duas bases padronizadas, entao a
    # probabilidade que ele devolve tem o nivel da MEDIA das duas - o que nao
    # corresponde a nenhuma das duas. No EdNet 86,2% nao voltam e no OULAD 8,5%;
    # um numero so nao pode estar certo nos dois lugares.
    #
    # A correcao e uma logistica de UMA variavel (o logito do modelo) ajustada
    # dentro de cada coorte: mantem a ORDEM identica - AUC nao muda uma casa -
    # e conserta o NIVEL. E a traducao em codigo da frase que repetimos no
    # relatorio: a ordem transfere, o nivel nao.
    coortes = {}
    for c in sorted(p.coorte.unique()):
        s = p[p.coorte == c]
        z = d[d.coorte == c]
        pr = m.predict_proba(z[USADOS])[:, 1]
        logito = np.log(np.clip(pr, 1e-6, 1 - 1e-6) / (1 - np.clip(pr, 1e-6, 1 - 1e-6)))
        rec = LogisticRegression(max_iter=2000).fit(logito.reshape(-1, 1), s.voltou)
        a_rec, b_rec = float(rec.intercept_[0]), float(rec.coef_[0][0])
        prc = rec.predict_proba(logito.reshape(-1, 1))[:, 1]
        antes = abs((1 - pr).mean() - (1 - s.voltou).mean())
        depois = abs((1 - prc).mean() - (1 - s.voltou).mean())
        print(f'  {c}: erro da media prevista {antes:.3f} -> {depois:.3f} '
              f'| AUC {roc_auc_score(s.voltou, pr):.4f} -> {roc_auc_score(s.voltou, prc):.4f}')
        risco = 1 - prc
        ref = (s.groupby(s.dias_recentes.clip(0, 10)).voltou
               .agg(['mean', 'size']).reindex(range(11)))
        coortes[c] = {
            'n': int(len(s)),
            'retencao': float(s.voltou.mean()),
            'media': {k: float(s[k].mean()) for k in USADOS},
            'desvio': {k: float(s[k].std()) for k in USADOS},
            'recal_a': a_rec,
            'recal_b': b_rec,
            'limiar10': float(np.quantile(risco, 0.90)),
            'limiar25': float(np.quantile(risco, 0.75)),
            'referencia': [None if np.isnan(v) else round(float(v), 4)
                           for v in ref['mean']],
            'referencia_n': [0 if np.isnan(v) else int(v) for v in ref['size']],
        }

    saida = {
        'features': USADOS,
        'pesos': [float(x) for x in m.coef_[0]],
        'intercepto': float(m.intercept_[0]),
        'coortes': coortes,
        'auc': medido['auc'],
        'linha_de_base': medido['linha_de_base'],
        'janela_dias': 30,
        'janela_recente': 10,
    }
    caminho = SAIDA.parent / 'docs' / 'modelo.json'
    json.dump(saida, open(caminho, 'w'), indent=2, ensure_ascii=False)
    print(f'-> {caminho}')
    print(json.dumps({k: v for k, v in saida.items() if k != 'coortes'}, indent=2)[:600])
