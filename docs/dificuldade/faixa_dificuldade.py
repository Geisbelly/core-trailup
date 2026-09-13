"""Classificador de faixa de dificuldade da questao.

Sem dependencia: aritmetica e um limiar. Todo parametro e explicito.
Validado em 6,5 M respostas do EdNet - ver FAIXA_DIFICULDADE.md.
"""
from dataclasses import dataclass
from math import sqrt

@dataclass(frozen=True)
class Config:
    corte_dificil: float = 0.50   # acerto abaixo disto = dificil
    corte_facil: float = 0.80     # acerto acima disto  = facil
    prior: int = 15               # forca do encolhimento para a media global
    min_respostas: int = 10       # abaixo disto nunca classifica
    margem_erro: float = 1.0      # em erros-padrao; 0 desliga a abstencao

FAIXAS = ('dificil', 'media', 'facil', 'indeterminado')

def classificar(acertos: int, respostas: int, media_global: float, cfg: Config = Config()):
    """Devolve (faixa, taxa_estimada, motivo).

    A taxa vem encolhida para a media global: com poucas respostas a estimativa
    crua e pior que o global (medido: abaixo de 10 alunos ela erra mais).
    """
    if respostas < cfg.min_respostas:
        return 'indeterminado', None, f'so {respostas} respostas (minimo {cfg.min_respostas})'
    taxa = (acertos + media_global * cfg.prior) / (respostas + cfg.prior)
    if cfg.margem_erro > 0:
        se = sqrt(max(taxa * (1 - taxa), 1e-9) / respostas) * cfg.margem_erro
        for corte in (cfg.corte_dificil, cfg.corte_facil):
            if taxa - se < corte < taxa + se:
                return 'indeterminado', taxa, f'estimativa {taxa:.0%} +-{se:.0%} cruza o corte de {corte:.0%}'
    faixa = 'dificil' if taxa < cfg.corte_dificil else ('media' if taxa < cfg.corte_facil else 'facil')
    return faixa, taxa, f'{respostas} respostas'

def confianca(respostas: int) -> float:
    """Acerto esperado da faixa, medido no EdNet. Interpolado entre os pontos
    observados; nao extrapola para cima de 100 respostas."""
    pontos = [(5, .55), (10, .65), (15, .69), (21, .72), (30, .76), (50, .80), (100, .84)]
    if respostas <= pontos[0][0]: return pontos[0][1]
    if respostas >= pontos[-1][0]: return pontos[-1][1]
    for (a, va), (b, vb) in zip(pontos, pontos[1:]):
        if a <= respostas <= b:
            return va + (vb - va) * (respostas - a) / (b - a)
    return pontos[-1][1]

if __name__ == '__main__':
    G = 0.668
    print(f'{"acertos/respostas":>18} {"faixa":>14} {"taxa":>7} {"conf":>6}  motivo')
    print('-' * 78)
    for ac, n in [(2, 4), (3, 10), (2, 12), (7, 10), (9, 10), (14, 21), (18, 21),
                  (15, 30), (24, 30), (60, 100), (20, 100)]:
        f, t, m = classificar(ac, n, G)
        print(f'{f"{ac}/{n}":>18} {f:>14} {f"{t:.0%}" if t else "-":>7} '
              f'{confianca(n):>5.0%}  {m}')
