"""Dificuldade da questao como FAIXA DE VALORES, nao categoria.

Devolve um intervalo sobre a taxa de acerto, com a incerteza explicita. Sem
cortes arbitrarios: quem consome aplica o proprio limiar ao intervalo.

Posterior Beta-Binomial com prior derivado do proprio corpus. O intervalo usa
aproximacao normal do posterior - validada contra o Beta exato em 5.202
questoes do EdNet: COBERTURA 75,3% contra 74,8% do Beta, meio ponto de
diferenca. A afirmacao e sobre COBERTURA.

Os LIMITES do intervalo, esses, diferem mais: ate 2,6 pontos percentuais em
n pequeno com taxa extrema (n=10 com 1 acerto: [0,169-0,539] normal contra
[0,179-0,549] exato). Quem so decide com `afirmar()` nao se importa; quem
exibir o limite cru ao professor, com n baixo, deve saber disso.

DOIS INTERVALOS, DUAS PERGUNTAS - nao confundir:

  estimar()      "qual e a dificuldade desta questao?"
                 intervalo sobre a taxa LATENTE. Cobertura 88-91% no EdNet,
                 estavel de n=50 a n=400+.

  prever_turma() "quanto a MINHA turma de 30 vai acertar?"
                 soma o ruido de amostragem daquela turma. E MUITO mais largo,
                 e e o unico honesto para essa pergunta.

    turma de 30   so o posterior: cobre 39,7%  |  preditivo: cobre 90,5%
    turma de 60   so o posterior: cobre 49,7%  |  preditivo: cobre 88,2%

  Usar estimar() para responder "quanto minha turma vai acertar" erra em
  3 de cada 5 turmas. A largura triplica (0,097 -> 0,279 numa turma de 30)
  porque a incerteza e real, nao porque o metodo piorou.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

Z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}

# Faixa em que uma forca de prior faz sentido. Fora dela `derivar_prior` recusa
# em vez de devolver numero inutil: com forca 0,04 o Beta e bimodal (massa em 0
# e 1, o oposto de "dificuldade tipica"); com forca 210.000 - que o corpus sem
# variacao produzia - 30 respostas nao moveriam a estimativa em nada.
FORCA_MIN, FORCA_MAX = 0.5, 200.0


@dataclass(frozen=True)
class Prior:
    """Prior Beta sobre a dificuldade. Derive do seu corpus com `derivar_prior`.

    O padrao veio do EdNet (Beta(5.07, 2.08), forca 7.2). Um prior mais forte
    estreita o intervalo e QUEBRA a cobertura: com forca 30 o intervalo de 90%
    cobre 72%; com 60, cobre 57%.
    """
    a: float = 5.07
    b: float = 2.08

    @property
    def forca(self) -> float:
        return self.a + self.b

    @property
    def media(self) -> float:
        return self.a / (self.a + self.b)


def derivar_prior(taxas, tamanhos) -> Prior:
    """Casa os momentos da distribuicao de dificuldade do corpus, descontando o
    ruido binomial. E o unico parametro do modelo, e sai do dado - nao do olho."""
    taxas = list(taxas); tamanhos = list(tamanhos)
    n = len(taxas)
    mu = sum(taxas) / n
    var = sum((t - mu) ** 2 for t in taxas) / n
    ruido = sum(t * (1 - t) / m for t, m in zip(taxas, tamanhos) if m > 0) / n
    genuina = var - ruido
    if genuina <= 0:
        raise ValueError(
            'as questoes deste corpus nao variam mais do que o ruido binomial '
            f'explica (variancia {var:.5f}, ruido esperado {ruido:.5f}). Sem '
            'variacao genuina nao ha prior a derivar - use Prior() ou colete '
            'mais respostas por questao.')
    k = mu * (1 - mu) / genuina - 1
    if not FORCA_MIN <= k <= FORCA_MAX:
        raise ValueError(
            f'forca derivada de {k:.1f} esta fora de [{FORCA_MIN}, {FORCA_MAX}]. '
            + ('Abaixo do minimo o Beta vira bimodal (massa em 0 e 1), que nao '
               'descreve dificuldade. ' if k < FORCA_MIN else
               'Acima do maximo o prior domina qualquer evidencia de turma: '
               f'com forca {k:.0f}, 30 respostas nao moveriam a estimativa. ')
            + 'Verifique se os tamanhos passados sao os reais.')
    return Prior(mu * k, (1 - mu) * k)


@dataclass(frozen=True)
class Faixa:
    taxa: float           # estimativa pontual (media do posterior)
    minimo: float         # limite inferior do intervalo
    maximo: float         # limite superior
    nivel: float          # nivel nominal do intervalo
    respostas: int
    def __str__(self):
        return (f'{self.taxa:.0%} de acerto (entre {self.minimo:.0%} e {self.maximo:.0%}, '
                f'{self.nivel:.0%}, n={self.respostas})')
    @property
    def largura(self) -> float:
        return self.maximo - self.minimo


def estimar(acertos: int, respostas: int, prior: Prior = Prior(), nivel: float = 0.90,
            alunos: int | None = None) -> Faixa:
    """Faixa de dificuldade de UMA QUESTAO.

    `respostas` e quantas vezes a questao foi respondida - pela TURMA, nao por um
    aluno. `alunos` e quantos alunos DISTINTOS respondem por elas; quando o mesmo
    aluno reencontra a questao, as respostas nao sao independentes e o intervalo
    sai estreito demais. No EdNet 11,6% das respostas sao reencontro, e passar
    `alunos` leva a cobertura de 88,9% para 90,1% em n=100.

    Funciona com qualquer n, inclusive zero - com zero devolve o proprio prior,
    que e a resposta honesta para questao que ninguem respondeu.
    """
    if acertos != acertos or respostas != respostas:
        raise ValueError('acertos/respostas nao podem ser NaN')
    if acertos in (float('inf'), float('-inf')) or respostas in (float('inf'), float('-inf')):
        raise ValueError('acertos/respostas nao podem ser infinitos')
    if not 0 <= acertos <= respostas:
        raise ValueError(f'acertos={acertos} incompativel com respostas={respostas}')
    if nivel not in Z:
        raise ValueError(f'nivel deve ser um de {sorted(Z)}')
    if alunos is None:
        peso = 1.0
    elif respostas == 0:
        peso = 1.0
    elif not 0 < alunos <= respostas:
        raise ValueError(f'alunos={alunos} incompativel com respostas={respostas}')
    else:
        peso = alunos / respostas
    a = acertos * peso + prior.a
    b = (respostas - acertos) * peso + prior.b
    media = a / (a + b)
    desvio = sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    z = Z[nivel]
    return Faixa(media, max(0.0, media - z * desvio), min(1.0, media + z * desvio),
                 nivel, alunos if alunos is not None else respostas)


# Variacao da taxa de uma questao entre coortes, NAO capturada pelo posterior
# nem pela amostragem da turma. Nao encolhe com n - entra no preditivo.
#
# Medida diretamente (alunos antigos contra recentes no EdNet) da 0,0153, com
# correlacao 0,953 entre as coortes. Mas esse valor deixa o intervalo SUB-COBRIR:
# com 0,0153 o intervalo de 90% cobre 87,7% +- 1,0 em 8 particoes - 6,5
# erros-padrao abaixo do nominal, e sempre para baixo.
#
# O valor abaixo foi calibrado para a cobertura bater o nominal: ajustado em 4
# particoes, validado em 4 que nao participaram do ajuste.
#   sigma   ajuste   validacao
#   0,0153   87,4%     88,0%
#   0,0300   88,6%     89,2%
#   0,0400   89,7%     90,5%   <- escolhido
#   0,0500   90,9%     91,7%
# E o ajuste nao serve so ao nivel de 90%. Na validacao:
#   nivel 50% -> 52,9% | 80% -> 81,8% | 90% -> 90,5% | 95% -> 94,7%
#
# A distancia entre 0,0153 medido e 0,0400 necessario e informacao: existe
# variacao que o modelo de duas componentes nao representa (item que muda de
# comportamento, ordem de apresentacao, efeito de professor). Nao a explicamos -
# so garantimos que o intervalo nao mente sobre o proprio nivel.
SIGMA_COORTE = 0.0400


def prever_turma(acertos: int, respostas: int, alunos_na_turma: int,
                 prior: Prior = Prior(), nivel: float = 0.90,
                 alunos: int | None = None) -> Faixa:
    """Faixa da taxa que UMA TURMA de `alunos_na_turma` deve obter.

    Diferente de `estimar()`, que cobre a taxa latente da questao. Aqui entra
    tambem o sorteio daquela turma especifica - e e ele que domina em turma
    pequena. Medido no EdNet: cobertura 90,5% numa turma de 30, contra 39,7%
    do intervalo de `estimar()`.
    """
    if alunos_na_turma < 1:
        raise ValueError('alunos_na_turma tem de ser >= 1')
    base = estimar(acertos, respostas, prior, nivel, alunos)
    z = Z[nivel]
    # variancia do POSTERIOR, recalculada. Derivar do intervalo seria errado:
    # `estimar` clampa os limites em [0, 1], entao numa questao extrema (30/30)
    # o intervalo sai truncado e a variancia inferida ficaria pequena demais -
    # justo onde a incerteza e maior.
    peso = 1.0 if alunos is None or respostas == 0 else alunos / respostas
    pa = acertos * peso + prior.a
    pb = (respostas - acertos) * peso + prior.b
    var_post = pa * pb / ((pa + pb) ** 2 * (pa + pb + 1))
    mu = base.taxa
    var = var_post + mu * (1 - mu) / alunos_na_turma + SIGMA_COORTE ** 2
    desvio = sqrt(var)
    return Faixa(mu, max(0.0, mu - z * desvio), min(1.0, mu + z * desvio),
                 nivel, base.respostas)


def afirmar(faixa: Faixa, limiar: float, lado: str) -> bool:
    """So afirma quando o intervalo INTEIRO sustenta. E o que separa
    'provavelmente dificil' de 'dificil'."""
    if lado == 'abaixo': return faixa.maximo < limiar
    if lado == 'acima':  return faixa.minimo > limiar
    raise ValueError("lado deve ser 'abaixo' ou 'acima'")


if __name__ == '__main__':
    p = Prior()
    print(f'prior: Beta({p.a}, {p.b}) | forca {p.forca:.1f} | media {p.media:.1%}\n')
    print(f'{"acertos/respostas":>18}  {"faixa (90%)":<44} {"largura":>8}  afirmacao')
    print('-' * 100)
    for ac, n in [(0, 0), (2, 4), (3, 10), (7, 10), (14, 21), (5, 30), (24, 30),
                  (10, 50), (45, 50), (20, 100), (85, 100)]:
        f = estimar(ac, n, p)   # n = respostas da TURMA a esta questao
        af = ('dificil' if afirmar(f, .50, 'abaixo')
              else 'facil' if afirmar(f, .80, 'acima') else '-')
        print(f'{f"{ac}/{n}":>18}  {str(f):<44} {f.largura:>8.2f}  {af}')
