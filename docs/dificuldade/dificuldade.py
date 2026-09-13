"""Dificuldade da questao como FAIXA DE VALORES, nao categoria.

Devolve um intervalo sobre a taxa de acerto, com a incerteza explicita. Sem
cortes arbitrarios: quem consome aplica o proprio limiar ao intervalo.

Posterior Beta-Binomial com prior derivado do proprio corpus. O intervalo usa
aproximacao normal do posterior - validada contra o Beta exato em 11.338
questoes do EdNet, diferenca de cobertura menor que 1 ponto (ver
FAIXA_DIFICULDADE.md).

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

Z = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}


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
    genuina = max(var - ruido, 1e-6)
    k = mu * (1 - mu) / genuina - 1
    if k <= 0:
        raise ValueError('corpus sem variacao genuina de dificuldade')
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
