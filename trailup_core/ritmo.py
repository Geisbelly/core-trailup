"""Ritmo da questao: rapida ou lenta.

A forma dos grupos NAO foi escolhida. Tres metodos que descobrem a quantidade de
grupos rodaram sobre 10.526 questoes do EdNet; o unico estavel (HDBSCAN, ARI
1,000 variando seu parametro de 50 a 400) encontrou DOIS grupos, separados
quase inteiramente pelo TEMPO (d de Cohen 3,30 na latencia, 0,50 no acerto).

Consequencia de desenho: o acerto NAO tem estrutura de grupo - e continuo, e por
isso vai como intervalo (ver dificuldade.py). So o ritmo vira categoria.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass

# Fronteira entre os dois grupos descobertos: mediana de 40 s.
# O grupo rapido vai ate 40 s (p95 = 27 s); o lento comeca em 36 s (p5 = 52 s).
FRONTEIRA_SEG = 40.0

# acerto da classificacao de ritmo por n de respostas, medido no EdNet
# Acerto MEDIDO da classificacao com n respostas, contra a classificacao feita
# sobre 200+ respostas. 8.673 questoes, tres reamostragens independentes.
# A versao anterior desta tabela dizia (5, .98), (10, .99), (21, 1.0) - vinha
# de uma medida que a auditoria de 2026-09-13 refutou, e afirmava CERTEZA em
# n=21, que nunca e verdade.
#
#   n=3  0,953 [0,950-0,956]     n=15  0,993 [0,991-0,994]
#   n=5  0,973 [0,971-0,975]     n=21  0,994 [0,993-0,995]
#   n=8  0,986 [0,985-0,988]     n=30  0,996 [0,995-0,997]
#   n=10 0,989 [0,987-0,990]     n=50  0,997 [0,996-0,997]
_CONF = [(5, .973), (10, .989), (21, .994), (50, .997)]

RITMOS = {
    'rapida': 'respondida em menos de ~40 s: reconhecimento, aplicacao direta',
    'lenta':  'respondida em mais de ~40 s: exige elaboracao ou releitura',
}


@dataclass(frozen=True)
class Ritmo:
    valor: str            # 'rapida' | 'lenta'
    latencia: float       # mediana observada, em segundos
    confianca: float
    respostas: int
    def __str__(self):
        # uma casa decimal: com .0% o teto medido (0,997) aparece como 100%,
        # que e a falsa certeza que esta tabela existe para nao afirmar
        return f'{self.valor} ({self.latencia:.0f}s mediana, confianca {self.confianca:.1%})'


def ritmo(latencia_mediana: float, respostas: int) -> Ritmo:
    """`respostas` = quantos ALUNOS responderam a questao.

    Precisa de muito pouco: 5 respostas classificam o ritmo com 97,3% de
    acerto contra a classificacao feita sobre 200+ respostas. E o eixo barato -
    o caro e o acerto.

    A confianca devolvida NUNCA chega a 1,0, nem com 50 respostas: o teto
    medido e 0,997. A tabela anterior afirmava 1,00 em n=21.
    """
    if respostas < 5:
        raise ValueError(f'{respostas} respostas e pouco: o ritmo precisa de >=5')
    if latencia_mediana <= 0:
        raise ValueError('latencia mediana deve ser positiva')
    c = _CONF[-1][1]
    for n, v in _CONF:
        if respostas <= n:
            c = v
            break
    return Ritmo('lenta' if latencia_mediana > FRONTEIRA_SEG else 'rapida',
                 latencia_mediana, c, respostas)


def descrever(faixa, r: Ritmo) -> str:
    """Junta os dois eixos: categoria no que e discreto, intervalo no que e continuo.

    `faixa` vem de dificuldade.estimar().
    """
    return (f'{r.valor}, acerto entre {faixa.minimo:.0%} e {faixa.maximo:.0%} '
            f'({faixa.nivel:.0%}, n={faixa.respostas})')


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dificuldade import estimar, Prior, afirmar
    p = Prior()
    casos = [(24, 30, 15.0), (14, 30, 18.0), (25, 30, 85.0), (13, 30, 90.0),
             (60, 100, 22.0), (20, 100, 75.0), (4, 6, 12.0)]
    print(f'{"acertos/n":>10} {"lat":>6}  {"perfil":<52} afirmacao')
    print('-' * 92)
    for ac, n, lat in casos:
        f = estimar(ac, n, p)
        r = ritmo(lat, n)
        af = ('dificil' if afirmar(f, .50, 'abaixo')
              else 'facil' if afirmar(f, .80, 'acima') else '-')
        print(f'{f"{ac}/{n}":>10} {lat:>5.0f}s  {descrever(f, r):<52} {af}')
    print()
    for k, v in RITMOS.items():
        print(f'  {k:<8} {v}')
