"""Questao defeituosa: discriminacao (correlacao item-total).

Se os alunos que MAIS acertam no geral erram esta questao, algo esta errado -
gabarito trocado, enunciado ambiguo, distrator melhor que a resposta.

Medido no EdNet (10.636 questoes, duas metades independentes de alunos):
discriminacao media 0,208.

    negativa numa metade dos alunos      2,96%  (315 questoes)
    negativa NAS DUAS metades            0,53%  ( 56 questoes)

A queda de 2,96% para 0,53% e o resultado: a maior parte do que aparece como
negativo numa metade nao aparece na outra - e ruido amostral, nao defeito.
E banco comercial curado; item quebrado nao sobrevive ali.

Replicacao entre metades: +0,396 (IC95 +0,379 a +0,413). Marcando as piores,
so 18-27% confirmam (lift 4-6x sobre a base, precisao baixa demais para exibir
sem ressalva).

CORRECAO 2026-09-13: este cabecalho afirmava "zero questoes com valor negativo".
Era bug do script de medicao, que comparava contra -1 em vez de 0.

Por que fica aqui mesmo assim: no TrailUp as questoes sao escritas por professor
e GERADAS POR IA. Item com gabarito errado ou distrator ambiguo e esperado, nao
excepcional - e este e um dos poucos metodos que deve valer MAIS no produto do
que valeu no corpus de referencia.

COMO MARCAR, MEDIDO (2026-09-13, 9.877 questoes, TRES particoes de alunos:
duas para marcar e uma, nunca vista, para conferir). Alvo: ser negativa na
terceira particao (prevalencia 3,53%).

    regra (usa duas particoes)      marca   precisao          lift
    negativa numa particao            239     13,8%           3,9x
    negativa em p0 OU p1              549     12,2%           3,5x
    media das duas < 0                132     25,0%           7,1x   <- use esta
    negativa nas DUAS                  39     28,2%           8,0x
    media das duas < -0,05             27     29,6%           8,4x

    Os lifts acima sao sobre uma base de 3,53%. NAO CITE O LIFT SOZINHO: ele e
    a precisao dividida por essa base, e a base depende do arranjo de medicao,
    nao da regra. Exigindo um minimo diferente de respostas por questao, com a
    MESMA regra (8 sementes cada):

        minimo de respostas    base   precisao    lift
             20               10,42%    20,6%     2,0x
             30                7,70%    22,5%     2,9x
             60                2,88%    20,0%     6,9x

    A precisao mal se move; o lift triplica. Exigir mais respostas limpa o
    ruido da particao de conferencia, derruba a base e infla o lift sem que
    nada tenha melhorado. O que compara regras e a PRECISAO, medida no mesmo
    arranjo: 22,5% desta contra 13,8% de marcar por uma medida so.

    ESTABILIDADE (8 particoes): precisao media 22,5% com desvio 4,0% - o 25,0%
    publicado fica a 0,6 desvios. Confere.

Dividir os alunos em duas metades e exigir que a MEDIA seja negativa quase
DOBRA a precisao (13,8% -> 25,0%) marcando metade das questoes. E o que
`confirmar()` faz.

A correlacao da discriminacao entre particoes independentes e de apenas 0,33 -
por isso uma medida so erra tanto, e por isso a confirmacao ajuda tanto.

Sem dependencia externa.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

MIN_RESPOSTAS = 40


@dataclass(frozen=True)
class Discriminacao:
    valor: float           # -1 a 1
    respostas: int
    veredito: str          # 'suspeita' | 'fraca' | 'ok' | 'indeterminado'
    def __str__(self):
        return f'discriminacao {self.valor:+.2f} ({self.veredito}, n={self.respostas})'


def discriminacao(respostas: list[tuple[bool, float]],
                  min_respostas: int = MIN_RESPOSTAS) -> Discriminacao:
    """`respostas` = lista de (acertou_esta_questao, habilidade_geral_do_aluno).

    A habilidade DEVE excluir esta questao, senao a propria questao infla a
    correlacao. Use (acertos_totais - acertou_aqui) / (respostas_totais - 1).
    """
    n = len(respostas)
    if n < min_respostas:
        return Discriminacao(float('nan'), n, 'indeterminado')
    x = [1.0 if a else 0.0 for a, _ in respostas]
    y = [h for _, h in respostas]
    mx, my = sum(x) / n, sum(y) / n
    sx = sqrt(sum((v - mx) ** 2 for v in x) / n)
    sy = sqrt(sum((v - my) ** 2 for v in y) / n)
    if sx == 0 or sy == 0:
        return Discriminacao(float('nan'), n, 'indeterminado')
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n * sx * sy)
    if r < 0.0:    v = 'suspeita'    # aluno bom erra: revisar gabarito
    elif r < 0.05: v = 'fraca'       # nao separa ninguem
    else:          v = 'ok'
    return Discriminacao(round(r, 3), n, v)


def confirmar(disc_metade_a: float, disc_metade_b: float,
              limiar: float = 0.0) -> bool:
    """Marca a questao como suspeita so quando as duas metades concordam.

    Divida os alunos que responderam a questao em duas metades aleatorias,
    calcule a discriminacao em cada uma, e passe as duas aqui.

    Medido no EdNet contra uma terceira particao nunca vista: precisao 25,0%
    contra 13,8% de marcar por uma medida so - e a precisao confere entre
    particoes (22,5% +- 4,0% em 8 sementes). Com limiar -0,05 vai a 29,6%,
    marcando um quinto das questoes.

    O lift de 7,1x que o cabecalho cita vale PARA AQUELA BASE (3,53%). Ele nao
    e propriedade da regra - ver a tabela de sensibilidade no cabecalho antes
    de repetir esse numero em qualquer lugar.

    Precisao de 25% quer dizer que TRES EM CADA QUATRO marcacoes sao falso
    positivo. Serve para ordenar fila de revisao humana, nao para despublicar
    questao automaticamente.

    A confirmacao importa porque a discriminacao de uma questao correlaciona
    apenas 0,33 entre particoes independentes de alunos - uma medida isolada
    e, em boa parte, ruido amostral.
    """
    a, b = float(disc_metade_a), float(disc_metade_b)
    # `discriminacao` devolve NaN quando nao da para medir (n baixo, ou todo
    # mundo acertou / errou). Sem esta guarda, NaN < limiar e False e a questao
    # nao-mensuravel seria silenciosamente tratada como "nao suspeita".
    if a != a or b != b:
        raise ValueError('discriminacao indeterminada (NaN) numa das metades: '
                         'nao da para confirmar. Trate como "sem evidencia", '
                         'nao como "questao ok".')
    return (a + b) / 2.0 < limiar
