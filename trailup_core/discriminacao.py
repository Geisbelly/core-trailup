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

Por isso o limiar de exibicao e conservador: com precisao de 18-27% medida num
corpus de prevalencia baixa, so vale sinalizar o que for extremo. A precisao
sobe com a prevalencia - quanto, e questao empirica a medir no TrailUp.

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
