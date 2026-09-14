"""Pre-avaliacao rapida de resposta discursiva.

NAO substitui a LLM. Da uma estimativa instantanea enquanto ela trabalha, e
ordena respostas para o professor revisar.

Medido no classEx (1.167 respostas, validacao agrupada por aluno):

    versao anterior (8 features)              Spearman 0,451 | QWK 0,318
    esta versao, features do pipeline sklearn Spearman 0,463 | QWK 0,350
    ESTA VERSAO PONTA A PONTA (df 0,50)       Spearman 0,483
    ESTA VERSAO PONTA A PONTA (df 0,40)       Spearman 0,501   <- atual
    boosting sobre as mesmas features         Spearman 0,429 | QWK 0,398

A auditoria de 2026-09-13 chamou preparar()/avaliar() sobre os textos crus e
mediu 0,483 - MELHOR que os 0,463 do pipeline sklearn que gerou os pesos. As
duas implementacoes do grafo diferem (o pipeline usa as 60 palavras mais
frequentes como stopword; este modulo deriva por frequencia de documento, o
que no classEx da apenas 5). O numero que descreve o que ESTE codigo faz e
0,483.

LIMITE DA ESCALA: as previsoes ficam entre 2,31 e 5,00. O modulo nao consegue
dizer "muito ruim" - o piso efetivo e 2,3 numa escala de 1 a 5. Para triagem
isso nao atrapalha (as 10 piores tem nota real 2,20 contra 3,50 do geral),
mas nao use o valor absoluto como nota.

OS TRES CAMPOS AUXILIARES, VALIDADOS SEPARADAMENTE (Spearman com a nota humana):

    cobertura            +0,462   forte - quase o da nota inteira (+0,483)
    conceitos_faltando   -0,410   forte
    divagacao            -0,114   FRACO, e a escala engana

DIVAGACAO NAO E O QUE O NOME SUGERE. E a fracao dos tokens da resposta que nao
aparecem no gabarito nem no enunciado - e em prosa a maioria das palavras nao
aparece mesmo (conectivos, exemplos, reformulacao). A distribuicao real:

    p10 0,57 | mediana 0,72 | p90 0,83

Ou seja: divagar 72% E O NORMAL. Exibir "divaga 72%" para um professor sobre
uma resposta de nota 4 diz o oposto do que ele vai entender. Use apenas
COMPARADO a mediana da propria turma, nunca em absoluto - e prefira
`cobertura` e `conceitos_faltando`, que sao fortes e diretos.
    + encoder multilingue (torch, ~900 MB)    Spearman 0,532 | QWK 0,433
    TETO: execucoes INDEPENDENTES do LLM      Spearman 0,871-0,897

Fica claro pelo teto: isto nao avalia resposta. Ordena e tria.

CORRECAO 2026-09-13: o teto citado antes era 0,899, que vinha de comparar UMA
execucao do LLM com a MEDIA das tres - media que contem aquela execucao. Entre
execucoes independentes o teto e ~0,88.

O rotulo tem pouca variacao: 87% das notas sao 3 ou 4 (media 3,50, desvio 0,70).
Isso comprime qualquer correlacao medida contra ele.

COMO TRIAGEM, QUE E O USO: as 10 respostas de menor previsao tem nota real
media 1,87 contra 3,50 do geral - mais de dois desvios abaixo.
Como pre-filtro de duas pontas, o erro nas pontas e 12-14%, nao 5-8%.

O que ficou provado que NAO adianta, para nao ser tentado de novo:
  - enriquecer a referencia com conteudo do dominio: PIORA (0,486 -> 0,396)
  - mais dados de treino: curva plana (dobrar rendeu +0,030)
  - encoder pre-treinado: +0,057, e nao dispensa estas features
  - CAMADA SEMANTICA (LSA sobre o corpus): nao acrescenta NADA. As 13 features
    sem ela dao 0,459; as 15 com ela, 0,458. Todo o ganho do conjunto ampliado
    vem de n_nos e n_arestas, que sao contagem pura (+0,014 e +0,015)

ATENCAO AOS COEFICIENTES: foram ajustados em texto ALEMAO, de uma disciplina
(macroeconomia). As FEATURES sao agnosticas de idioma - sao operacoes de
conjunto sobre tokens. Os PESOS nao. Reajustar com respostas em portugues
assim que houver ~100 com nota humana; ate la, usar so para ORDENAR
(a ordenacao e bem mais robusta a peso errado que a nota absoluta).

Sem dependencia externa.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_TOKEN = re.compile(r'\w{3,}', re.UNICODE)
# JANELA e DF_STOPWORD foram escolhidos sem medida. Varridos em 2026-09-13,
# com IC95 por bootstrap agrupado por aluno (300 reamostras):
#
#   JANELA=3, df=0,5   Spearman 0,472  [0,426 - 0,516]
#   JANELA=4, df=0,5   Spearman 0,483  [0,440 - 0,526]   <- padrao
#   JANELA=4, df=0,4   Spearman 0,501  [0,456 - 0,542]
#   JANELA=5, df=0,5   Spearman 0,477  [0,435 - 0,517]
#   JANELA=8, df=0,5   Spearman 0,388  [0,343 - 0,438]   <- unico pior de fato
#
# CORRECAO DE METODO (2026-09-13). Eu tinha concluido que 0,4 e 0,5 eram
# indistinguiveis porque os IC se sobrepoem. COMPARAR IC SOBREPOSTOS NAO E UM
# TESTE - duas medidas correlacionadas podem ter IC largos e uma diferenca
# consistente. O teste certo e a DIFERENCA PAREADA, nas mesmas reamostras:
#
#   df 0,4 menos df 0,5:  media +0,0187 | desvio 0,0048
#   positiva em 100% das 40 reamostras | IC95 [+0,0117, +0,0290]
#
# O ganho sobrevive, e eu tinha rejeitado errado. DF_STOPWORD passa a 0,40.
#
# Pelo mesmo teste pareado, o conjunto de 10 features tambem se confirma:
#   10 menos 8 features: +0,0153 [+0,0121, +0,0176], positivo em 100%
#
# A JANELA continua em 4: de 3 a 5 a diferenca pareada nao exclui zero, e a
# fronteira real e em 8, onde a qualidade cai fora do intervalo.
JANELA = 4            # coocorrencia: palavras a ate 4 posicoes formam aresta
MIN_DOCS = 10         # abaixo disso nao da para estimar stopword por frequencia
DF_STOPWORD = 0.40    # medido: 0,40 bate 0,50 em teste pareado (+0,019)

# Ajustados por regressao ridge no classEx. Ver aviso no cabecalho.
# Ridge sobre 10 features, validacao cruzada agrupada POR ALUNO (classEx).
# Spearman 0,463 e QWK 0,350, contra 0,451 / 0,318 do conjunto de 8.
#
# NAO INTERPRETAR OS SINAIS ISOLADAMENTE. As features sao colineares e o ridge
# distribui peso entre elas: 'divagacao' aparece positiva aqui e negativa na
# versao de 8, sem que o construto tenha mudado. So a soma tem sentido.
_PESOS = {
    'no_peso':                +1.986397,   # cobertura ponderada dos conceitos do gabarito
    'no_cobertura':           +0.368581,
    'cob_conceitos_centrais': +0.882276,   # cobre os conceitos mais conectados?
    'divagacao':              +0.132323,
    'razao_tam':              +0.250126,
    'aresta_cobertura':       +0.658591,
    'aresta_peso':            -2.120108,
    'log_len':                +0.812077,
    'n_nos':                  -0.013081,   # tamanho do grafo da resposta
    'n_arestas':              -0.002815,
}
_INTERCEPTO = -1.364109


def _tokens(texto: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(str(texto))]


def stopwords(textos: list[str], df_max: float = DF_STOPWORD) -> set[str]:
    """Deriva stopwords por FREQUENCIA DE DOCUMENTO - agnostico de idioma.

    Palavra que aparece em mais de `df_max` dos textos nao distingue nada.

    Precisa de um CORPUS: passe enunciados, gabaritos e material do topico
    inteiro, nao so a questao em avaliacao. Com menos de MIN_DOCS textos
    devolve conjunto vazio - melhor nenhuma stopword do que transformar todo o
    vocabulario em stopword, que era o que a versao anterior fazia com dois
    textos curtos: o grafo de referencia saia VAZIO e tudo pontuava igual.
    """
    if len(textos) < MIN_DOCS:
        return set()
    df: Counter = Counter()
    for t in textos:
        df.update(set(_tokens(t)))
    limite = df_max * len(textos)
    return {w for w, c in df.items() if c > limite}


def _grafo(texto: str, stop: set[str]) -> tuple[Counter, Counter]:
    ws = [w for w in _tokens(texto) if w not in stop]
    nos = Counter(ws)
    ar: Counter = Counter()
    for i, a in enumerate(ws):
        for b in ws[i + 1:i + JANELA]:
            if a != b:
                ar[(min(a, b), max(a, b))] += 1
    return nos, ar


@dataclass(frozen=True)
class Referencia:
    """Grafo do gabarito, montado uma vez por questao e reaproveitado."""
    nos: Counter
    arestas: Counter
    centrais: list[str]
    tokens_enunciado: set[str]
    stop: set[str] = field(default_factory=set)


def preparar(gabarito: str, enunciado: str = '', stop: set[str] | None = None,
             n_centrais: int = 10) -> Referencia:
    """Pre-computa o gabarito. Faca uma vez por questao, nao por resposta.

    `stop`: passe `stopwords(materiais_do_topico)`. Sem ele o grafo inclui
    palavras funcionais, o que dilui um pouco a cobertura mas nao quebra nada -
    ao contrario de usar stopwords mal estimadas.
    """
    # sem corpus nao ha como estimar stopword; vazio e o padrao seguro
    st = stop if stop is not None else set()
    nos, ar = _grafo(gabarito, st)
    grau: dict[str, int] = defaultdict(int)
    for (a, b), w in ar.items():
        grau[a] += w
        grau[b] += w
    centrais = [w for w, _ in sorted(grau.items(), key=lambda x: -x[1])[:n_centrais]]
    nq, _ = _grafo(enunciado, st) if enunciado else (Counter(), Counter())
    return Referencia(nos, ar, centrais, set(nq), st)


@dataclass(frozen=True)
class PreAvaliacao:
    nota: float                 # 1 a 5
    cobertura: float            # 0 a 1 - quanto do gabarito a resposta toca
    divagacao: float            # 0 a 1; mediana ~0,72 - NAO leia como % de divagacao
    conceitos_faltando: list[str]

    @property
    def percentual_estimado(self) -> float:
        """A escala 1--5 reescrita em 0--100. NAO e porcentagem de acerto.

        MEDIDO NO CORPUS classEx (1.167 respostas), contra a nota de referencia
        na mesma escala:
            erro absoluto medio   13,0 pontos percentuais
            vies                  +5,9 pontos (exibe alto)
            Spearman              +0,422

        Treze pontos de erro medio em cima de um numero de dois digitos: quem le
        ``72%`` entende uma nota, e a nota real daquela resposta esta tipicamente
        entre 59 e 85. Por isso ``__str__`` mostra a FAIXA, nao este numero.

        Calibrar o vies foi testado (regressao linear com holdout por tarefa):
        remove o vies (+5,9 -> 0,0) mas NAO reduz o erro (ganho -0,48 pontos,
        IC95 [-1,34, +0,19], melhora em 3/5 dobras). Nao foi aplicado - trocar
        um numero errado por outro numero errado sem ganho medido e ruido.
        """
        return round(max(0.0, min(100.0, (self.nota - 1.0) / 4.0 * 100.0)), 1)

    @property
    def faixa_percentual(self) -> str:
        """Faixa ordinal para triagem. E o que esta validado para exibicao.

        Cortes em 50 e 75 na escala, ou seja notas 3,0 e 4,0. No corpus classEx
        as faixas separam de verdade, em ordem e sem cruzar:
            baixo    4,5% das respostas   nota real media  32,2%
            medio   70,8%                                  61,2%
            alto    24,8%                                  70,5%

        Compara a nota CRUA, nao o ``percentual_estimado`` arredondado - senao
        uma nota de 3,9999 vira 75,0 no arredondamento e sobe de faixa sozinha.
        """
        pct = (self.nota - 1.0) / 4.0 * 100.0
        if pct < 50.0:
            return 'baixo'
        if pct < 75.0:
            return 'medio'
        return 'alto'

    def __str__(self):
        falta = (', '.join(self.conceitos_faltando[:4]) or 'nenhum dos centrais')
        return (f'{self.faixa_percentual} (~{self.nota:.1f}/5, +-0,5) | cobre {self.cobertura:.0%} do gabarito, '
                f'fora do gabarito {self.divagacao:.0%} (tipico ~72%) | falta: {falta}')


def avaliar(resposta: str, ref: Referencia) -> PreAvaliacao:
    """Estimativa instantanea. Use para ORDENAR e para feedback provisorio."""
    import math
    nr, ar = _grafo(resposta, ref.stop)
    sr, sg = set(nr), set(ref.nos)
    er, eg = set(ar), set(ref.arestas)
    ni, ei = sr & sg, er & eg
    tot_n = max(sum(ref.nos.values()), 1)
    tot_a = max(sum(ref.arestas.values()), 1)
    f = {
        'no_peso': sum(ref.nos[w] for w in ni) / tot_n,
        'no_cobertura': len(ni) / max(len(sg), 1),
        'cob_conceitos_centrais': (sum(w in sr for w in ref.centrais) / len(ref.centrais)
                                   if ref.centrais else 0.0),
            # fracao dos tokens fora do gabarito e do enunciado. Mediana ~0,72:
        # NAO e "quanto o aluno divagou" - ver o cabecalho.
        'divagacao': len(sr - sg - ref.tokens_enunciado) / max(len(sr), 1),
        'razao_tam': len(sr) / max(len(sg), 1),
        'aresta_cobertura': len(ei) / max(len(eg), 1),
        'aresta_peso': sum(min(ar[e], ref.arestas[e]) for e in ei) / tot_a,
        'log_len': math.log1p(len(str(resposta))),
        'n_nos': float(len(sr)),
        'n_arestas': float(len(er)),
    }
    nota = _INTERCEPTO + sum(_PESOS[k] * v for k, v in f.items())
    return PreAvaliacao(max(1.0, min(5.0, nota)), f['no_cobertura'], f['divagacao'],
                        [w for w in ref.centrais if w not in sr])


def triar(respostas: list[str], ref: Referencia) -> list[tuple[int, PreAvaliacao]]:
    """Ordena da mais fraca para a mais forte - o uso mais defensavel.

    Medido (validacao agrupada por aluno, 1.167 respostas):

        quantas piores   8 features   10 features   geral
              5             2,20          1,80       3,50
             10             2,10          2,03       3,50
             20             2,37          2,03       3,50

    Ordenar exige bem menos precisao que pontuar. A cauda ALTA fica um pouco
    pior com 10 features (as 10 melhores: 4,37 -> 4,13) - o conjunto novo foi
    escolhido pela cauda baixa, que e o uso declarado.
    """
    av = [(i, avaliar(r, ref)) for i, r in enumerate(respostas)]
    return sorted(av, key=lambda x: x[1].nota)


if __name__ == '__main__':
    gab = ('A inflacao ocorre quando ha aumento generalizado de precos na economia. '
           'O banco central controla a inflacao elevando a taxa de juros, o que reduz '
           'o consumo e o investimento, diminuindo a demanda agregada.')
    enun = 'Explique como o banco central combate a inflacao.'
    casos = [
        ('completa',  'O banco central eleva a taxa de juros para combater a inflacao. '
                      'Isso reduz consumo e investimento, diminuindo a demanda agregada '
                      'e o aumento generalizado de precos.'),
        ('parcial',   'O banco central aumenta os juros quando ha inflacao.'),
        ('divagando', 'A inflacao e muito ruim para o pais e prejudica as familias '
                      'pobres, que sofrem no supermercado com o preco da comida.'),
        ('vazia',     'Nao sei responder.'),
    ]
    # em producao, o corpus de stopwords e o material do topico; aqui e minimo
    ref = preparar(gab, enun, stop=stopwords([gab, enun] + [c[1] for c in casos]))
    print(f'conceitos centrais do gabarito: {ref.centrais[:6]}\n')
    for rot, r in casos:
        print(f'  {rot:>10}: {avaliar(r, ref)}')
    print('\n  triagem (mais fraca primeiro):',
          [casos[i][0] for i, _ in triar([c[1] for c in casos], ref)])
