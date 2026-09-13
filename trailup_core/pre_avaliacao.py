"""Pre-avaliacao rapida de resposta discursiva.

NAO substitui a LLM. Da uma estimativa instantanea enquanto ela trabalha, e
ordena respostas para o professor revisar.

Medido no classEx (1.167 respostas, validacao agrupada por aluno):

    versao anterior (8 features)              Spearman 0,451 | QWK 0,318
    esta versao (10 features, sem dependencia) Spearman 0,463 | QWK 0,350
    boosting sobre as mesmas features         Spearman 0,429 | QWK 0,398
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
JANELA = 4            # coocorrencia: palavras a ate 4 posicoes formam aresta
MIN_DOCS = 10         # abaixo disso nao da para estimar stopword por frequencia
DF_STOPWORD = 0.5     # aparece em mais da metade dos documentos -> stopword

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
    divagacao: float            # 0 a 1 - quanto fala de fora
    conceitos_faltando: list[str]
    def __str__(self):
        falta = (', '.join(self.conceitos_faltando[:4]) or 'nenhum dos centrais')
        return (f'~{self.nota:.1f}/5 | cobre {self.cobertura:.0%} do gabarito, '
                f'divaga {self.divagacao:.0%} | falta: {falta}')


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
             10             2,10          2,10       3,50
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
