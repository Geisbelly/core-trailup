# Diário — avaliador discursivo

## 13/09/2026 — definição do problema

Escolhemos estimar a qualidade de uma resposta discursiva comparando a
resposta ao gabarito por tokens, conceitos e relações de coocorrência. O
objetivo operacional é reduzir chamadas desnecessárias à LLM e ordenar
respostas para revisão.

## 13/09/2026 — dados e validação

Usamos o dataset classEx, com 1.167 respostas, 249 alunos e 8 tarefas. O
desfecho disponível é a média de três avaliações do GPT-4; não há nota humana
no arquivo. A validação é agrupada por aluno para impedir que respostas do
mesmo aluno vazem entre treino e teste.

## 13/09/2026 — tentativas

Testamos comprimento, similaridade TF-IDF/LSA, cobertura de conceitos e
arestas do grafo, enriquecimento do gabarito e encoder multilíngue. O melhor
resultado foi Spearman 0,532 com grafo mais encoder. A estrutura das arestas
acrescentou pouco, e enriquecer o gabarito piorou a ordenação.

## 13/09/2026 — decisão final

O módulo não dá nota automática nem afirma porcentagem real de acerto. Ele
expõe a faixa ordinal, a nota com a incerteza e os conceitos ausentes. O uso
aprovado é triagem; respostas intermediárias devem continuar para a LLM ou
para revisão humana.

## 13/09/2026 — auditoria do percentual de exibição

O percentual de exibição foi medido contra a referência: erro absoluto médio
de 13,0 pontos e viés de +5,9 pontos. É erro demais para um número de dois
dígitos que o leitor lê como nota, então `__str__` passou a mostrar a FAIXA,
que separa em ordem e sem cruzar (nota real média 32,2% / 61,2% / 70,5%).

Calibrar o viés foi testado com holdout por tarefa: remove o viés mas não
reduz o erro (ganho -0,48, IC95 [-1,34, +0,19]). Não aplicado.

Dois defeitos menores no mesmo bloco: a faixa comparava o percentual
ARREDONDADO, então 3,9999 virava 75,0 e subia de faixa sozinho; e o
percentual saía de 0-100 quando o PreAvaliacao era construido direto, porque
a trava do 1-5 esta no avaliar e nao na classe. Ambos com teste.

Limitação pendente: coletar cerca de 100 respostas em português com avaliação
humana para recalibrar os pesos e medir validade no domínio do TrailUp.
