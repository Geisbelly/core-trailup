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
expõe um percentual de exibição derivado da escala 1–5, a faixa ordinal e os
conceitos ausentes. O uso aprovado é triagem; respostas intermediárias devem
continuar para a LLM ou para revisão humana.

Limitação pendente: coletar cerca de 100 respostas em português com avaliação
humana para recalibrar os pesos e medir validade no domínio do TrailUp.
