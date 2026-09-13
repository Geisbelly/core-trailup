# Avaliador de resposta discursiva — quatro tentativas de fechar o gap

**Data:** 2026-09-12, revisto em 2026-09-13 · **Base:** classEx · **Módulo:** [`pre_avaliacao.py`](../modulo/pre_avaliacao.py)

> **Revisão de 2026-09-13.** Duas correções. A tabela de pré-filtro publicada antes reportava 5–8% de erro nas pontas; a reexecução do script dá **12–14%**, e a afirmação de que a ponta alta é mais confiável que a baixa **não se sustenta** — são equivalentes. E o "teto de 0,899" precisa de ressalva: ele é a concordância de **uma execução com a média das três**, que inclui a própria execução. A comparação honesta é entre execuções independentes: **0,871 a 0,897**.

---

## 1. Pergunta de negócio

Toda resposta aberta do TrailUp hoje vai para a LLM — é a chamada mais frequente do fluxo. A pergunta é se dá para evitar parte disso:

> **Dá para estimar a nota de uma resposta discursiva sem chamar a LLM?**

E, se não der com qualidade de nota, dá para fazer algo útil mais barato — ordenar para revisão, dar feedback provisório, filtrar as pontas.

---

## 2. Dados

**Fonte:** classEx — Geschwind et al. (2026), licença **CC BY**. Respostas abertas de universitários alemães em economia, com enunciado, gabarito e nota.

| | |
|---|---|
| respostas | **1.167** |
| alunos | 249 |
| tarefas distintas | 8 |
| tamanho da resposta (caracteres) — p10 / mediana / p90 | 203 / **384** / 624 |
| tamanho do gabarito — mediana | 466 |

**Unidade de análise:** a resposta. **Validação agrupada por aluno** — nenhum aluno aparece no treino e no teste ao mesmo tempo.

### O rótulo, e sua limitação mais séria

A nota de conteúdo (1–5) é a **média de três execuções do GPT-4**. Distribuição:

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| 0,5% | 10,9% | 28,1% | **59,1%** | 1,4% |

Média 3,50, desvio **0,70**. Na prática **87% das notas são 3 ou 4**. Isso é quase um problema binário com duas classes desbalanceadas, e limita estruturalmente qualquer correlação: não há variação para explicar nas pontas.

> **Não é nota humana.** É um LLM avaliando, e o alvo do modelo é reproduzir o julgamento dele. Isso serve para medir *quanto de um julgamento de LLM cabe em features de superfície* — que é a pergunta de custo. Não serve para afirmar que o resultado corresponde à correção de um professor.

---

## 3. O teto: quanto o próprio LLM concorda consigo mesmo

Se as três execuções do GPT-4 discordam entre si, nenhum modelo pode superar essa concordância — ela é o limite do alvo.

| par de execuções | Spearman | QWK | concordância exata |
|---|---|---|---|
| 1 × 2 | 0,871 | 0,879 | 86,9% |
| 1 × 3 | 0,883 | 0,883 | 87,6% |
| 2 × 3 | 0,897 | 0,906 | 90,1% |
| *1 × média das três* | *0,955* | — | — |

**O teto honesto é ≈0,88**, não 0,955: a última linha compara uma execução com uma média que **a contém**, o que infla a concordância por construção. A versão anterior deste documento usava 0,899, que é dessa família.

---

## 4. As métricas

**Spearman** — correlação entre a ordem prevista e a ordem real. Mede se o modelo **ordena** certo, sem exigir que acerte o valor. Vai de −1 a 1.

**QWK** (kappa quadrático ponderado) — concordância entre duas notas de 1 a 5, descontando o acaso, penalizando mais os erros grandes. 0 = acaso, 1 = concordância perfeita. É a métrica padrão de correção automática de redação.

**MAE** — erro absoluto médio, em pontos da escala de 1 a 5.

Intervalos de confiança por **bootstrap de alunos** (não de respostas): respostas do mesmo aluno não são independentes, e reamostrar respostas daria intervalos falsamente estreitos.

---

## 5. Tentativa 1 — grafo de relação entre resposta e gabarito

**A ideia:** cada texto vira um grafo — conceitos são nós, coocorrência numa janela de 4 palavras gera arestas. Comparar resposta com gabarito deixa de ser "tem as mesmas palavras" e passa a ser "**liga as mesmas coisas**". Camada semântica por LSA treinada no próprio corpus, para casar conceitos que não são a mesma palavra, sem baixar modelo.

| modelo | Spearman | QWK | MAE |
|---|---|---|---|
| só o comprimento da resposta | 0,253 | 0,219 | 0,554 |
| similaridade (TF-IDF + LSA) + comprimento | 0,401 | 0,302 | 0,511 |
| grafo: **só os nós** (conceitos) | 0,482 | 0,387 | 0,481 |
| grafo: nós + **arestas** (relações) | 0,488 | 0,388 | 0,478 |
| grafo + **semântica LSA** | 0,488 | 0,391 | 0,479 |
| grafo + similaridade | **0,501** | **0,415** | 0,471 |

**O grafo bate a similaridade simples: +0,10 de Spearman. Mas não pelo motivo da ideia.**

- a parte **relacional** (arestas) acrescenta **+0,006**
- a camada **semântica** (LSA) acrescenta **+0,000**

Correlação isolada de cada feature com a nota:

| feature | o que é | Spearman |
|---|---|---|
| `no_peso` | cobertura dos conceitos do gabarito, ponderada por frequência | **+0,408** |
| `cov_med` | cobertura ponto a ponto das frases do gabarito (encoder) | +0,399 |
| `no_cobertura` | fração dos conceitos do gabarito presentes na resposta | +0,396 |
| `cos_tfidf` | cosseno TF-IDF entre resposta e gabarito | +0,392 |
| `emb_cos_gab` | cosseno de *embedding* com o gabarito | +0,358 |
| `cob_conceitos_centrais` | cobertura dos 10 conceitos mais centrais do grafo | +0,343 |
| `log_len` | comprimento da resposta | +0,303 |
| `aresta_peso` | cobertura das **arestas** do gabarito | +0,291 |
| `divagacao` | fração da resposta que não toca o gabarito | **−0,193** |

**A moldura de grafo produziu features melhores; a estrutura de grafo, quase nada.** O que carrega é cobertura do gabarito e penalização de divagação — que são medidas de conjunto, não de relação.

---

## 6. Tentativa 2 — enriquecer a referência com corpus do domínio

**Hipótese:** o gabarito é pobre (mediana 466 caracteres) e por isso a semântica não tem substrato. Enriquecer deveria ajudar.

Testado sem raspar nada: usar as respostas **bem avaliadas de outros alunos** da mesma tarefa como corpus de domínio (o aluno avaliado nunca entra na própria referência; só respostas do treino de cada dobra).

| referência | Spearman | QWK |
|---|---|---|
| **só o gabarito** | **0,486** | **0,415** |
| + 5 melhores respostas | 0,497 | 0,390 |
| + 15 melhores | 0,490 | 0,407 |
| + 40 melhores | 0,428 | 0,333 |
| + todas as do treino | 0,396 | 0,361 |

**Enriquecer piora**, e o modo de falhar é estrutural: quanto mais texto entra na referência, mais **genérica** ela fica — qualquer resposta cobre parte dela, e a cobertura perde poder discriminativo.

**O gabarito funciona por ser preciso, não por ser rico.**

---

## 7. Tentativa 3 — mais dados de treino

| exemplos de treino | Spearman | ganho |
|---|---|---|
| 93 | 0,383 | — |
| 186 | 0,433 | +0,051 |
| 326 | 0,455 | +0,022 |
| 466 | 0,460 | +0,005 |
| 653 | 0,483 | +0,024 |
| 793 | 0,498 | +0,015 |
| 933 | 0,490 | −0,007 |

Dobrar de 466 para 933 rendeu **+0,030**. A curva está no regime plano: chegar perto de 0,88 exigiria ordens de magnitude a mais. **Não é volume que falta.**

---

## 8. Tentativa 4 — encoder multilíngue pré-treinado

**Hipótese:** o gargalo é a representação. Features de superfície medem "mencionou os conceitos certos", não "disse algo verdadeiro sobre eles".

`paraphrase-multilingual-MiniLM-L12-v2`, incluindo **cobertura ponto a ponto** — quebrar o gabarito em frases e medir se cada uma encontra correspondência semântica na resposta (o mais próximo de como um humano corrige).

| modelo | Spearman | QWK | MAE |
|---|---|---|---|
| grafo (superfície) | 0,488 | 0,388 | 0,478 |
| encoder: só o cosseno do texto inteiro | 0,365 | 0,301 | 0,521 |
| encoder + cobertura ponto a ponto | 0,476 | 0,400 | 0,480 |
| **grafo + encoder (18 features)** | **0,532** | 0,433 | 0,450 |
| *(teto: execuções independentes do LLM)* | *≈0,88* | *≈0,88* | — |

**O encoder não é melhor que as features de superfície — é complementar.** Sozinho dá 0,476 contra 0,488 do grafo; juntos, 0,532. Melhor resultado da série, e ainda a **60% do caminho**.

Em tarefa nunca vista (leave-one-Task-out): **0,429**, variando de 0,267 a 0,639 entre as 8 tarefas.

---

## 9. O que isso conclui

Quatro caminhos independentes, quatro respostas:

| caminho | efeito |
|---|---|
| estrutura relacional (grafo) | +0,006 |
| referência mais rica | **negativo** |
| mais dados | +0,030 ao dobrar, achatando |
| encoder pré-treinado | +0,057 |

O gap não é de dado, nem de representação isolada, nem de estrutura. **É de julgamento:** decidir se um argumento está *correto* exige raciocinar sobre o conteúdo — que é exatamente o que o LLM faz e o que features de similaridade não fazem, por mais sofisticadas.

---

## 10. Posicionamento realista

### Como pré-filtro de dois lados — mais fraco do que eu havia reportado

Dispensar a LLM nas duas pontas e mandar o meio para ela. Erro = a decisão automática estava errada (ponta baixa marcada como "precisa de feedback" mas a nota era ≥3,5; ponta alta marcada como "adequada" mas era <3,5):

| corte | LLM em | erro na ponta baixa | erro na ponta alta |
|---|---|---|---|
| 5% / 95% | 90% | **14%** | **12%** |
| 10% / 90% | 80% | 15% | 12% |
| 15% / 85% | 70% | 19% | 14% |
| 20% / 80% | 60% | 25% | 12% |
| 25% / 75% | 50% | 25% | 14% |

**Correção:** a versão anterior reportava 5–8% de erro e afirmava que a ponta alta é mais confiável que a baixa. A reexecução mostra **12–14% nas duas**, e simetria entre elas. Um corte em 10%/90% economiza 20% das chamadas ao custo de errar uma em oito nas pontas — decisão de produto, não obviedade.

### Como triagem — é aqui que funciona

| recorte | nota real média |
|---|---|
| as 10 de menor previsão | **1,87** |
| as 20 de menor previsão | 2,27 |
| as 50 de menor previsão | 2,56 |
| *geral* | *3,50* |

As dez piores previstas têm média **1,87 contra 3,50 do geral** — em uma escala cujo desvio é 0,70, são mais de dois desvios abaixo. **Ordenar exige menos que pontuar**, e a triagem funciona bem mesmo com Spearman 0,53.

**É este o uso que o módulo implementa** (`triar()`), junto com o feedback provisório — quais conceitos do gabarito faltaram — enquanto a LLM trabalha.

---

## 11. Sobre buscar questões de concurso na internet

A proposta de montar corpus com questões públicas do mesmo assunto esbarra em duas coisas antes da técnica:

1. **Enriquecer a referência piorou** (§6). Se o corpus de domínio for usado como referência de comparação, o efeito esperado é negativo.
2. **Questões de concurso costumam ser protegidas por direito autoral.** Usar para treinar é uma discussão; redistribuir dentro do produto é outra. Precisa de decisão antes da coleta, não depois.

O uso em que o corpus de domínio **poderia** ajudar é outro: treinar o *encoder* no vocabulário da matéria (adaptação de domínio), não usá-lo como referência. É experimento diferente e mais caro, e a §8 sugere retorno modesto.

---

## 12. Limites

- **O rótulo é LLM, não humano.** Mede quanto de um julgamento de LLM cabe em features de superfície — não corretude pedagógica.
- **87% das notas são 3 ou 4.** Pouca variação para explicar; o Spearman é medido num alvo quase binário, o que o comprime.
- **Alemão, economia, universitário, 8 tarefas.** As features são agnósticas de idioma (operações de conjunto sobre tokens), mas os **pesos ajustados não são** — é por isso que `pre_avaliacao.py` é para ordenar, não para dar nota, até haver ~100 respostas em português com nota humana.
- **Leave-one-Task-out cai para 0,429** e varia de 0,267 a 0,639 entre tarefas. Em tarefa nova o desempenho é imprevisível.
- **249 alunos, 1.167 respostas.** Os IC por aluno (§4) já refletem isso: Spearman 0,532 tem incerteza de ±0,05.
