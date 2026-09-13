# M2 — domínio por tópico: a regra em produção contra um modelo

**Base:** EdNet KT3 · **Módulo:** [`dominio.py`](../../trailup_core/dominio.py) · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê) · **Registro da rodada de melhoria:** [MELHORIAS.md](MELHORIAS.md)

---

## 1. Pergunta de negócio

O TrailUp decide quando chamar a LLM com base num número chamado `dominio_estimado`, produzido por uma classe chamada `DeepKnowledgeTracingAnalyzer` (`api/app/services/linear_analysis_pipeline.py:488-531`). Ela não é um modelo — é esta regra:

```
dominio = base ± 0,08 × acertos − 0,07 × erros,  limitado a [0,05, 0,98]
confianca = 0,66   (constante, linha 525)
```

Todo gate escrito sobre `dominio < 0,45` (`state_builder.py:26-30`, `routing.py:43-52`) decide sobre esse número.

> **Este número significa o que o nome diz?** E se não significa, um modelo faria melhor — inclusive no dia 1, com o banco vazio?

---

## 2. Dados

**Fonte:** EdNet KT3 (TOEIC, adultos, autoestudo coreano), licença **CC BY-NC 4.0**.

| | |
|---|---|
| respostas finais | **6.504.124** |
| alunos | 27.792 |
| questões | 11.555 |
| **taxa de acerto global** | **0,668** |
| respostas por aluno — p5 / p25 / mediana / p75 / p95 | 21 / 32 / **70** / 199 / 981 |
| respostas por questão — p5 / p25 / mediana / p75 / p95 | 82 / 200 / **369** / 642 / 1.737 |

**Unidade de análise:** a resposta. Uma linha por vez que um aluno respondeu a uma questão.

### O rótulo estava contaminado, e o conserto mudou a taxa base

Ao reextrair o stream completo do KT3 apareceu isto: no EdNet o aluno pode responder à mesma questão **várias vezes antes do `submit`**, e só a última vale. A extração original guardou **todas** as linhas `respond` como respostas independentes.

- **21,7% das linhas** eram tentativas intermediárias contadas como resposta final;
- não é ruído aleatório — o aluno troca justamente quando errou ou hesitou;
- resultado: **acurácia global medida em 0,577, quando a real é 0,668** — 9 pontos de viés, na direção de fazer os alunos parecerem piores do que são.

Nenhum ajuste de modelo teria compensado isso. Detalhes em [MELHORIAS §6](MELHORIAS.md).

### Desenho de validação

**Split por usuário, 80/10/10** — 22.233 alunos no treino, 2.779 em validação, 2.780 em teste (5,21 M / 632 mil / 661 mil respostas). Prever a próxima resposta de um aluno **já visto no treino** é outro problema, e não é o do TrailUp.

**Features causais:** no passo *i* só entra o que aconteceu antes de *i*.

**Dificuldade da questão estimada só no treino**, com encolhimento para a média global (prior 20).

> **Blindagem contra vazamento.** Features **posteriores à resposta** (latência e trocas da própria questão) não podem entrar no modelo que prevê aquela resposta. O script tem lista explícita (`POSTERIOR`) e `assert` nos argumentos — ver o apêndice para o vazamento que motivou isso.

---

## 3. As métricas, definidas

Quatro métricas, porque elas respondem a perguntas diferentes e a regra falha em umas e não em outras.

**AUC** — pego uma resposta certa e uma errada, ao acaso: qual a chance de o modelo dar probabilidade maior para a certa? Mede **ordenação**. 0,50 é cara ou coroa.

**log-loss** — penalidade logarítmica pela probabilidade atribuída ao que de fato aconteceu. Pune com força quem afirma com confiança e erra. **Referência importante:** chutar sempre a taxa base dá 0,635 aqui. Qualquer coisa **acima** disso é pior que não ter modelo nenhum.

**Brier** — erro quadrático médio da probabilidade. Menor é melhor.

**ECE** (erro esperado de calibração) — agrupa as previsões em faixas e mede a diferença média entre a **probabilidade prevista** e a **frequência observada**. Responde: *o número significa o que diz?* ECE 0,20 quer dizer que, em média, o número erra 20 pontos percentuais.

**AP** (precisão média) e **lift** — usados na §7, onde o alvo é raro.

Intervalos de confiança por bootstrap (120 reamostragens do conjunto de teste).

---

## 4. A regra em produção não é uma probabilidade

| modelo | AUC | IC 95% | log-loss | Brier | ECE |
|---|---|---|---|---|---|
| B0 — prevalência (sempre a taxa base) | 0,500 | — | 0,635 | 0,221 | 0,000 |
| **B1 — regra do TrailUp** | **0,587** | [0,586 – 0,588] | **0,897** | 0,261 | **0,201** |
| B2 — dificuldade da questão | 0,704 | [0,703 – 0,706] | 0,574 | 0,196 | 0,007 |
| B3 — acerto acumulado do aluno | 0,599 | [0,598 – 0,600] | 0,675 | 0,218 | 0,024 |
| B4 — EWMA do aluno no tópico | 0,587 | [0,585 – 0,588] | 0,729 | 0,242 | 0,134 |
| **M2 — gradient boosting (25 features)** | **0,753** | [0,752 – 0,755] | **0,540** | **0,182** | **0,002** |

Três leituras:

**A regra tem log-loss pior que chutar a média** (0,897 contra 0,635). Ela não é uma estimativa ruim — é uma quantidade que **custa mais que não ter estimativa**.

**ECE de 0,201.** O número que ela chama de `dominio_estimado` erra, em média, **20 pontos percentuais**. O modelo erra 0,2 ponto — cem vezes menos.

**Saber a dificuldade da questão (B2) já bate a regra por 0,12 de AUC**, sem olhar para o aluno.

### Onde exatamente a calibração quebra

| a regra diz | o observado é | casos |
|---|---|---|
| 0–20% | **41,8%** | 17.554 |
| 20–40% | 50,7% | 25.791 |
| 40–60% | 58,3% | 67.225 |
| 60–80% | 61,9% | 83.559 |
| **80–100%** (média prevista 94,4%) | **70,9%** | **466.700** |

**71% de todas as decisões caem na última linha**, onde a regra afirma 94% e a realidade é 71%.

**Por que ela satura.** O ponto de equilíbrio é `0,07 / 0,15 = 0,467`: qualquer aluno acima de 46,7% de acerto sobe até o teto e fica lá; abaixo, despenca ao piso. No teste, **42,9% dos passos estão em 0,98** e só 26,7% caem entre 0,2 e 0,8. A regra é quase binária — e `tendencia`, que deriva do nível absoluto e não da variação, fica presa em "ascendente".

### Calibração do modelo, para comparação

| decil | previsto | observado | n |
|---|---|---|---|
| 1 | 28,5% | 28,6% | 66.083 |
| 2 | 44,4% | 44,6% | 66.083 |
| 3 | 53,3% | 53,4% | 66.083 |
| 4 | 60,4% | 60,6% | 66.083 |
| 5 | 66,5% | 66,6% | 66.083 |
| 6 | 72,0% | 72,2% | 66.082 |
| 7 | 77,1% | 77,2% | 66.083 |
| 8 | 82,3% | 82,6% | 66.083 |
| 9 | 88,0% | 88,3% | 66.083 |
| 10 | 94,8% | 95,1% | 66.083 |

Dez decis, erro máximo de 0,5 ponto. **ECE = 0,0021.**

---

## 5. Para aluno novo, a regra é pior que chance

A regra começa em 0,5 e move ±0,08 **sem olhar a questão**. Acertou uma? 0,58 — seja a pergunta trivial ou a mais difícil do banco.

| resposta do aluno | linhas | modelo | IC 95% | regra | IC 95% | só dificuldade |
|---|---|---|---|---|---|---|
| **1ª a 5ª** | 13.900 | **0,723** | [0,716 – 0,732] | **0,486** | [0,477 – 0,494] | 0,709 |
| 6ª a 20ª | 41.700 | 0,735 | [0,731 – 0,739] | 0,576 | [0,571 – 0,582] | 0,705 |
| 21ª a 100ª | 130.775 | 0,745 | [0,742 – 0,748] | 0,572 | [0,570 – 0,575] | 0,699 |
| 101ª+ | 474.454 | 0,755 | [0,753 – 0,756] | 0,578 | [0,576 – 0,580] | 0,704 |

**AUC 0,486, com intervalo inteiramente abaixo de 0,50.** Nas primeiras cinco respostas a regra é **anti-informativa**: ordenar ao contrário do que ela diz seria melhor.

E é exatamente aí que o TrailUp opera — são ~170 sessões utilizáveis na base inteira.

Note também que **"só dificuldade" é praticamente constante** (0,699 a 0,709) em todas as faixas. A questão informa igual no primeiro dia e no centésimo; é o histórico do aluno que precisa acumular.

---

## 6. De onde vem o sinal

| conjunto de features | n | AUC | log-loss |
|---|---|---|---|
| **completo** (comportamento + conteúdo) | 25 | **0,753** | 0,540 |
| só comportamento (sem nada da questão) | 14 | 0,641 | 0,605 |
| só conteúdo (só a questão) | 5 | 0,706 | 0,573 |

**A questão sozinha (0,706) supera todo o histórico comportamental (0,641).**

O ganho não está no histórico do aluno: está na **dificuldade da questão estimada sobre um corpus de respostas**. São complementares — juntos, 0,753 — mas quem carrega é o conteúdo.

**Para o TrailUp isso é o requisito de transferência: sem corpus por questão, o modelo cai para ~0,64** — ainda acima da regra (0,587, e 0,486 no início), mas longe do teto.

**E o corpus se forma rápido.** AUC por número de vezes que a questão foi respondida no treino:

| questão vista | AUC |
|---|---|
| 21–100 vezes | 0,760 |
| 101–1.000 | 0,761 |
| 1.000+ | 0,735 |

**Vinte e uma respostas já entregam o resultado do corpus maduro** — uma turma de 30 alunos passando uma vez pela questão. A janela fria é curta.

---

## 7. O gate: treinar no alvo que ele decide

### O alvo tem de ser o que o gate decide

O gate não pergunta *"o aluno acerta a próxima?"*. Pergunta *"este aluno vai travar neste tópico?"*. Usar `1 − p(acerta a próxima)` como proxy **não serve**: uma questão difícil derruba esse `p` sem que o aluno esteja travado.

Medido: o proxy dá AUC 0,698 contra 0,761 da regra que ele substituiria. **O proxy é pior que a regra.**

### O alvo direto

```
vai_travar = 1 se a taxa de acerto do aluno nas PRÓXIMAS 10 respostas
               do mesmo tópico ficar abaixo de 50%
```

577.878 amostras de teste, **taxa base 12,3%**.

> A taxa base de 12,3% é baixa: use **lift** e **precisão no ponto de operação**, não acurácia.

### Resultado

| critério | AUC | IC 95% | AP | IC 95% | precisão @10% | lift | cobertura |
|---|---|---|---|---|---|---|---|
| **modelo do gate** | **0,779** | [0,778 – 0,781] | **0,378** | [0,374 – 0,382] | **41,6%** | **3,39×** | 34% |
| acerto acumulado no tópico | 0,735 | [0,733 – 0,737] | 0,279 | [0,276 – 0,282] | 36,1% | 2,94× | 29% |
| regra do TrailUp | 0,703 | [0,701 – 0,705] | 0,299 | [0,296 – 0,302] | 35,5% | 2,90× | 31% |
| EWMA no tópico | 0,675 | [0,674 – 0,677] | 0,236 | [0,234 – 0,239] | 27,8% | 2,27× | 23% |

**Lift** = quantas vezes a precisão do alerta supera a taxa base. **Cobertura** = que fração de todos os que travam o alerta pega.

### Ponto de operação

| dispara em | modelo: precisão / lift / cobertura | regra: precisão / lift / cobertura |
|---|---|---|
| 5% | 51,6% / 4,21× / 21% | 46,6% / 3,80× / 19% |
| **10%** | **41,6% / 3,39× / 34%** | 35,5% / 2,90× / 31% |
| 15% | 35,8% / 2,92× / 44% | 31,3% / 2,55× / 39% |
| 20% | 31,9% / 2,60× / 52% | 27,9% / 2,27× / 47% |
| 30% | 26,6% / 2,17× / 65% | 22,3% / 1,82× / 60% |

**Em 10% de disparo o modelo acerta 41,6% contra 35,5% da regra — 17% a mais de precisão, com cobertura maior.**

A vantagem é real: **+0,076 de AUC, +0,079 de AP**.

### O gate também está calibrado

| decil | previsto | observado | n |
|---|---|---|---|
| 1 | 0,8% | 0,6% | 57.788 |
| 5 | 8,1% | 8,0% | 57.788 |
| 10 | 40,6% | 41,6% | 57.788 |

**ECE = 0,0039, Brier = 0,0921.** Isso é o que permite **mexer no limiar sem recalibrar**: se 10% de disparo for caro demais, 5% é uma mudança de parâmetro, não de modelo.

### Cada disparo evitado custa quanto

Cada disparo é uma passagem por `agente_conteudo` e, quando `gerar_materiais` abre, por `agente_geracao_midia` — a chamada mais cara do sistema: **13 a 29 chamadas de texto + TTS + 10 a 18 imagens por perfil × tópico**.

---

## 8. São dois modelos, não um

| modelo | pergunta | consome | AUC |
|---|---|---|---|
| `m2_model_v3` | o aluno acerta esta questão? | `dominio_estimado`, `confianca`, `tendencia` — o que o professor vê | 0,753 |
| `m2_gate_v3` | o aluno vai travar neste tópico? | decide o nível do gate | 0,779 |

**Usar um no papel do outro degrada o gate** — ver o apêndice.

---

## 9. O que rendeu pouco

| aposta | ganho de AUC |
|---|---|
| **189 tags de habilidade** do EdNet | +0,001 |
| **latência e hesitação** por questão | +0,003 |
| janela recente, sessão, tempo relativo | +0,007 |

A primeira era a hipótese forte — decompor por skill é o núcleo do knowledge tracing. Deu ruído. A razão provável: `acc_prev_part` e `ewma_part` já capturam quase toda a estrutura de habilidade, porque **as tags são fortemente aninhadas dentro de `part`**. A [análise de estrutura](../dificuldade/FAIXA_DIFICULDADE.md) confirma por outro caminho: neste corpus a dificuldade é unidimensional.

Há também um teto conhecido: **0,76 é o patamar do DKT na literatura para EdNet**; o SAINT+ chega a 0,79 com transformer sobre a sequência inteira. O limite aqui não é de ajuste, é de arquitetura — e um modelo de sequência não se justifica numa API que hiberna, para ganhar 0,03.

**Mais features renderam pouco. O que rendeu foi olhar para o dado** — o alvo do gate (§7) e a contaminação do rótulo (§2).

---

## 10. O que isso significa para o produto

1. **Trocar a regra é ganho garantido, não aposta.** Em todo corte testado o modelo é melhor, e no início da vida do aluno a diferença é entre informação e ruído (0,723 contra 0,486).

2. **O valor principal é calibração, não ranking.** ECE de 0,201 para 0,002. O número passa a significar o que diz — é o que torna o limiar do gate ajustável e a `confianca` real. Hoje ela é a constante 0,66.

3. **`confianca` pode ser derivada de verdade.** O ECE fica entre 0,002 e 0,004 em todas as faixas de observação; o que varia é a AUC. **Confiança escala com *n*; calibração não precisa de correção.**

4. **Requisito crítico: corpus de dificuldade por questão.** `questao_aluno` é chaveada por `(aluno_id, questao_id, tentativa)` na questão **base**, então acumula. Mas questões **personalizadas não gravam em `questao_aluno`** (`QuestionActivity.tsx:1281`) — e são elas a maior parte do fluxo adaptativo. É o que separa 0,64 de 0,75, e o limiar é de apenas ~21 respostas por questão. **Se as personalizadas não gravarem, o sistema nunca sai da janela fria.**

5. **Cabe na API que hiberna:** 1,7 a 2,8 MB, sem dependência nativa (`HistGradientBoostingClassifier`; LightGBM foi descartado por exigir `libomp`).

6. **`aluno_topico_dominio_hist` deixa de ser opcional.** Metade das features é acumulada por tópico; o upsert atual destrói a trajetória a cada ciclo.

7. **A decisão de licença vem antes do deploy.** Com o banco vazio não há como treinar em dado próprio, então o que iria ao ar são pesos derivados do EdNet (**CC BY-NC**). Uso acadêmico está coberto; produto comercial não. Saídas: (a) assumir escopo acadêmico e registrar em ADR; (b) usar só a *receita* — features, hiperparâmetros, o achado das 21 respostas — e ajustar os coeficientes na primeira turma; (c) procurar um dataset de knowledge tracing permissivo, que não existe nesta pasta.

---

## 11. Limites

- **Domínio distante.** TOEIC, app de autoestudo coreano, adulto pagante, sem turma nem professor. `part` (1–7) faz o papel de tópico, o que é **analogia, não equivalência**.
- **Viés de seleção.** Só usuários com ≥20 respostas. Em amostra sem filtro a taxa de acerto seria mais baixa.
- **Licença CC BY-NC.** Nenhum peso treinado aqui vai para produto comercial. O que transfere é a receita.
- **A transferência fica sem medida até haver aluno real.** O banco está vazio, então não há replay possível. O que está demonstrado é que *a família de modelo* bate *a família de regra* em todos os cortes testados, inclusive nos mais pobres — **não o quanto ela bate no TrailUp**. A primeira turma real é o teste, e precisa ser instrumentada para servir de medição: o modelo grava `p` e o resultado, e compara-se depois.
- **Não testado:** modelo sequencial (DKT/SAKT/SAINT). A literatura reporta 0,76–0,79 em EdNet; este tabular chegou a 0,753 com 25 features — a margem provavelmente não paga a complexidade.
- **Os IC são estreitos porque o teste é grande** (661 mil respostas de 2.780 alunos). Eles medem incerteza amostral, **não** incerteza de transferência de domínio, que é muito maior e não está quantificada em lugar nenhum deste documento.

---

## Apêndice — o que mudou, e por quê

Três correções, em ordem de impacto.

### 1. O rótulo estava contaminado em 21,7%

Descrito na §2, porque é propriedade do dado e não só histórico. Tudo que foi treinado antes da descoberta usou um rótulo que contava tentativas intermediárias como respostas finais, e a acurácia global medida era **0,577 quando a real é 0,668**.

O efeito não foi uniforme: a contaminação **inflava o valor do comportamento**, porque as tentativas intermediárias *são* comportamento.

| | rótulo contaminado | rótulo correto |
|---|---|---|
| só comportamento | 0,683 | **0,641** |
| só a questão | 0,667 | **0,706** |
| regra do TrailUp | 0,612 | **0,587** |
| modelo completo | 0,755 | 0,753 |

A leitura inverteu: antes o comportamento parecia empatar com a questão; corrigido, **a questão sozinha supera todo o histórico do aluno**.

### 2. O gate estava sendo treinado no alvo errado

A proposta original usava `1 − p(acerta a próxima)` como sinal do gate. Medido contra o alvo real ("vai travar no tópico"), esse proxy dá **AUC 0,698 contra 0,761 da regra que ele substituiria** — teria ido a produção piorando o gate.

Daí os **dois modelos** da §8: um para o número que o professor vê, outro para a decisão do gate.

### 3. Um vazamento por aspas de shell

Uma rodada passou os argumentos errados por um problema de quoting no laço (`for a in "proxima base"` faz `ALVO="proxima base"`, então o guard `if ALVO == 'proxima'` nunca dispara). Features **posteriores à resposta** — latência e trocas da própria questão — entraram no modelo que prevê aquela resposta: **AUC 0,764, contaminado**.

A blindagem: lista explícita `POSTERIOR` e `assert` nos argumentos.

### Números que mudaram entre versões

| | antes | final |
|---|---|---|
| corpus | 8,7 M respostas | **6,5 M** respostas finais |
| taxa de acerto global | 0,577 | **0,668** |
| regra do TrailUp | AUC 0,612, ECE 0,222 | **0,587**, ECE 0,201 |
| regra, aluno novo | 0,478 | **0,486** (IC inteiro abaixo de 0,50) |
| só comportamento | 0,683 | **0,641** |
| gate | 0,806 (alvo contaminado) | **0,779** |
