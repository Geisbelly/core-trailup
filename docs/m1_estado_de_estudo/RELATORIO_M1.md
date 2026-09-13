# M1 — estado de estudo: resultado negativo, e o que funciona no lugar

**Data:** 2026-09-12, revisto em 2026-09-13 · **Base:** ARES / OSF 8y3zp · **Modelo:** `m1_model.pkl` (2 KB)

> **Revisão de 2026-09-13.** A §6 é nova e muda a leitura do resultado positivo: a "base pessoal do aluno", que dava AUC 0,784, **não distingue nada dentro de um mesmo aluno** (AUC intra-aluno 0,379). Ela mede estilo de resposta, não estado. O único sinal que funciona por dentro é a dificuldade do texto.

---

## 1. Pergunta de negócio

O TrailUp tem seis "analisadores" na `linear_analysis_pipeline.py` que, a partir de telemetria (toques, pausas, tempo de tela), afirmam se o aluno está **fluindo, travado ou disperso**, e disparam intervenção da LLM com base nisso. Nenhum deles foi validado — são regras escritas à mão com nomes de modelo.

> **Este aluno está travado agora, e dá para saber isso pelo comportamento dele nos últimos minutos?**

---

## 2. Dados

**Fonte:** ARES — Lee et al. (2025), *Data in Brief*, licença **CC BY 4.0**. Logs xAPI de **200 universitários** lendo textos em inglês como segunda língua, por 6 semanas, numa plataforma de leitura assistida.

**Por que este dataset e não outro:** é o único da pasta que tem, no mesmo registro, **comportamento instrumentado** e **estado declarado pelo próprio aluno**. Sem as duas coisas juntas não há como testar a pergunta — só há como assumir a resposta, que é o que o código faz hoje.

| | |
|---|---|
| alunos | 200 |
| textos distintos | 82 |
| sessões | 907 |
| leituras extraídas | 1.866 |
| **leituras com dificuldade declarada** | **1.683** (90,2%) |

**Unidade de análise: a leitura** = `(aluno, sessão, texto)`. Uma linha por vez que um aluno leu um texto numa sessão. Sessão = bloco separado do seguinte por mais de 30 minutos.

| duração da leitura | |
|---|---|
| p25 | 295 s |
| **mediana** | **555 s** (9 min) |
| p75 | 1.055 s |

Mediana de **9 leituras por aluno** (p25 = 6, p75 = 11).

### Duas correções de extração, registradas nos scripts

O primeiro corte quebrava a leitura **a cada modal de explicação de palavra**, produzindo "episódios" de 25 s que não são leitura nenhuma. E a nota de estrela não carrega `assignmentId` no evento, então precisa ser atribuída por posição temporal dentro da sessão — e consolidada pelo *último* valor, porque arrastar a estrela emite vários eventos intermediários.

---

## 3. O desfecho: o que "travado" quer dizer aqui

Depois de cada texto, o aluno declara **dificuldade de 1 a 5**. O alvo é:

```
travado = 1  se a dificuldade declarada for 4 ou 5
travado = 0  se for 1, 2 ou 3
```

**517 de 1.683 leituras — 30,7%.** Essa é a taxa base contra a qual tudo se lê.

Distribuição da nota declarada:

| 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| 8,2% | 14,7% | **46,4%** | 18,8% | 11,9% |

**Quase metade das notas é exatamente 3.** Isso limita qualquer modelo: o instrumento tem pouca resolução, e o teto de acerto não é o teto do método.

O corte em ≥4 é escolha minha, feita para ter classes utilizáveis. Testei também `dificuldade = 5` e `interesse ≤ 2` como alvos alternativos (§4).

---

## 4. As medidas de comportamento

Onze quantidades calculadas sobre a sequência de eventos da leitura — nada declarado, nada inferido:

| feature | o que conta |
|---|---|
| `dur` | segundos entre o primeiro e o último evento da leitura |
| `ativo_sec` | soma dos intervalos **menores que 30 s** — tempo "com o aluno presente" |
| `ativo_ratio` | `ativo_sec / dur` — fração do tempo em que houve atividade |
| `n_ev` | total de eventos |
| `ev_min` | eventos por minuto |
| `n_press` | cliques |
| `n_focus` | vezes que a janela do texto ganhou foco |
| `n_token` | palavras tocadas (pedido de tradução/significado) |
| `gap_med`, `gap_p90` | intervalo mediano e p90 entre eventos consecutivos |
| `frac_pausa` | fração dos intervalos acima de 30 s |
| `revisita` | quantas vezes o aluno já tinha lido aquele texto |

São os análogos diretos do que os analisadores do TrailUp usam: tempo de tela, ociosidade, densidade de interação, pausa.

**Uma limitação do instrumento, importante para ler o resultado:** `ativo_ratio` tem **mediana de 0,115**. Em 88% do tempo de leitura não há evento nenhum — porque **quem lê não clica**. O ARES instrumenta clique, não leitura.

---

## 5. Métrica e desenho de validação

**AUC** — pego uma leitura em que o aluno declarou dificuldade alta e uma em que não, ao acaso: qual a chance de o modelo dar valor maior para a primeira? 0,50 é cara ou coroa.

**AP** (precisão média) — área sob a curva precisão-recall; com taxa base de 30,7%, um modelo sem informação tem AP ≈ 0,307.

**Brier** — erro quadrático médio da probabilidade. Menor é melhor.

**ECE** — erro esperado de calibração: diferença média entre a probabilidade prevista e a frequência observada, por faixa. Diz se o número significa o que diz.

**Validação:** `GroupKFold` de 5 dobras **agrupada por aluno** — nenhum aluno aparece no treino e no teste ao mesmo tempo. Prever a próxima leitura de um aluno já visto é outro problema, e não é o do TrailUp.

**A base pessoal é construída causalmente:** para a leitura *k* de um aluno, usa só as leituras 1 a *k−1* dele, com suavização para a média global (prior 3). A dificuldade do texto vem só do conjunto de treino, com prior 25.

Intervalos de confiança por bootstrap (600 reamostragens).

---

## 6. Resultado

| modelo | AUC | IC 95% | AP | Brier | ECE |
|---|---|---|---|---|---|
| **só comportamento** (11 features, boosting) | **0,505** | [0,474 – 0,535] | 0,315 | 0,227 | 0,097 |
| só dificuldade do texto | 0,581 | [0,552 – 0,612] | 0,369 | 0,209 | 0,011 |
| **só base pessoal do aluno** | **0,784** | [0,757 – 0,809] | 0,668 | 0,163 | 0,041 |
| as duas bases + interação | **0,789** | [0,763 – 0,813] | 0,667 | 0,160 | 0,029 |
| as duas bases + comportamento | 0,781 | [0,755 – 0,806] | 0,655 | 0,163 | 0,033 |
| chutar a taxa base | 0,500 | — | 0,307 | 0,213 | 0,000 |

**O comportamento é acaso.** AUC 0,505, com intervalo [0,474 – 0,535] que contém 0,50 confortavelmente. Não é um modelo mal ajustado — é ausência de sinal.

**E acrescentá-lo piora:** as duas bases sozinhas dão 0,789; somando as 11 features de comportamento, 0,781. O comportamento não acrescenta pouco: ele acrescenta ruído que custa 0,008.

Testei também as outras famílias e os outros alvos, com o mesmo resultado:

| variação | AUC |
|---|---|
| logística em vez de boosting | 0,480 |
| random forest | 0,495 |
| só `dur` | 0,505 |
| só `ativo_ratio` | 0,514 |
| alvo `dificuldade = 5` | 0,45 – 0,53 |
| alvo `interesse ≤ 2` | 0,44 – 0,54 |
| alvo puramente comportamental (texto expirou / nunca submetido) | 0,50 |

Nenhuma família, nenhum conjunto de features, nenhum alvo passa de 0,53.

---

## 7. O teste que faltava: a base pessoal é estilo, não estado

O resultado "base pessoal AUC 0,784" parecia o achado positivo do estudo. Ele exige um teste que eu não tinha feito.

A suspeita: a base pessoal é a média das declarações anteriores daquele aluno. Se existe aluno que **marca 4 em tudo** e aluno que **marca 2 em tudo**, ela acerta muito só separando os dois tipos de pessoa — sem nunca dizer *qual leitura* foi difícil para um aluno específico. E é exatamente isso que o produto precisa saber.

**A separação é medível:** calcular o AUC **dentro de cada aluno** e agregar os pares. Se a medida só captura estilo, o AUC intra-aluno cai para 0,50.

| modelo | AUC geral | **AUC intra-aluno** |
|---|---|---|
| só comportamento | 0,505 | **0,500** |
| só dificuldade do texto | 0,581 | **0,654** |
| **só base pessoal do aluno** | **0,784** | **0,379** |
| as duas bases + interação | 0,789 | 0,561 |

*(1.853 pares utilizáveis — pares de leituras do mesmo aluno com desfechos diferentes.)*

Três leituras deste quadro:

**A base pessoal não distingue nada por dentro — distingue ao contrário.** 0,379 está abaixo de 0,50. Todo o 0,784 dela é diferença *entre pessoas*. O valor abaixo de 0,50 tem explicação mecânica: a base é a média acumulada das declarações anteriores, então ela **sobe depois que o aluno declara dificuldade** e a declaração seguinte tende a ser mais baixa — regressão à média vira anticorrelação.

**A dificuldade do texto melhora quando se olha por dentro** (0,581 → 0,654). Ela é o único componente que de fato responde "esta leitura foi difícil".

**O comportamento continua em exatamente 0,500.** Coerente com a §6: não há sinal ali para encontrar.

Os números da população sustentam a leitura de estilo: a variância da nota **entre alunos** (0,534) é quase três vezes a variância **entre textos** (0,196). E dos 200 alunos, **61 nunca declararam 4 ou 5** e **12 declararam sempre** — 36,5% da amostra é constante.

> **Consequência de produto:** se o TrailUp exibir a base pessoal como "nível de dificuldade percebida", estará exibindo o temperamento do aluno como se fosse o estado dele. Serve para **calibrar a escala** do check-in daquele aluno — "para você, isto é um 4" — e não para afirmar que ele está travado neste material.

---

## 8. Quanto check-in é preciso

Usando o modelo das duas bases, por número de check-ins anteriores do próprio aluno:

| check-ins anteriores | casos | AUC | IC 95% | ECE |
|---|---|---|---|---|
| 0 (primeira leitura) | 200 | 0,549 | [0,460 – 0,638] | 0,095 |
| 1–2 | 385 | 0,723 | [0,658 – 0,777] | 0,062 |
| **3–5** | 508 | **0,808** | [0,764 – 0,850] | 0,055 |
| 6+ | 590 | 0,846 | [0,813 – 0,875] | 0,042 |

**Satura em 3 a 5 check-ins.** É o mesmo formato do achado do M2 (≈21 respostas formam o corpus de dificuldade de uma questão): pouca coleta, do tipo certo, satura rápido.

Com zero check-ins o intervalo inclui 0,50 — na primeira leitura o modelo não sabe nada, e é honesto dizer isso em vez de exibir um número.

---

## 9. O que isso significa para o produto

1. **M1, como estava desenhado, não existe.** Inferir `fluindo / travado / disperso` de telemetria comportamental não se sustenta no único dado que permite testar. Trocar "seis analisadores de regra" por "um modelo comportamental" trocaria uma ficção por outra, mais cara.

2. **O que funciona é declarado, não inferido.** Dois toques do aluno valem mais que toda a telemetria — e o que ele declara é calibrado pela dificuldade que a turma atribuiu ao mesmo material.

3. **Fecha o argumento da LGPD pelo lado técnico.** A [issue #195](https://github.com/BitStudioLabs/trail-up/issues/195) argumenta que frames de câmera e `touch_samples` são coletados sem consumidor. Este resultado é mais forte: a **categoria de sinal** que eles representam não prediz o estado do aluno. Coletar dado sensível de menor para alimentar inferência que não funciona não tem defesa — nem jurídica, nem técnica.

4. **O check-in sobe de fase.** No plano ele era Fase 3, "quando houver aluno". Ele é o mecanismo: é rótulo e feature ao mesmo tempo, e satura em 3–5 respostas.

5. **Converge com o M2.** Lá o ganho veio da dificuldade da questão, não do comportamento. Aqui a dificuldade do texto é o único sinal que sobrevive ao teste intra-aluno, e o comportamento é negativo. Dois datasets independentes, domínios diferentes, mesma conclusão: **o sinal está na dificuldade do conteúdo; não está na telemetria de momento.**

---

## 10. Limites

- **O ARES instrumenta clique, não leitura.** Sem scroll, sem *dwell* por material, sem ociosidade com limiar de 15 s — coisas que o TrailUp tem (`MetricasContext.tsx:110`). O `ativo_ratio` mediano de 0,115 mostra o quanto isso pesa. Então o resultado é sobre **sinais do tipo ARES**, e não prova que a telemetria mais rica do TrailUp não possa fazer melhor. O que ele faz é **inverter o ônus da prova**: quem quiser manter os analisadores precisa mostrar o contrário, medido.

- **200 alunos universitários, leitura em L2, 6 semanas.** Transfere desenho e ordem de grandeza, não coeficientes.

- **O instrumento é grosseiro.** 46,4% das notas são exatamente 3. Parte do teto de 0,84 pode ser do instrumento, não do método.

- **O alvo é uma declaração, não um estado.** Mesmo o 0,654 intra-aluno da dificuldade do texto prevê o que o aluno **vai relatar**. Que isso coincida com estar travado é suposição adicional, não medida.

- **Não testado:** sequência dentro da leitura (padrão temporal em vez de agregado). O estudo previa HMM sobre a sequência de verbos; com o agregado tão perto do acaso o retorno esperado é baixo, mas não está descartado.

- **Sem grupo de controle.** Observacional. Nada aqui autoriza inferência causal.
