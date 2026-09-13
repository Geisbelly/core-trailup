# Oito investigações — o que sobreviveu

**Bases:** EdNet KT3, OULAD, classEx, AI4EDU · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê)

Oito hipóteses testadas com o mesmo protocolo: **split que respeita a unidade de generalização**, **baseline explícito** e **um teste que poderia derrubar o resultado**. Este documento é o índice; cada item que virou módulo tem relatório próprio.

| # | investigação | resultado | serve? |
|---|---|---|---|
| 1 | Tempo esperado de resposta | R² 0,561 só com a questão | **sim** |
| 2 | Questão defeituosa | fenômeno raro no corpus; marcação imprecisa | método pronto, corpus errado |
| 3 | Detecção de chute | validado; traço +0,922 | **sim** |
| 4 | Curva de esquecimento | queda de 26 pontos em 6 meses | **sim** |
| 5 | Risco de evasão | AUC 0,748 → **0,783** após o conserto do SQL | **sim**, abaixo da meta |
| 6 | Avaliador de resposta aberta | Spearman 0,532 (teto ≈0,88) | parcial |
| 7 | Roteador de intenção do chat | **38% contra 60% de chutar** | **não** |
| 8 | Dificuldade pelo texto | r = −0,929 com n=8 | inconclusivo |

---

## 1. Tempo esperado de resposta — funciona, e é da questão

**A pergunta:** quanto tempo esta questão deve levar? Serve para estimar duração de atividade e para detectar quem está muito fora do esperado.

**Dado:** EdNet, 6.504.124 respostas, split por aluno (80/20).

**A métrica:** **R²** sobre o log da latência — a fração da variação que o preditor explica. 0 significa "não explica nada além da média"; negativo significa que é pior que usar a média. Uso log porque a latência tem cauda longa (mediana 21,7 s, p90 86,3 s).

| preditor | R² (log) | erro relativo mediano |
|---|---|---|
| sempre a mediana global | −0,037 | 47,4% |
| **mediana DA QUESTÃO** | **0,561** | **27,0%** |
| mediana DO ALUNO | −0,037 | 47,4% |
| questão × aluno | 0,561 | 27,0% |

> **Por que o aluno dá exatamente o mesmo que o global.** Sob split por aluno, o aluno de teste **nunca apareceu no treino** — a mediana dele não existe, e o valor cai para o global por construção. É o cenário real do TrailUp, onde aluno novo chega toda semana. Num recorte em que o aluno é visto no treino, a mediana dele dá R² 0,007 — igualmente nada.

**É o quinto experimento consecutivo a apontar para a questão.** É também por isso que [`tempo.py`](../../trailup_core/tempo.py) usa a mediana da questão **sem ajuste por aluno** — ajustar piora.

**O uso secundário:** detectar quem está muito acima do esperado *naquela questão*.

| | fração das respostas | acerto | IC 95% |
|---|---|---|---|
| 3× mais lento que a mediana da questão | **2,7%** | **52,7%** | [52,5% – 53,0%] |
| o resto | 97,3% | 67,2% | [67,2% – 67,3%] |

Quem demora o triplo acerta **14,5 pontos menos**. É comparação com a distribuição da própria questão — não inferência de estado emocional.

---

## 2. Questão defeituosa — o método está certo, o corpus quase não tem o fenômeno

Detalhe completo em [TRACOS_E_GRUPOS §3](../tracos/TRACOS_E_GRUPOS.md).

**Resumo:** **2,96%** das questões têm discriminação negativa numa metade dos alunos, e **0,53% confirmam nas duas metades**.

O fenômeno existe, é raro, e marcar com precisão é difícil: das 315 marcadas numa metade, **17,8% confirmam** na outra (base 2,85%, lift 6,2×).

**Vale mais no TrailUp** — lá as questões são escritas por professor e geradas por IA, então a prevalência deve ser bem maior, e a precisão de marcação sobe com a prevalência.

---

## 3. Chute — validado, e é traço

Detalhe em [TRACOS_E_GRUPOS §4](../tracos/TRACOS_E_GRUPOS.md).

**Definição:** resposta rápida demais **para aquela questão** (<30% da latência mediana dela) **e** errada. 1,01% das respostas.

**A validação:** no reencontro da mesma questão pelo mesmo aluno (594.398 casos),

| na primeira vez | acerta ao reencontrar | IC 95% |
|---|---|---|
| errou **chutando** | **65,1%** | [64,0% – 66,1%] |
| errou trabalhando | 72,5% | [72,4% – 72,7%] |
| tinha acertado | 86,1% | [86,0% – 86,3%] |

Confiabilidade do traço: **+0,922**.

---

## 4. Curva de esquecimento — o resultado mais limpo da série

**A pergunta:** quando trazer a questão de volta?

**Dado:** 594.555 reencontros da mesma questão pelo mesmo aluno, no EdNet.

**O desenho:** para cada par `(aluno, questão)`, comparar o primeiro encontro com o segundo, em função do **intervalo entre eles**.

| intervalo | **acertou antes** | IC 95% | **errou antes** | IC 95% |
|---|---|---|---|---|
| < 1 hora | 90,1% | [89,8 – 90,5] | **83,6%** | [83,4 – 83,8] |
| 1 h a 1 dia | 90,7% | [90,3 – 91,0] | 79,3% | [79,0 – 79,6] |
| 1–3 dias | 87,3% | [86,9 – 87,8] | 71,5% | [71,1 – 71,9] |
| 3–7 dias | 85,9% | [85,4 – 86,4] | 66,1% | [65,7 – 66,5] |
| 1–3 semanas | 83,5% | [83,1 – 83,9] | 61,4% | [61,0 – 61,8] |
| 3 sem a 2 meses | 81,9% | [81,4 – 82,4] | 58,1% | [57,6 – 58,6] |
| 2–6 meses | 83,1% | [82,6 – 83,6] | **57,5%** | [56,8 – 58,1] |

**Quem acertou perde 7 pontos em seis meses. Quem errou perde 26.**

Os intervalos de confiança não se sobrepõem entre faixas adjacentes na coluna "errou", e a curva é monotônica até 2 meses. **Aprendizado raso evapora; consolidado resiste.**

### O controle de composição

A objeção óbvia: talvez as questões reencontradas depois de meses sejam simplesmente mais difíceis.

| intervalo | dificuldade média das questões |
|---|---|
| < 1 hora | 0,625 |
| 1–3 dias | 0,621 |
| 1–3 semanas | 0,628 |
| 3 sem a 2 meses | 0,644 |
| 2–6 meses | **0,668** |

As questões de intervalo longo são **mais fáceis**, não mais difíceis. A composição joga **contra** o efeito observado — o esquecimento real é pelo menos tão forte quanto a tabela mostra.

**Uso:** regra de revisão espaçada que sai direto do dado ([`revisao.py`](../../trailup_core/revisao.py)). Errou → revisar em dias; acertou → em semanas. É decisão de trilha, não de estado.

---

## 5. Risco de evasão — útil, mas abaixo da meta que eu mesmo pus

> **Os números desta seção são da versão anterior ao conserto de `evasao.sql`.** A auditoria encontrou dois defeitos nos coeficientes publicados — probabilidade 11× superestimada e um sinal invertido — e o refit levou o AUC de 0,748 a **0,783**, com lift de 3,4× para **4,1×** no top 10%. Ver [a auditoria](../auditoria/AUDITORIA.md). O que segue descreve a investigação original, que continua válida como método.

**Dado:** OULAD, painel **aluno × semana**, **523.386 linhas**, 171.047 entregas com data.

**O alvo:** o aluno **evade dentro da janela de previsão** — não "evade em algum momento". **3,31% das linhas** são positivas. É um alvo raro, e isso muda quais métricas fazem sentido.

**Desenho:** **split por coorte** — treina em 2013B / 2013J / 2014B, testa em **2014J**. É mais severo que o split aleatório usado em boa parte da literatura, e é o cenário real: o modelo prevê uma turma que ainda não existia quando foi treinado.

**As métricas, aqui:** com 3,6% de positivos no teste, acurácia não serve para nada (dizer "ninguém evade" acerta 96,4%). **AP** (precisão média) tem como referência de acaso a própria taxa base — 0,036. E **lift** é quantas vezes a precisão do alerta supera essa base.

| modelo | AUC | AP |
|---|---|---|
| regra: sem clique há ≥14 dias | 0,596 | 0,050 |
| só a nota média até agora | 0,586 | 0,047 |
| logística completa (exportável para SQL) | 0,737 | 0,102 |
| **boosting completo** | **0,748** | **0,106** |
| *(acaso)* | *0,500* | *0,036* |

O plano estipulava **AUC ≥ 0,80 na semana 4**; obtive **0,746**. **Não bati a meta.**

### Por semana do curso

| semana | positivos | AUC | AP |
|---|---|---|---|
| 2 | 3,4% | 0,638 | 0,061 |
| **4** | 3,7% | **0,746** | 0,111 |
| 8 | 3,5% | 0,751 | 0,129 |
| 12 | 2,9% | 0,704 | 0,060 |
| 16 | 2,6% | 0,768 | 0,094 |

A semana 2 é claramente pior (0,638) — cedo demais, o aluno ainda não gerou histórico. A partir da semana 4 estabiliza em torno de 0,75.

### A métrica de produto

**Alertando os 10% de maior risco:** precisão **12,3%** contra base de 3,6% — **lift 3,4×**, cobrindo **34%** de quem evade.

Lido de outro jeito: de cada 8 alertas, 1 é alguém que de fato evadiria. Se a intervenção for barata (uma notificação), o lift compensa; se for cara (contato humano), 12% de precisão pode não bastar. **É decisão de produto, e o número está aqui para ela ser tomada com o valor certo.**

### Equidade

| grupo | taxa de evasão | AUC | cobertura no top 10% |
|---|---|---|---|
| `disability = N` | 3,5% | 0,749 | 34% |
| `disability = Y` | 5,1% | 0,738 | **34%** |

**Sem gap de cobertura** — o grupo com maior taxa de evasão é alertado na mesma proporção. A diferença de AUC (0,011) é pequena. Não testei outros atributos protegidos.

### Prever "em algum momento" não é mais fácil

A explicação intuitiva para o resultado abaixo da meta seria que prever **quando** é mais difícil que prever **se**. Não é o caso. Alvo "evade em algum momento do curso" (15,1% de positivos):

| alvo | AUC | AP |
|---|---|---|
| evade **nesta janela** | **0,748** | 0,106 |
| evade **em algum momento** | 0,701 | 0,279 |

**O alvo "em algum momento" ordena pior** (0,701 contra 0,748), e por semana a diferença é maior ainda (semana 4: 0,666 contra 0,746). A AP dele parece melhor só porque a taxa base é 4× maior — **AP não compara entre alvos com prevalências diferentes**, e essa é exatamente a armadilha.

> **Nota para quem rodar o script:** `02_completo.py` imprime no fim uma conclusão em sentido contrário ("prever SE evade é mais fácil"). É texto obsoleto, contradito pela saída do próprio script.

As causas prováveis do 0,75 são outras: **conjunto de features magro** (cliques agregados, sem quebra por tipo de recurso) e o **split por coorte**, mais severo que o da literatura.

Implementado como [`evasao.sql`](../../sql/evasao.sql) — logística em SQL puro, para rodar via `pg_cron`, porque a API hiberna e isto precisa de relógio. A função devolve **faixa de risco**, não score, para o professor.

## 6. Avaliador de resposta aberta — parcial

Detalhe completo em [AVALIADOR_DISCURSIVO](../discursiva/AVALIADOR_DISCURSIVO.md).

**Resumo:** grafo de conceitos + encoder multilíngue chega a **Spearman 0,532** contra um teto de ≈0,88 (concordância entre execuções independentes do próprio LLM). O ganho não veio da estrutura de grafo (+0,006 das arestas) nem da camada semântica (+0,000) — veio da **cobertura ponderada dos conceitos do gabarito** e da **penalização de divagação**.

**Serve para triagem, não para nota.** As 10 respostas de menor previsão têm nota real média **1,87** contra 3,50 do geral.

---

## 7. Roteador de intenção do chat — não funciona, e quase enganou

**Dado:** AI4EDU, 217 sessões, 45 alunos. Rótulo = qual ferramenta o aluno usou.

**Primeiro resultado: 88% de acurácia** (maioria 43%). Parecia o melhor resultado da série.

Não era. **62% das sessões com ferramenta começam com o prefixo de template daquela ferramenta** — `Summarizer` em 100% dos casos, `Conversation across time` em 92%, `Self-assessment quiz` em 84%. O modelo estava lendo **o template que a própria ferramenta insere na mensagem**. Circular.

**Teste honesto** — prever a ferramenta usando só os turnos **seguintes ao primeiro**, que o aluno escreve livre:

| | acurácia |
|---|---|
| chutar a maioria | **60%** |
| modelo sobre o texto livre | **38%** |

**Pior que chutar.** Sem o template não sobra sinal de intenção. O item morre neste corpus.

> Este é o negativo mais valioso da série: **teria ido a produção parecendo excelente.** A lição de método é geral — quando um resultado é bom demais, procurar primeiro o que no dado já contém a resposta.

---

## 8. Dificuldade pelo texto do enunciado — inconclusivo

**A lacuna que trava o cold start:** questão nova, zero respostas. Metadado não prediz (58% contra 55% de chutar — ver [FAIXA_DIFICULDADE §9](../dificuldade/FAIXA_DIFICULDADE.md)). **Texto nunca tinha sido testado** porque o EdNet não traz enunciado. classEx traz.

Correlação das features do texto com a nota média da tarefa:

| feature | Spearman |
|---|---|
| comprimento médio de palavra **no gabarito** | **−0,929** |
| diversidade lexical do enunciado | +0,690 |
| nº de frases no gabarito | −0,663 |
| nº de palavras no enunciado | −0,548 |

**São 8 tarefas e 10 features testadas.** Com *n* = 8, uma correlação de 0,93 aparece por acaso com facilidade — e as features não são independentes entre si, então não há correção de múltiplos testes que salve. O sinal é plausível (gabarito com palavras longas indica conteúdo técnico, que rende nota menor), mas **isto é hipótese, não resultado**.

O teste precisa de dezenas de tarefas. É barato refazer quando houver.

---

## O que isso soma

**Funcionam e viraram módulo:** tempo esperado (1), chute (3), esquecimento (4), evasão (5).

As quatro têm a mesma forma: **medem uma quantidade acumulada e estável**, não um estado momentâneo. É a mesma divisa que separou o [M1](../m1_estado_de_estudo/RELATORIO_M1.md) (falhou, AUC 0,51) dos [traços de aluno](../tracos/TRACOS_E_GRUPOS.md) (0,70–0,98).

**Dois negativos úteis:** o roteador de intenção (7) era circular e teria ido a produção parecendo excelente; a questão defeituosa (2) está pronta mas precisa de um corpus onde o fenômeno seja comum — que é justamente o TrailUp.

**Um parcial:** o avaliador (6) bate a similaridade simples mas fica a 60% do caminho até o LLM. Serve como triagem, não como substituto.

**Um inconclusivo:** dificuldade pelo texto (8) — a única via conhecida para resolver o cold start, e ainda sem poder estatístico para decidir.

---

## Limites que valem para todos os oito

- **Nenhum foi medido no TrailUp.** O banco está vazio; tudo vem de corpora estrangeiros, em domínios distantes (TOEIC adulto, universidade a distância britânica, economia universitária alemã). O que transfere é a **forma** do achado e a **ordem de grandeza da evidência necessária** — não os coeficientes.
- **Tudo observacional.** Nenhuma das associações medidas autoriza inferência causal. "Quem chuta aprende menos" não diz que impedir o chute faria aprender mais.
- **EdNet é CC BY-NC.** Os itens 1, 2, 3 e 4 saem dele.
- **Os intervalos de confiança medem incerteza amostral**, que nos itens de EdNet é minúscula pelo tamanho. A incerteza que importa — de transferência de domínio — não está quantificada em lugar nenhum.

---

## Apêndice — o que mudou, e por quê

**Item 1 (tempo esperado).** O R² reportado era 0,651 para o modelo completo e 0,571 só com a mediana da questão. A reexecução sob split por aluno dá **0,561** para a mediana da questão. E o "R² 0,007 do aluno" vinha de um recorte em que o aluno era visto no treino; sob split por aluno a mediana dele **não existe**, e o preditor cai para o global (R² −0,037). A conclusão é a mesma nos dois casos.

**Item 2 (questão defeituosa).** A afirmação "**zero** questões com discriminação negativa" era bug de contagem — o script comparava contra −1 em vez de 0. Detalhes em [TRACOS — apêndice](../tracos/TRACOS_E_GRUPOS.md).

**Item 5 (evasão).** Os números reproduzem (AUC 0,748, lift 3,4×). O que faltava era a **quebra por semana**: a semana 2 dá 0,638 — o alerta não funciona no início do curso, que é quando alguém gostaria de tê-lo. A versão anterior dava um número único e escondia isso.

**Item 6 (avaliador discursivo).** Teto e pré-filtro corrigidos — ver [AVALIADOR_DISCURSIVO — apêndice](../discursiva/AVALIADOR_DISCURSIVO.md).

### Uma armadilha de métrica, registrada

Ao comparar os dois alvos de evasão, a **AP** do alvo "em algum momento" é 0,279 contra 0,106 do alvo "nesta janela" — parece muito melhor. Mas a taxa base é 15,1% contra 3,6%: **AP não compara entre alvos de prevalência diferente**. O AUC, que é invariante à taxa base, mostra o contrário (0,701 contra 0,748).
