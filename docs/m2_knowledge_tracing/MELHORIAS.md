# Rodada de melhoria — o que funcionou, o que não, e dois erros corrigidos

> **Este documento é o registro do caminho, não a referência.** Os números que valem estão no [RELATORIO_M2.md](RELATORIO_M2.md), refeito em 2026-09-13 com reexecução completa sobre o rótulo corrigido, intervalos de confiança e tabelas de calibração. Mantenho este aqui porque as duas correções que ele descobriu — o alvo errado do gate (§2) e o rótulo contaminado (§6) — são o conteúdo mais útil da rodada, e a ordem em que apareceram importa para não repetir o erro.
>
> As seções 1 a 4 foram medidas **antes** da descoberta da contaminação e não devem ser citadas como resultado.

2026-09-12, sobre os modelos de [M2](RELATORIO_M2.md) e [M1](../m1_estado_de_estudo/RELATORIO_M1.md). Mesmo protocolo: EdNet/ARES, split por usuário, dificuldade estimada só no treino.

---

## 1. M2, previsão da próxima resposta: ganho pequeno

Hipótese: o EdNet traz **189 tags** de habilidade por questão e a v1 usava só `part` (7 valores). Decompor por skill é o núcleo do knowledge tracing — devia ser o grande salto.

| Conjunto | feats | AUC | log-loss | ECE | ganho |
|---|---|---|---|---|---|
| v1 (referência) | 15 | 0,7547 | 0,5685 | 0,0025 | — |
| **v1 + tags (189 skills)** | 21 | 0,7560 | 0,5675 | 0,0036 | **+0,0012** |
| v2 sem tags (janela recente + tempo relativo + sessão) | 22 | 0,7613 | 0,5628 | 0,0044 | +0,0066 |
| v2 completo | 28 | 0,7617 | 0,5626 | 0,0047 | +0,0069 |

**A hipótese estava errada.** As tags deram +0,0012 — ruído. O ganho veio de features banais: acerto nas últimas 5 respostas, tempo de resposta relativo à mediana **do próprio aluno**, e posição dentro da sessão.

Provável razão: `acc_prev_part` e `ewma_part` já capturam quase toda a estrutura de habilidade que as tags trariam, porque as tags são fortemente aninhadas dentro de `part`.

+0,007 no total é pouco, e há uma explicação de teto: **0,76 é o patamar do DKT na literatura para EdNet**; o SAINT+ chega a 0,79 com transformer sobre a sequência inteira. O limite aqui não é de ajuste, é de arquitetura — e um modelo de sequência não se justifica numa API que hiberna, para ganhar 0,03.

## 2. O erro de desenho: eu estava otimizando o alvo errado

O gate não pergunta *"o aluno acerta a próxima?"*. Pergunta *"este aluno vai travar neste tópico?"*. Eu vinha usando `1 − p(acerta a próxima)` como sinal do gate — e uma questão difícil derruba esse `p` sem que o aluno esteja travado.

Treinando **direto** no alvo do gate: a taxa de acerto do aluno nas próximas 10 respostas do mesmo tópico ficar abaixo de 50% (7,8 M amostras, 26,7% positivas).

| Critério | AUC | AP | Brier | ECE |
|---|---|---|---|---|
| **alvo direto** | **0,8061** | **0,6814** | 0,1451 | 0,0090 |
| acerto acumulado no tópico | 0,7721 | 0,5727 | — | — |
| regra do TrailUp | 0,7612 | 0,6038 | — | — |
| **proxy que eu propunha** | **0,6982** | 0,4614 | 0,2126 | — |

**O proxy era pior que a regra que ele ia substituir** (0,698 contra 0,761). Se tivesse ido para produção como estava desenhado, teria piorado o gate.

Precisão por taxa de disparo:

| Dispara em | alvo direto | cobertura | proxy | regra |
|---|---|---|---|---|
| 5% | **94,0%** | 17% | 58,4% | 82,6% |
| **10%** | **87,0%** | 31% | 52,0% | 79,3% |
| 15% | 76,9% | 41% | 51,6% | 73,3% |
| 20% | 69,2% | 49% | 50,5% | 65,3% |
| 30% | 58,1% | 62% | 47,1% | 55,0% |

Ponto de operação sugerido: **10% de disparo, 87% de precisão, 31% de cobertura**. Um em cada dez ciclos abre a LLM, e quando abre está certo em 87% das vezes.

Modelo salvo: `m2_gate.pkl` (2,8 MB).

## 3. Consequência para a arquitetura

São **dois modelos, não um**, e cada um responde a sua pergunta:

| Modelo | Pergunta | Uso | AUC |
|---|---|---|---|
| `m2_model_v2` | o aluno acerta esta questão? | `dominio_estimado`, `confianca`, `tendencia` — o que o professor vê | 0,762 |
| `m2_gate` | o aluno vai travar neste tópico? | decide o nível do gate | 0,806 |

Usar um para o papel do outro é o erro que esta rodada encontrou.

## 4. M1: o ganho é calibração, não ordenação

| Modelo | AUC | AP | Brier | ECE |
|---|---|---|---|---|
| v1: proporção binária, prioris a olho, média simples | 0,8012 | 0,6827 | 0,1760 | 0,1135 |
| v2a: prioris ajustadas (25/3) | 0,8036 | 0,6909 | 0,1770 | 0,1202 |
| v2b: nota média 1–5 em vez de binária | 0,7792 | 0,6723 | 0,2348 | 0,2449 |
| **v2c: logística sobre as duas bases + interação** | 0,7971 | 0,6767 | **0,1574** | **0,0322** |

Ordenação não melhora (+0,002 no melhor caso) — o teto ali é do instrumento: 46% das notas são exatamente 3. Mas a **calibração melhora 3,5×** (ECE 0,113 → 0,032), e é ela que o plano identifica como o valor do modelo.

E melhora onde o modelo era cego, no aluno novo:

| Check-ins do próprio aluno | v1 | **v2c** |
|---|---|---|
| 0 | 0,500 | **0,573** |
| 1–2 | 0,642 | **0,741** |
| 3–5 | 0,813 | 0,822 |
| 6+ | 0,841 | 0,840 |

Com zero check-ins o v1 não sabia nada; o v2c já usa a dificuldade do texto medida na turma. Modelo salvo: `m1_model.pkl` (2 KB).

## 5. O que não foi tentado

- **Modelo de sequência** (DKT/SAKT/SAINT) no M2 — ganho esperado de 0,03 pela literatura, contra custo de latência e dependências numa API que hiberna.
- **Tempo real por questão** (`enter` → `respond` no KT3 bruto) — a extração guardou só as respostas. É a feature mais promissora que ficou de fora.
- **Ordinal no M1** — prever a nota 1–5 em vez do binário; a tentativa com média contínua piorou, mas um modelo ordinal próprio não foi testado.

---

## 6. O rótulo estava contaminado — 21,7% das linhas não eram a resposta final

Ao reextrair o stream completo do KT3 para pegar a latência por questão, apareceu isto: no EdNet o aluno pode responder a mesma questão **várias vezes antes do `submit`**, e a que vale é a última. A extração original guardou todas as linhas `respond` como respostas independentes.

- **21,7% das respostas** eram tentativas intermediárias contadas como resposta;
- e não é ruído aleatório: o aluno troca justamente quando errou ou hesitou;
- resultado: **acurácia global medida em 0,577, quando a real é 0,668** — 9 pontos de viés.

Tudo que foi treinado antes da §7 usou esse rótulo. As conclusões qualitativas se sustentam (a regra é ruim, o alvo direto é melhor que o proxy), mas as magnitudes não valem.

Feito o conserto, o dataset passa a ter **6.504.124 respostas finais** de 27.792 usuários, mais três sinais que não existiam:

| Sinal | Valor |
|---|---|
| latência até a resposta final | mediana 21,7 s, p90 86,3 s |
| **trocou de alternativa** | 15,9% das respostas — hesitação medida |
| tempo lendo a explicação anterior | presente em 93,2% |

### Um vazamento, e a blindagem

A primeira rodada v3 passou os argumentos errados por um problema de aspas no laço, e as features **posteriores à resposta** (latência e trocas da própria questão) entraram no modelo que prevê aquela resposta. AUC 0,764, contaminado.

O script agora tem lista explícita de features posteriores e `assert` nos argumentos: o vazamento não volta por descuido de quoting.

## 7. Números finais, sobre o rótulo corrigido

| Alvo | conjunto | feats | AUC | AP | log-loss | ECE |
|---|---|---|---|---|---|---|
| acerta a próxima | base | 19 | 0,7507 | 0,8541 | 0,5419 | 0,0029 |
| acerta a próxima | **+ sinais novos** | 25 | **0,7538** | 0,8569 | 0,5396 | 0,0022 |
| vai travar no tópico | base | 19 | 0,7784 | 0,3743 | 0,3115 | 0,0032 |
| vai travar no tópico | **+ sinais novos** | 30 | **0,7795** | 0,3783 | 0,3109 | 0,0039 |

Latência e hesitação deram **+0,003** e **+0,001**. Terceira aposta de feature que não paga.

**Modelo do gate contra as alternativas, no mesmo dado corrigido** (taxa-base 12,3%, contra 26,7% no dado contaminado — só o *lift* compara entre rodadas):

| Critério | AUC | AP | lift@10% |
|---|---|---|---|
| **modelo do gate (v3)** | **0,7795** | **0,3783** | **3,40×** |
| acerto acumulado no tópico | 0,7351 | 0,2791 | 2,94× |
| regra do TrailUp | 0,7032 | 0,2992 | 2,90× |
| EWMA no tópico | 0,6754 | 0,2363 | 2,27× |

| Dispara em | modelo | lift | cobertura | regra | lift | cobertura |
|---|---|---|---|---|---|---|
| 5% | 51,5% | 4,2× | 21% | 46,6% | 3,8× | 19% |
| **10%** | **41,7%** | **3,4×** | 34% | 35,5% | 2,9× | 31% |
| 15% | 35,8% | 2,9× | 44% | 31,3% | 2,6× | 39% |
| 20% | 31,9% | 2,6× | 52% | 27,9% | 2,3× | 47% |

A vantagem sobre a regra é real — +0,076 de AUC, +0,079 de AP, 17% mais precisão no ponto de operação — e **menor do que o dado contaminado sugeria**. Modelos: `m2_gate_v3.pkl`, `m2_model_v3.pkl`.

## 8. O saldo desta rodada

O que **não** deu resultado, em ordem de decepção:

| Aposta | Ganho |
|---|---|
| 189 tags de habilidade | +0,001 |
| latência e hesitação por questão | +0,003 |
| janela recente, sessão, tempo relativo | +0,007 |

O que deu:

1. **Treinar no alvo que o gate decide** — e descobrir que o proxy que eu propunha era pior que a regra que ele ia substituir.
2. **Achar o rótulo contaminado** — 21,7% das linhas e 9 pontos de viés na acurácia. Nenhum ajuste de modelo teria compensado isso.

Mais features renderam pouco; olhar para o dado rendeu as duas correções que importavam.
