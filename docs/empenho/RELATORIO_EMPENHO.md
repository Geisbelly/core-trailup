# Empenho do aluno — medível, estável, e não prediz aprendizado

**Base:** EdNet KT3 · **Veredito:** testado em **três operacionalizações**, todas nulas · **Scripts:** [`scripts/`](scripts/)

---

## 1. A pergunta, e por que ela é diferente das que já existem

| modelo | pergunta | veredito |
|---|---|---|
| `dominio` | ele **consegue**? | no módulo |
| `engajamento` | ele **vai continuar**? | no módulo |
| **empenho** | ele **está tentando**? | **este documento** |

O engajamento prediz permanência, e está medido que **não prediz aprendizado** (Spearman −0,04 a +0,03 com ganho de acerto). A aposta do empenho era preencher esse buraco: se esforço prediz aprender, ele mede algo que nenhum módulo atual mede.

---

## 2. Como se mede esforço sem rótulo de esforço

Não existe rótulo de "empenho" em dataset nenhum, e inventar um corte repetiria o erro do `k=4` das questões. A saída é medir como **resíduo de regressão**:

```
log(tempo na resposta) ~ log(tempo esperado da questão)
                       + dificuldade da questão
                       + habilidade do aluno

empenho = resíduo
```

O resíduo é o tempo investido **além do que a tarefa exige e a capacidade do aluno explica**. É o mesmo método que a [análise de estrutura de dificuldade](../dificuldade/FAIXA_DIFICULDADE.md) usa para remover habilidade e dificuldade antes de comparar questões.

**6.504.124 respostas, 27.792 alunos.** A regressão explica R² 0,564:

| termo | coeficiente |
|---|---|
| log do tempo esperado da questão | **+0,961** |
| habilidade do aluno | −0,221 |
| dificuldade da questão | −0,025 |

O tempo esperado da questão carrega quase tudo — consistente com os outros cinco experimentos deste estudo.

---

## 3. A medida é boa: estável e não é capacidade disfarçada

**É traço estável.** Correlação entre metades aleatórias e independentes das respostas do mesmo aluno: **+0,920** (16.790 alunos).

**Não é capacidade com outro nome:**

| empenho × | Spearman |
|---|---|
| taxa de acerto | **−0,144** |
| leu a explicação após errar | **+0,456** |
| horas de estudo somadas | +0,186 |
| sessões | +0,114 |
| abandonou a sessão | +0,107 |
| revisou a resposta | −0,126 |
| variação da taxa de acerto | +0,015 |

A correlação mais forte é com **ler a explicação depois de errar** (+0,456), que é o comportamento mais face-válido para "esforço". E a correlação com acerto é fraca e **negativa** — não está medindo aluno bom.

Até aqui, é uma boa medida.

---

## 4. O teste que decide, e ele reprova

### (a) No reencontro da mesma questão

425.773 casos em que o aluno **errou** na primeira vez. Ele se esforçou mais naquela tentativa — acerta mais ao reencontrar?

| quintil de empenho na 1ª vez | acerta ao reencontrar | IC 95% |
|---|---|---|
| q1 (menor) | 70,4% | [70,1 – 70,7] |
| q2 | 71,1% | [70,7 – 71,4] |
| q3 | 73,8% | [73,5 – 74,1] |
| q4 | **73,9%** | [73,6 – 74,2] |
| q5 (maior) | 72,8% | [72,5 – 73,1] |

**AUC 0,516.** E a curva **não é monotônica** — sobe até q4 e cai em q5. Dentro de cada quartil de dificuldade da questão: 0,503 a 0,518.

Controle: a dificuldade da questão sozinha dá AUC 0,584 para o mesmo desfecho. **O conteúdo prediz mais que o esforço.**

### (b) Ganho de acerto ao longo do histórico

16.790 alunos, ganho médio +0,020 entre a primeira e a segunda metade das respostas:

| medida | Spearman com o ganho | **controlando o acerto inicial** |
|---|---|---|
| **empenho** | +0,044 | **−0,025** |
| horas de estudo | −0,105 | −0,008 |
| revisou a resposta | −0,034 | −0,048 |
| leu a explicação | −0,015 | −0,024 |
| variação da taxa de acerto | +0,074 | — |
| taxa de acerto | +0,003 | — |

O +0,044 bruto **vira −0,025 ao controlar pelo acerto inicial**. Era regressão à média: quem começa pior tem mais espaço para subir, e também investe mais tempo por resposta.

---

## 5. Segunda tentativa: o comportamento **após** o erro

O resíduo de tempo correlaciona +0,456 com ler a explicação depois de errar. A hipótese seguinte: talvez o esforço que importa não seja o tempo na questão, e sim o que o aluno faz **depois de errá-la**.

**Desenho.** `expl_sec[i]` é a explicação lida *antes* da resposta *i*. Então o que o aluno fez após errar a questão *i* está na linha *i+1*. Casos: errou na 1ª vez, tem resposta seguinte, e reencontrou a questão — **425.838**.

**E um placebo.** A explicação lida **antes** do erro. Se o efeito for específico do comportamento pós-erro, o placebo tem de predizer menos.

| sinal | AUC |
|---|---|
| explicação lida **após** o erro | 0,524 |
| explicação lida **antes** (PLACEBO) | **0,527** |
| abandonou a sessão em seguida | 0,535 |
| assistiu aula em seguida | 0,515 |
| tempo até a próxima resposta | 0,513 |
| continuou no mesmo tópico | 0,511 |
| *(controle)* facilidade da questão | *0,585* |
| *(controle)* habilidade do aluno | *0,574* |

**O placebo prediz mais que o sinal.** Não há efeito específico do pós-erro: quem lê explicação lê antes e depois, e o que aparece é o traço do aluno, não a reação ao erro.

A tabela por faixa mostra onde está a única estrutura real:

| tempo na explicação após o erro | acerta ao reencontrar |
|---|---|
| **não leu** | **59,2%** |
| <5 s | 75,3% |
| 5–15 s | 74,0% |
| 15–30 s | 73,8% |
| 30–60 s | 73,4% |
| 60 s+ | **72,8%** |

O salto é **ler ou não ler** (59,2% → 75,3%). Dentro de "leu", mais tempo é **ligeiramente pior** — não há dose-resposta. E "não leu" concentra sessões abandonadas, que correlacionam com tudo de ruim.

Dentro de cada célula de dificuldade × habilidade, o AUC fica entre 0,507 e 0,549.

---

## 6. Terceira tentativa: **trajetória** do próprio aluno

A crítica de desenho às duas primeiras: elas comparam **alunos entre si**. Foi exatamente isso que derrubou a "base pessoal" do [M1](../m1_estado_de_estudo/RELATORIO_M1.md) — ela separava quem reclama de quem não reclama, sem dizer nada sobre um aluno específico.

A versão certa compara o aluno **com ele mesmo**: o empenho dele **caiu, está constante, ou subiu?** Cada aluno é seu próprio controle.

**Desenho.** Blocos de 25 respostas consecutivas. Empenho do bloco = resíduo médio. Trajetória = inclinação dos 3 blocos anteriores, normalizada pelo desvio **do próprio aluno**. Desfechos no futuro do que gerou a trajetória. **190.405 blocos, 9.689 alunos.**

### (a) O acerto cai no bloco seguinte?

| trajetória | acerto cai em | IC 95% |
|---|---|---|
| empenho **caiu** | 44,7% | [44,3 – 45,1] |
| empenho constante | 44,3% | [43,9 – 44,7] |
| empenho **subiu** | 43,9% | [43,5 – 44,3] |

Base 44,3%. **AUC 0,504.**

### (b) O aluno para de responder?

| trajetória | para em | IC 95% |
|---|---|---|
| empenho **caiu** | 5,5% | [5,3 – 5,7] |
| empenho constante | **4,4%** | [4,3 – 4,6] |
| empenho **subiu** | 5,4% | [5,2 – 5,5] |

Base 5,1%. **AUC 0,503.**

A única estrutura é um **U**: mudar em qualquer direção precede parar um pouco mais que ficar constante. O efeito é de 1 ponto percentual — real na amostra, pequeno demais para acionar qualquer coisa.

**AUC intra-aluno: 0,496.** Dentro do aluno, nada.

> **Um artefato a registrar:** o acerto do próprio bloco dá AUC 0,264 para "o acerto cai". É regressão à média em estado puro — bloco bom é seguido de bloco pior por aritmética. Qualquer modelo que use acerto recente para prever queda de acerto vai parecer excelente por esse motivo.

---

## 7. O nulo é real, não é medida ruim

Antes de concluir, a pergunta que separa "não há efeito" de "minha medida é ruído": a trajetória **se replica**?

Dividindo as respostas de cada bloco em duas metades independentes e calculando a trajetória em cada uma:

| medida | concordância entre metades |
|---|---|
| nível de empenho no bloco | **+0,794** |
| **trajetória (inclinação)** | **+0,544** |
| trajetória, corrigida para o bloco inteiro (Spearman-Brown) | **0,705** |

Com confiabilidade de 0,705, o **teto de correlação com um desfecho perfeito seria 0,839**. A medida tinha folga de sobra para detectar um efeito.

**Então o nulo é do fenômeno, não do instrumento.** A trajetória de empenho mede algo estável e reproduzível — que simplesmente não antecipa queda de desempenho nem abandono.

---

## 8. Veredito

**Três operacionalizações independentes, três nulos.**

| tentativa | melhor AUC | por que caiu |
|---|---|---|
| nível de esforço (resíduo de tempo) | 0,516 | dificuldade da questão prediz mais (0,584) |
| comportamento após o erro | 0,524 | **o placebo prediz mais** (0,527) |
| trajetória do próprio aluno | 0,504 | nulo com medida confiável (teto 0,839) |

**O empenho é medível e estável, e não prediz nem aprendizado nem abandono.** Vai para o mesmo lugar do engajamento: mede uma propriedade real do aluno que não se traduz em desfecho.

Não entra no módulo, por duas razões:

1. **Não passa no teste que justificava construí-lo.** A aposta era preencher o buraco que o engajamento deixa; não preenche.
2. **Exige instrumentação que o TrailUp não tem.** Depende de tempo por questão — a mesma lacuna que trava `chute` e `tempo`. Custaria coleta nova para entregar um número sem consequência medida.

**O que sobrevive disto:** uma única diferença grande, e ela não é sobre esforço — é sobre **ler ou não ler a explicação** (59,2% contra 75,3% de acerto no reencontro). Não é dose-resposta, e provavelmente é confundida com abandono de sessão. Vale medir como comportamento binário, não como grau de empenho.

**E uma ressalva sobre o uso descritivo.** "Empenho caiu / constante / subiu" continua sendo um rótulo que dá para exibir — a medida é confiável (0,705) e o aluno é seu próprio controle. O que os dados não autorizam é **disparar intervenção** a partir dele: ele não antecipa queda de desempenho (0,504) nem abandono (0,503). Exibir como descrição, nunca como alerta.

---

## 9. Limites

- **Três operacionalizações, não todas as possíveis.** Tempo residual, comportamento pós-erro e trajetória. Tempo alto pode ser esforço ou distração — o EdNet não distingue, e `n_quits` só ajuda parcialmente.
- **A janela é de 25 respostas.** Uma trajetória em escala de dias ou de sessões pode se comportar diferente; não foi testada.
- **Uma base só.** O engajamento só mostrou seus problemas na segunda base. Este resultado não foi replicado.
- **O alvo de aprendizado é limitado.** "Acertar ao reencontrar" mede retenção de item, não compreensão. O ganho entre metades é grosseiro e dominado por regressão à média.
- **EdNet é TOEIC, autoestudo adulto.** Num contexto com prazo e nota, o tempo investido pode significar outra coisa.
- **Observacional.** Nada aqui diz que fazer o aluno investir mais tempo o faria aprender mais.
