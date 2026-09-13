# Empenho do aluno — medível, estável, e não prediz aprendizado

**Base:** EdNet KT3 · **Veredito:** testado e **não entra no módulo** · **Scripts:** [`scripts/`](scripts/)

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

## 5. Veredito

**O empenho é medível e estável, e não prediz aprendizado.** Vai para o mesmo lugar do engajamento: mede uma propriedade real do aluno que não se traduz em ganho.

Não entra no módulo, por duas razões:

1. **Não passa no teste que justificava construí-lo.** A aposta era preencher o buraco que o engajamento deixa; não preenche.
2. **Exige instrumentação que o TrailUp não tem.** Depende de tempo por questão — a mesma lacuna que trava `chute` e `tempo`. Custaria coleta nova para entregar um número sem consequência medida.

**O que sobrevive disto:** a correlação de +0,456 entre resíduo de tempo e ler a explicação após errar. Se houver um módulo de esforço um dia, é provável que ele se apoie no comportamento **após o erro**, e não no tempo por resposta.

---

## 6. Limites

- **Uma operacionalização, não a única.** "Esforço = tempo além do previsível" é defensável e é a que se mede sem rótulo. Tempo alto pode ser esforço ou distração — o EdNet não distingue as duas coisas, e `n_quits` só ajuda parcialmente.
- **Uma base só.** O engajamento só mostrou seus problemas na segunda base. Este resultado não foi replicado.
- **O alvo de aprendizado é limitado.** "Acertar ao reencontrar" mede retenção de item, não compreensão. O ganho entre metades é grosseiro e dominado por regressão à média.
- **EdNet é TOEIC, autoestudo adulto.** Num contexto com prazo e nota, o tempo investido pode significar outra coisa.
- **Observacional.** Nada aqui diz que fazer o aluno investir mais tempo o faria aprender mais.
