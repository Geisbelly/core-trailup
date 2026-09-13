# Dificuldade da questão por turma — a métrica do professor

**Base:** OULAD (CC BY 4.0) · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê)

---

## 1. Pergunta de negócio

A [dificuldade da questão](FAIXA_DIFICULDADE.md) é medida sobre todos os alunos que a responderam. Mas o professor não ensina "todos os alunos" — ele ensina **uma turma**, e a pergunta dele é outra:

> **Esta questão foi difícil *para a minha turma*, ou ela é difícil para todo mundo?**

A distinção importa porque as ações são diferentes. Questão difícil para todos é problema da questão (ou do conteúdo). Questão difícil **só nesta turma** é sinal de que algo daquela aula não pegou — e é aí que o professor age.

**Esta métrica é para o professor ver, não para o modelo consumir.** A §7 mostra que a forma boa de exibir é ruim de calcular, e vice-versa.

---

## 2. Dados

**Fonte:** OULAD — Open University Learning Analytics Dataset, licença **CC BY 4.0**.

**Por que este dataset:** é o único da pasta com **estrutura de turma**. O EdNet é app de autoestudo — cada aluno sozinho, sem turma, sem professor, sem coorte. Não serve para esta pergunta.

| | |
|---|---|
| notas | 173.739 |
| alunos | 28.785 |
| turmas (módulo × apresentação) | **22** |
| avaliações distintas (slots) | 78 |
| alunos por turma — mediana | **1.204** |
| avaliações por turma — mediana | **9** |

**Unidade de análise:** o par `(avaliação, turma)`.

**Turma** = `código do módulo` + `apresentação` (ex.: `DDD-2013B`). **Slot** = a mesma avaliação comparável entre turmas do mesmo módulo (`módulo | tipo | ordem`), porque o `id_assessment` muda a cada apresentação.

### O alvo, e por que este corte

O OULAD dá **nota de 0 a 100**, não acerto/erro. Para ter um análogo binário de "acertou":

```
y = 1 se a nota for >= 80
```

A escolha do corte não é neutra:

| corte | % positivos |
|---|---|
| ≥40 | 95,6% |
| ≥60 | 83,9% |
| ≥70 | 69,4% |
| **≥80** | **50,5%** |
| ≥90 | 25,1% |

**≥80 é o único que dá classes equilibradas.** Com ≥40, 96% passam e não há nada a discriminar. É escolha minha, feita por essa razão, e deve ser revista se o TrailUp tiver distribuição diferente.

---

## 3. A turma importa? Sim, mas a avaliação importa muito mais

Decomposição da variância da taxa de acerto, sobre os pares (avaliação, turma) com ≥20 alunos:

| fonte de variação | variância |
|---|---|
| **entre avaliações** (a mesma turma em provas diferentes) | **0,0424** |
| **entre turmas** (a mesma prova em turmas diferentes) | **0,0041** |

**A avaliação pesa 10,3× mais que a turma.**

Mas a cauda não é desprezível. Amplitude (turma com maior acerto menos turma com menor acerto, na mesma avaliação):

| | valor |
|---|---|
| mediana | 0,040 |
| p90 | **0,151** |
| máximo | **0,419** |

**11,5% das avaliações variam mais de 15 pontos percentuais entre turmas.** Exemplos reais: `DDD|TMA|6` vai de 60 a 75 pontos conforme a turma; `CCC|CMA|4`, de 60 a 75; `BBB|TMA|3`, de 59 a 71.

**Conclusão de desenho:** nem "calcular por turma do zero" (turma pequena vira ruído) nem "ignorar a turma" (perde a cauda que importa). **Encolhimento hierárquico** — dificuldade global corrigida pela turma, com peso proporcional à evidência daquela turma.

---

## 4. A dificuldade da turma só existe depois que a turma responde

Testar numa **turma nova** não responde nada: por definição não há dificuldade-de-turma para consultar, todas as variantes caem para o global (AUC idêntico) e o percentil vira constante (AUC 0,500).

O teste que responde caminha **dentro da avaliação, ao longo dos alunos**: o aluno na posição *k* é previsto usando só os *k−1* anteriores **da mesma turma**.

| alunos da turma que já responderam | global | turma (cru) | **hierárquico** |
|---|---|---|---|
| 5–9 | 0,688 | **0,657** ⚠️ | **0,694** |
| 10–19 | 0,698 | 0,700 | **0,715** |
| 20–49 | 0,701 | 0,715 | **0,720** |
| 50–99 | 0,695 | 0,723 | **0,724** |
| 100+ | 0,683 | 0,722 | **0,722** |

**Abaixo de 10 alunos, a taxa crua da turma é pior que simplesmente usar o global.** Com pouca evidência, "personalizar" é injetar ruído.

**O hierárquico nunca perde** — é o estimador a usar sempre:

```
dificuldade_turma = (acertos_da_turma + global × 15) / (respostas_da_turma + 15)
```

O prior de 15 foi o melhor no teste e é estável entre 5 e 40.

---

## 5. Quando mostrar ao professor

**Métrica exibida não se avalia por AUC.** AUC responde "o modelo ordena bem?"; a pergunta aqui é outra: *quando eu afirmo que a turma destoa, eu estou certo?*

### A medida

```
z = (taxa da turma − taxa global) / erro-padrão
```

onde o **erro-padrão** é o da proporção, `√(p(1−p)/n)`, com *n* = alunos da turma naquela avaliação.

> **Detalhe que muda o resultado:** a taxa global tem de ser calculada **excluindo a própria turma** (*leave-one-out*). Se a turma entra no seu próprio ponto de referência, o desvio encolhe artificialmente e a regra deixa de disparar — `|z| ≥ 2` cai de 15% para 3% dos casos. **A versão correta é a que exclui.**

### O teste

Replicação: mede o desvio na **metade A** dos alunos da turma e confere a **direção** na **metade B** (alunos independentes).

Primeiro, a evidência de que há o que medir:

| tamanho da metade A | pares | correlação A–B | mesmo sinal |
|---|---|---|---|
| 50–199 alunos | 13 | +0,355 (p=0,234) | 54% |
| **200+ alunos** | 175 | **+0,949** (p<0,001) | 85% |

Com 200 alunos por metade o desvio é altamente reprodutível. Com 50–199 ele não é distinguível de ruído (p=0,23) — o que já antecipa o problema da turma pequena (§6).

### A regra

| regra | exibe em | acerta a direção | desvio médio em B |
|---|---|---|---|
| sempre | 100% | 82% | 0,067 |
| \|z\| ≥ 1,0 | 74% | 90% | 0,082 |
| \|z\| ≥ 1,5 | 59% | 94% | 0,095 |
| **\|z\| ≥ 2,0** | **49%** | **96%** | 0,105 |
| \|z\| ≥ 2,5 | 40% | 99% | 0,123 |

**Exibir sempre erra a direção em quase uma avaliação a cada cinco.** Com `|z| ≥ 2`, aparecem metade das avaliações — as que realmente destoam — com 96% de acerto de direção.

---

## 6. Com turma do tamanho real

As turmas do OULAD têm **mediana de 1.204 alunos**. Turma de escola tem ~30. Isso não é detalhe: o erro-padrão cresce com 1/√n, então tudo o que a §5 mostrou precisa ser refeito no tamanho certo.

Subamostrando cada turma para *N* alunos, 30 repetições por tamanho, referência = o desvio medido com **todos** os alunos daquela turma:

| alunos na turma | `\|z\| ≥ 2` dispara | acerta a direção | correlação com a verdade | *exibir sempre acertaria* |
|---|---|---|---|---|
| 15 | 11% | 93% | 0,65 | *67%* |
| 20 | 12% | 93% | 0,69 | *69%* |
| **30** | **15%** | **94%** | **0,77** | ***72%*** |
| 40 | 16% | 97% | 0,82 | *73%* |
| 60 | 21% | 97% | 0,87 | *77%* |
| 100 | 26% | 99% | 0,92 | *80%* |

*(188 pares (turma, avaliação) com ≥100 alunos servem de referência.)*

**Numa turma de 30, exibir sempre erraria a direção em mais de uma avaliação a cada quatro** (72% de acerto). Com a regra `|z| ≥ 2`, aparecem **15% das avaliações** — as que realmente destoam — com **94% de acerto de direção**.

A correlação com a verdade (0,77 em n=30) mostra o outro lado: o **valor** do desvio numa turma de 30 é ruidoso, mesmo quando a **direção** está certa. É mais um argumento para exibir posição/percentil e não o número cru.

---

## 7. Duas formas, dois usos — e não dá para usar a mesma

Testei as duas representações:

| uso | formato | por quê |
|---|---|---|
| **Professor (exibir)** | **percentil dentro da turma** | É ranking que ele age sobre: *"esta é a 2ª questão mais difícil da sua turma"*. Direto, comparável entre turmas. |
| **Modelo (prever)** | **taxa absoluta com encolhimento** | O percentil **perde o nível absoluto** e custa caro: como feature, AUC 0,744 contra 0,792 da taxa absoluta. |

**O percentil é bom para ler, ruim para calcular.** Usar cada um no seu lugar.

Exemplo real, turma `DDD-2013B`:

| avaliação | acerto | percentil na turma |
|---|---|---|
| `DDD\|Exam\|9` | 10% | **7% — a mais difícil** |
| `DDD\|TMA\|6` | 24% | 14% |
| `DDD\|CMA\|7` | 28% | 21% |
| `DDD\|TMA\|14` | 55% | 93% |
| `DDD\|CMA\|11` | 64% | **100% — a mais fácil** |

> **Granularidade:** a mediana é de **9 avaliações por turma**, então o percentil anda de 11 em 11 pontos. Com poucas questões, mostrar a **posição** (*"2ª de 9"*) é mais honesto que a porcentagem — 7% e 14% sugerem uma precisão que não existe quando há 9 itens.

---

## 8. O desenho recomendado

1. **Sempre calcular** a dificuldade hierárquica (prior 15).
2. **Exibir o desvio da turma só quando `|z| ≥ 2`**, com a taxa global calculada *leave-one-out*.
3. Abaixo disso, mostrar a dificuldade **sem afirmar** que é característica daquela turma.
4. Para o professor, **percentil ou posição**; para o modelo, **taxa absoluta**.

---

## 9. Limites

- **OULAD são avaliações, não questões.** Nota 0–100, não acerto/erro por item. A estrutura (turma × item × aluno) é a mesma; a granularidade não. Uma prova inteira e uma questão não têm a mesma variância nem o mesmo *n*.
- **Turmas universitárias a distância, com mediana de 1.204 alunos.** A simulação de turma pequena subamostra essas turmas — aproxima o **tamanho**, não a **dinâmica** de uma sala de aula presencial.
- **O limiar `z ≥ 2` assume independência entre alunos.** Numa turma real há efeito de professor, de horário, de quem estudou junto. O erro-padrão verdadeiro é **maior** que o binomial, e o limiar deveria ser mais conservador, não menos.
- **`y ≥ 80` é escolha arbitrária**, feita para ter classes equilibradas (§2).
- **22 turmas.** A decomposição de variância da §3 tem pouco grau de liberdade no nível da turma; a razão de 10,3× vem sem intervalo de confiança.
- **Observacional.** "Esta turma foi pior nesta avaliação" não diz por quê — professor, horário, composição, ou acaso residual.

---

## Apêndice — o que mudou, e por quê

**A referência global precisa ser *leave-one-out*.** Uma implementação que inclui a própria turma no cálculo da taxa global encolhe o desvio artificialmente: `|z| ≥ 2` dispara em **3%** dos pares em vez de **15%**, e a funcionalidade parece inútil numa turma de 30. A acurácia de direção é parecida nas duas versões (93–96%); o que muda é quantas vezes a régua diz alguma coisa.

**A razão "a avaliação pesa 4× mais que a turma"** vinha de comparar diferenças medianas. A decomposição de variância da §3 dá **10,3×**. As duas medem coisas ligeiramente diferentes; a de variância é a que está no documento.
