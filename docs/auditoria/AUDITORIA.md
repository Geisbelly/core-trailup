# Auditoria — cada número reconferido chamando o código

**Data:** 2026-09-13 · **Scripts:** [`scripts/`](scripts/)

Todo número afirmado nos cabeçalhos dos módulos foi remedido **chamando as funções do módulo**, não reimplementações. É o que pega divergência entre o que o código faz e o que a documentação diz.

Base: EdNet KT3, 6.504.124 respostas, acerto global 0,6677, split por aluno 70/30.

---

## Resultado das 7 partes

**87 verificações. 47 defeitos encontrados**, todos da mesma família: uma saída afirmando uma escala que ninguém mediu. Nenhum apareceria olhando AUC.

| onde | o que afirmava | o que era |
|---|---|---|
| `ritmo._CONF` | confiança 1,00 | teto real 0,997 |
| `ritmo.__str__` | exibia "100%" | arredondamento de 0,997 |
| `engajamento.risco` | dispara na taxa pedida | estoura para 27,5% |
| `evasao.sql` | probabilidade de evasão | **11× maior que a real** |
| `evasao.sql` | sumir aumenta o risco | coeficiente invertido |
| `dominio.p` | probabilidade | ECE 0,056, comprimido |
| `dominio.confianca` | confiança cresce com n | erro real é plano |
| `precisa_reforco` | limiar 0,45 | mudou de 2,3% para 12,2% |
| `duracao_prevista` | soma de medianas | subestima 13% |
| `prioridade_revisao` | onde revisar rende mais | só mede esquecimento |
| `tempo.MIN_RESPOSTAS` | 5 respostas é estável | erro típico de 20% |
| `engajamento.MIN_EVENTOS` | 30 eventos e os eixos funcionam | AUC 0,644, não 0,810 |
| `dias_ate_revisar` | revise em N dias e retém X% | entrega X−4 pontos |
| `pre_avaliacao.divagacao` | % de divagação | mediana é 0,72 — 72% é o normal |
| `dominio` ponto fixo | é a taxa base | é 0,6412, e a direção estava invertida |
| `prever_turma` | variância do posterior | vinha do intervalo clipado |
| `gate.risco` | parâmetro "dificuldade" | recebia facilidade — invertia o termo |
| `dominio.incerteza` | usa a média global dada | fixava 0,67 e ignorava |
| `dominio.tendencia` | limiar 0,03 | recalibração fez disparar com metade |
| `revisao.retencao` | usa `acertou_antes` | ignorava em silêncio |
| `engajamento.trajetoria` | limiar fixo | muda com o nº de janelas |
| `derivar_prior` | prior do corpus | força 210.000 sem avisar |
| `tempo.demorando` | limiar de 3× | inverte na margem de erro da mediana |
| `discriminacao.confirmar` | marca suspeita | dizia "ok" para o não-mensurável |
| 20 entradas hostis | erro claro ou valor são | NaN e infinito vazando |
| `trailup_faixa_risco` | ~top 10% em `alto` | pegava 0,27% — faixa morta |
| `revisao._ACERTOU` | curva de esquecimento | sobe em dois pontos |
| `dominio.incerteza` | "encolhe com n" | só para taxa fixa |

E oito que **conferiram**: `cobertura` (+0,462) e `conceitos_faltando` (−0,410) do `pre_avaliacao`, `gate.MIN_RESPOSTAS = 5` (é o cotovelo exato: 0,534 abaixo dele, 0,660 nele), a aproximação normal do `dificuldade` (cobertura 75,3% contra 74,8% do Beta), `perfil_chute` (p90 e p99), a tabela `REFERENCIA` do engajamento (diferença 0,000), o limite de 3× do `demorando`, e `ritmo.FRONTEIRA_SEG` — cujo 40 s é praticamente o corte ótimo por Otsu (39,8 s), com d de Cohen **maior** que o documentado (4,18 contra 3,30).

---

## Parte 1: os números dos cabeçalhos

| verificação | afirmado | medido | |
|---|---|---|---|
| `dificuldade`: cobertura do posterior | 0,890 | **0,893** | ✅ |
| `dificuldade`: `prever_turma(30)` | 0,888 | **0,880** | ✅ |
| `dificuldade`: força do prior derivado | 7,0 | **7,09** | ✅ |
| `tempo`: R² da mediana da questão | 0,574 | **0,574** | ✅ |
| `tempo`: R² da média do log | 0,578 | **0,578** | ✅ |
| `ritmo`: acerto com 5 respostas | 0,980 | **0,973** | ⚠️ corrigido, e achou bug |
| `dominio`: 2 termos | 0,722 | **0,719** | ✅ |
| `dominio`: 3 termos | 0,727 | **0,725** | ✅ |
| `revisao`: binário | 0,665 | **0,666** | ✅ |
| `revisao`: interpolado pela proporção | 0,669 | **0,670** | ✅ |
| `chute`: diferença no reencontro | −0,076 | **−0,061** | ⚠️ corrigido |
| `gate`: AUC | 0,732 | **0,738** | ✅ |
| `gate`: precisão no ponto de 10% | 0,339 | **0,337** | ✅ |

---

## Os três que derivaram

**`ritmo`: 98% → 97,3%.** O relatório original afirmava 98% de acerto com 5 respostas. A primeira remedição, comparando as **5 primeiras** respostas contra a classificação com 50+, deu 96,4%. A remedição definitiva — **5 sorteadas** ao acaso, três reamostragens, contra a classificação com 200+ respostas — dá **97,3%**, e é esse o valor no módulo. A conclusão de desenho não muda: 5 respostas continuam sendo suficientes.

**`chute`: −7,6 pts → −6,1 pts.** A tabela do módulo diz que marcar com limiar 0,30 separa 7,6 pontos no acerto ao reencontrar. Chamando `chute.foi_chute()` sobre 128.431 casos, a diferença é **6,1 pontos**. A **forma** do achado se sustenta (encolhe monotonicamente conforme o limiar afrouxa); a magnitude varia de 6 a 8 pontos conforme o recorte. Anotado no módulo.

**`gate`: 0,732 → 0,738.** Melhor que o afirmado. Diferença de amostragem; o ponto de operação de 10% confere (33,7% contra 33,9%).

---

## O bug que a auditoria encontrou: `ritmo` afirmava certeza

Ao corrigir os 98% de acerto, apareceu que a **tabela de confiança inteira** vinha da mesma medida refutada:

```python
_CONF = [(5, .98), (10, .99), (21, 1.0)]   # antes
```

**Confiança 1,00 em n=21** — certeza absoluta, que nunca é verdade. O módulo dizia ao consumidor que tinha certeza da classificação com 21 respostas.

Remedido sobre 8.673 questões com 200+ respostas, três reamostragens independentes:

| n | acerto | IC 95% |
|---|---|---|
| 3 | 0,953 | [0,950 – 0,956] |
| **5** | **0,973** | [0,971 – 0,975] |
| 8 | 0,986 | [0,985 – 0,988] |
| 10 | 0,989 | [0,987 – 0,990] |
| 21 | **0,994** | [0,993 – 0,995] |
| 50 | **0,997** | [0,996 – 0,997] |

O teto medido é **0,997** — nem com 50 respostas a classificação é certa.

**E a falsa certeza voltava pela formatação.** Com `{:.0%}`, 0,997 é exibido como **"100%"**. Trocado para uma casa decimal, com teste que verifica que a string nunca contém "100%".

Três testes novos fixam isso: a confiança nunca chega a 1,0 em nenhum `n`, cresce monotonicamente, e a exibição nunca mostra 100%.

---

## Parte 2: os módulos medidos fora do EdNet

| verificação | afirmado | medido | |
|---|---|---|---|
| `pre_avaliacao`: Spearman **ponta a ponta** | 0,463 | **0,483** | ✅ melhor |
| `engajamento`: `ordenar()` no EdNet | 0,856 | **0,843** | ✅ |
| `engajamento`: `ordenar()` no OULAD | 0,862 | **0,869** | ✅ |
| `engajamento`: tabela `REFERENCIA` vs observado | — | **diferença 0,000** | ✅ |
| `dominio`: calibração do `p` | — | **ECE 0,056** | ❌ corrigido |
| `dominio`: `confianca` mede algo? | — | **não** | ❌ corrigido |

**`pre_avaliacao` ficou melhor chamando o módulo.** Os 0,463 vinham de features extraídas pelo pipeline sklearn que gerou os pesos. Chamando `preparar()`/`avaliar()` sobre os textos crus: **0,483**. As duas implementações do grafo diferem — o pipeline usa as 60 palavras mais frequentes como stopword, o módulo deriva por frequência de documento (no classEx, apenas 5). O número que descreve o que o código faz é 0,483.

**E apareceu um limite de escala:** as previsões ficam entre **2,29 e 5,00**. O módulo não consegue dizer "muito ruim" — o piso efetivo é 2,3 numa escala de 1 a 5. Para triagem não atrapalha (as 10 piores têm nota real 2,20 contra 3,50 do geral), mas o valor absoluto não serve como nota. Anotado no cabeçalho.

### O defeito: `risco()` estoura a taxa pedida

| taxa pedida | EdNet dispara em | OULAD dispara em |
|---|---|---|
| 10% | 11,5% | 10,0% |
| **20%** | **27,5%** | 20,8% |

`dias_recentes` é contagem discreta e **67% dos alunos do EdNet têm zero** — muitos empatam no limiar e o alerta estoura em 7,5 pontos.

O módulo agora devolve `Calibracao.taxa_efetiva[taxa]`, medida na própria coorte, para dimensionar a operação com o número certo. Dois testes fixam isso: empate só pode fazer disparar **mais**, nunca menos, e a taxa efetiva bate com o que `risco()` de fato dispara.

---

## Parte 3: `evasao.sql` — dois defeitos, um deles grave

O SQL foi auditado reimplementando a aritmética dele em Python e refazendo o ajuste sobre o OULAD.

### 1. A função devolvia probabilidade 11× maior que a real

| | valor |
|---|---|
| média devolvida pelo SQL publicado | **0,302** |
| taxa real de evasão | 0,027 |
| **fator de superestimação** | **11,1×** |
| ECE | 0,275 |
| AUC (ordenação) | 0,764 |

A causa: os coeficientes vinham de um ajuste com `class_weight='balanced'`, que otimiza ordenação e **destrói a calibração**. Quem lesse o número cru concluiria que um terço da turma está saindo.

**O defeito era invisível para quem usava só a faixa** — a ordenação continuava razoável. Só aparecia para quem lesse o score. É exatamente o tipo de erro que a regra "persistir a ação, não o diagnóstico" esconde em vez de evitar.

### 2. `p_semanas_sem_acesso` com o sinal invertido

Coeficiente **−0,081**: mais semanas sumido **reduzia** o risco.

| semanas sem acesso | risco devolvido | evasão real |
|---|---|---|
| 0 | 0,383 | 2,11% |
| 2 | 0,345 | 3,76% |
| 4 | 0,309 | 4,22% |

O risco cai enquanto a evasão real sobe, monotonicamente nas duas direções.

### O conserto

Coeficientes refeitos **sem** `class_weight`, sobre um painel de 564.006 linhas com split por coorte:

| | antes | depois |
|---|---|---|
| AUC | 0,764 | **0,783** |
| ECE | 0,275 | **0,006** |
| média prevista | 0,302 | **0,021** (real: 0,026) |
| lift@10% | 3,3× | **4,1×** |
| lift@5% | — | **5,6×** |
| equidade (cobertura N × Y) | — | 40% × **45%** |

**Nuance sobre `sem_clique`:** o coeficiente correto é ≈**zero** (−0,0008), não positivo. A relação **marginal** é forte e positiva, mas **condicionada aos cliques da semana ela desaparece** — "zero cliques" já codifica "sumiu". O erro do SQL não era só o sinal; era afirmar um efeito condicional que não existe.

**Quatro testes novos leem os coeficientes direto do arquivo `.sql`** e verificam que o aluno típico não recebe risco alto, que entrega perdida aumenta o risco, que mais cliques reduzem, e que sumir nunca reduz. Isso pega drift entre o arquivo e a documentação sem precisar de Postgres.

---

## Parte 4: `dominio` afirmava probabilidade sem ser

Mesma família dos defeitos do SQL. `dominio.dominio` devolve `p` como *"probabilidade de acertar a próxima"*, e só o AUC tinha sido medido.

### O `p` estava comprimido para o meio

| decil | previsto | observado | |
|---|---|---|---|
| 1 | 0,474 | **0,337** | −14 pts |
| 2 | 0,554 | 0,478 | −8 pts |
| 5 | 0,655 | 0,661 | ok |
| 9 | 0,769 | 0,860 | +9 pts |
| 10 | 0,825 | **0,937** | +11 pts |

**ECE 0,0564.** É o efeito conhecido de tirar média linear de duas probabilidades: comprime para o centro.

**Conserto:** correção de Platt em dois parâmetros, ajustada no treino e medida no teste.

| | AUC | ECE | faixa |
|---|---|---|---|
| antes | 0,725 | 0,0564 | 0,21 – 0,95 |
| **recalibrado** | 0,725 | **0,0083** | 0,04 – 1,00 |

Sete vezes melhor, AUC intacto. O `b ≈ 2` mede a compressão: a média linear encolhia o logito pela metade.

> **O ponto fixo é 0,6412** — resolvendo `a/(1−b)` no logito. **Não é 0,5 e não é a taxa base (0,6677).** Eu escrevi as duas coisas erradas: primeiro que era 0,5, depois que era a taxa base. Uma correção de Platt não preserva a média.
>
> A consequência tem sinal: acima de 0,6412 as previsões **sobem**, abaixo **descem**. Como a taxa base está acima do ponto fixo, a previsão média **sobe** — o oposto de "a recalibração centra na média do corpus". O teste antigo (`abs=0,01` contra 0,646) passava por folga.

### E `confianca` não media nada

A fórmula era `0,35 + 0,57·(1−e^(−n/8))`, indo de 0,48 a 0,92 conforme o aluno acumula respostas. Mas o erro real é **plano**:

| n no tópico | erro real \|p−y\| | desvio da taxa do aluno |
|---|---|---|
| 4–7 | 0,385 | 0,169 |
| 21–49 | 0,378 | 0,080 |
| 201+ | 0,382 | **0,022** |

O `\|p−y\|` não encolhe porque é dominado pelo **ruído de Bernoulli do resultado**, não pelo erro da estimativa. A confiança afirmava que a estimativa melhora muito com n, e não melhora — o que melhora é a certeza sobre a **taxa do aluno**.

`confianca` passa a ser `1 − 2·incerteza`, com `incerteza` sendo o desvio posterior da taxa. Teto honesto: **0,95**, porque com 201+ respostas a incerteza ainda é 0,022. E usa o posterior Beta para não devolver zero quando o aluno acertou ou errou tudo.

---

## Parte 5: o que depende dos limiares mudou junto

Recalibrar o `p` muda a **escala**. Tudo que corta nela precisava ser reconferido — e não estava.

### `precisa_reforco` passa a disparar 5× mais

| | dispara em | acerto de quem dispara |
|---|---|---|
| `p` comprimido (antes) | **2,3%** | 23,7% |
| `p` calibrado (agora) | **12,2%** | 36,8% |

**Não é regressão — é o limiar passando a significar o que diz.** Varrendo:

| limiar | dispara em | acerto observado | o limiar descreve? |
|---|---|---|---|
| 0,30 | 3,3% | 25,9% | sim |
| 0,40 | 8,3% | 33,2% | sim |
| **0,45** | **12,2%** | **36,8%** | sim |
| 0,50 | 17,2% | 40,5% | sim |

Em toda a faixa, quem dispara com `p < L` de fato acerta menos de `L`. Antes, `p < 0,45` selecionava só os 2,3% mais extremos porque tudo estava espremido em torno de 0,65.

> **Consequência de produto, e é grande:** quem atualizar o módulo e mantiver `limiar=0,45` recebe **cinco vezes mais** chamadas de reforço. Para manter o volume antigo, `limiar=0,28`. Para manter o significado, mantenha 0,45 e dimensione para 12%. Está no docstring da função.

### `tempo.demorando`: o limite de 3× confere, mas é conservador

| lentidão | acerto | respostas |
|---|---|---|
| 0 – 0,5× | 73,6% | 30.056 |
| 0,5 – 1× | 72,7% | 119.999 |
| **1 – 2×** | **62,7%** | 125.431 |
| 2 – 3× | 54,0% | 16.667 |
| 3 – 5× | 53,8% | 5.957 |
| 5 – 10× | 53,2% | 1.573 |

Acima de 3×: **53,8% contra 67,4%** abaixo — confirma o que o módulo afirmava.

Mas **a queda acontece entre 1× e 2×** (72,7% → 62,7%), já está em 54% na faixa de 2–3×, e acima disso é plano. Com 3× só **2,6%** das respostas são marcadas; baixar para 2,0 marca ~8% com praticamente a mesma separação. É escolha de operação, e agora está documentada.

---

## Parte 6: `duracao_prevista` somava medianas

**Soma de medianas não é a mediana da soma.** A função somava as medianas das questões para prever o tempo de uma atividade inteira, e o erro é sistemático:

| questões | soma das medianas | mediana real | média real | erro |
|---|---|---|---|---|
| 5 | 102 s | 117 s | 140 s | **−13%** |
| 10 | 207 s | 239 s | 291 s | **−13%** |
| 20 | 404 s | 466 s | 594 s | **−13%** |

Estável nos três tamanhos. A latência é assimétrica à direita e a soma se concentra na média, não na mediana.

**Conserto:** fator de correção medido, `1,155`. Leva 207 s a 239 s, a mediana real.

Testei também somar **médias** em vez de medianas — acerta a mediana total sem correção (243 s contra 239 s reais) mas tem erro absoluto pior (26% contra 24%). Ficou mediana + fator.

> **E a precisão foi documentada, porque não estava:** mesmo corrigida, só **~62%** das atividades de 10 questões caem dentro de ±25% do tempo real, com erro absoluto médio de **24%**. Serve para dizer "uns 4 minutos", não para cronometrar.

### `perfil_chute` confere exatamente

| | no módulo | medido |
|---|---|---|
| p90 | 0,020 | **0,0196** |
| p99 | 0,079 | **0,0786** |

As faixas pegam 9,7% e 1,0% dos alunos — exatamente o que os percentis prometem. Nenhum ajuste necessário.

---

## Parte 7: `prioridade_revisao` prometia ganho, entrega esquecimento

A função é `1 − retenção`, e o docstring dizia *"quanto esta questão precisa ser revisada agora"* — o que sugere que ela encontra **onde a revisão rende mais**.

Testado olhando o encontro **seguinte** ao reencontro, entre os que erraram no reencontro (60.433 casos):

| retenção prevista | acerta no encontro seguinte |
|---|---|
| 0,62 | 64,6% |
| 0,74 | 66,7% |
| 0,83 | 64,8% |
| 0,85 | 71,0% |
| 0,89 | **70,9%** |

**Item mais esquecido não rende mais depois — rende menos.** Entre os que acertaram no reencontro o padrão se repete (79,2% a 90,7%).

> **Mas isso não prova o contrário.** Os itens de baixa retenção são mais difíceis e mais antigos: iriam pior de qualquer jeito. **Sem sortear o momento da revisão, dado observacional não separa as duas coisas.** O que se pode afirmar é o que a função calcula, não o que ela sugeria calcular.

**Conserto:** entra `esquecimento()`, que é o mesmo número com o nome literal. `prioridade_revisao` vira alias, com a ressalva no docstring: tratar isso como "onde a revisão rende mais" é suposição de quem usa, não achado deste corpus.

---

## O que a auditoria **não** cobre

- **A transferência para o TrailUp.** Continua sem medida: o banco está vazio.
- **A transferência para o TrailUp.** Continua sem medida: o banco está vazio. O que esta auditoria garante é que os números publicados descrevem o que o código faz **no corpus de referência** — não que se sustentem em dado brasileiro e escolar.
- **Calibração dos que dependem de coorte.** `engajamento.ordenar` e `gate.risco` devolvem score para ordenar, não probabilidade; a auditoria confere a ordenação, não um nível absoluto que eles não afirmam ter.

---

## Parte 8: constantes estabelecidas sem base

Inventariei toda constante dos módulos e separei as que tinham medida das que eram escolha não justificada.

### `ritmo.FRONTEIRA_SEG = 40 s` — tem base, e melhor do que se dizia

Maximizando a separação entre as duas nuvens no log da mediana (critério de Otsu) sobre 11.421 questões, **o corte ótimo é 39,8 s**. O módulo usa 40,0.

| | |
|---|---|
| d de Cohen no corte de 40 s | **4,18** (o relatório dizia 3,30) |
| questões classificadas "rápidas" | 65% |
| distribuição das medianas | p25 18 s · **p50 22 s** · **p75 64 s** · p90 95 s |

O salto de 22 s para 64 s entre o p50 e o p75 é a bimodalidade que justifica **categoria** aqui, em vez de intervalo. Constante confirmada, e o `d` documentado estava subestimado.

### `tempo.MIN_RESPOSTAS = 5` — **não tem base**

O comentário dizia *"abaixo disso a mediana ainda oscila demais"*, sugerindo que a partir de 5 ela não oscila. Medido em 11.421 questões, contra a mediana com 50+:

| n | correlação | erro relativo típico |
|---|---|---|
| 3 | 0,854 | 25,9% |
| **5** | **0,886** | **20,2%** |
| 8 | 0,917 | 15,2% |
| 15 | 0,942 | 11,5% |
| 30 | 0,969 | 7,7% |

**Com 5 respostas o erro típico é de 20%.** Isso não é "estável".

A distinção que faltava: **5 serve para o ritmo e não serve para o tempo.** Classificar rápida/lenta acerta 97,3% com 5 respostas porque o corte de 40 s fica longe da mediana da maioria das questões — um erro de 20% raramente atravessa a fronteira. Mas `esperado()` devolve a mediana **como duração**, e aí os 20% entram direto no número.

Entra `tempo.precisao(n)`, que devolve o erro típico daquela estimativa, e o comentário passa a dizer o que a curva mostra: para ±10%, exija ~15 respostas; para ±8%, ~30.

### A que eu ia "melhorar" e teria piorado o sistema

`engajamento.JANELA_RECENTE = 10` foi escolhido sem medida, e é o eixo mais forte do módulo. Varrido nas duas bases:

| janela | EdNet | OULAD |
|---|---|---|
| 5 | 0,767 | 0,834 |
| 7 | 0,789 | 0,857 |
| **10 (padrão)** | 0,810 | **0,863** |
| **14** | **0,824** | 0,862 |
| 20 | **0,827** | 0,847 |
| 30 | 0,799 | 0,811 |

Lido assim, **14 é melhor**: ganha 0,014 no EdNet com intervalos que mal se tocam, e empata no OULAD. Recomputei tudo que dependia de 10 — tabela `REFERENCIA`, pesos do ordenador, cortes — para trocar.

**E o modelo compartilhado desabou:**

| janela | modelo único | só aquela base |
|---|---|---|
| 10 | EdNet **0,856** | 0,854 |
| **14** | EdNet **0,812** | 0,853 |
| 14 | OULAD 0,855 | 0,855 |

Com 14 dias a recência ocupa metade da janela de 30 e fica **colinear com a frequência** — o peso da frequência chega a ficar **negativo** (−0,025), sinal de ajuste degenerado. As duas bases divergem mais e os pesos compartilhados servem pior a ambas.

> **Otimizar o eixo isolado piorava o sistema em 0,044.** É a segunda vez nesta sessão que afinar uma parte degrada o todo — a primeira foi a logística linear ganhando AUC e perdendo calibração. Um teste novo trava o sintoma: peso de frequência negativo é ajuste degenerado, e falha.

E o **30 é pior que tudo de 5 para cima nas duas bases** — a janela cheia dilui. É isso que sustenta a recência existir como eixo separado da frequência.

### A aproximação normal: confere — e eu quase a reprovei medindo a coisa errada

`dificuldade` diz usar aproximação normal do posterior, *"validada contra o Beta exato, diferença de cobertura menor que 1 ponto"*.

Medi a diferença entre os **limites** dos intervalos e deu até **2,56 pontos** — ia marcar como defeito. Mas a afirmação é sobre **cobertura**, não sobre limites:

| | cobertura |
|---|---|
| normal (o módulo) | 75,3% |
| Beta exato | 74,8% |

**Meio ponto. A afirmação confere.** Errar o objeto da medida é o mesmo erro que esta auditoria encontrou nos outros — desta vez fui eu, no teste.

Fica documentado o que eu de fato descobri: **os limites diferem em até 2,6 pontos** em n pequeno com taxa extrema (n=10 com 1 acerto: [0,169–0,539] normal contra [0,179–0,549] exato). Quem decide por `afirmar()` não se importa; quem exibir o limite cru ao professor com n baixo, sim.

### `engajamento.MIN_EVENTOS = 30` — mesmo defeito do `tempo`

O comentário dizia *"abaixo disso os eixos são ruído"*, sugerindo que a partir de 30 funcionam. AUC da recência por faixa de eventos:

| eventos | alunos | AUC |
|---|---|---|
| **30–49** | 4.708 | **0,644** |
| 50–99 | 5.032 | 0,718 |
| 100–299 | 5.440 | 0,801 |
| 300–999 | 2.491 | **0,864** |

**No mínimo de 30 o eixo entrega 0,644, não os 0,810 do agregado.** O número de capa vem dos alunos com muita atividade.

30 fica como **piso para não devolver lixo**; para o desempenho anunciado, exija ~100. É a terceira vez que um mínimo vinha com comentário sugerindo suficiência — depois de `tempo.MIN_RESPOSTAS` e da confiança do `ritmo`.

### `dias_ate_revisar` quebra a promessa em 4 pontos

A função promete: *"revise em N dias e a retenção estará no alvo"*. Conferido — quem reencontra perto do dia sugerido acerta **sistematicamente menos**:

| alvo | dias sugeridos | acerto real | diferença |
|---|---|---|---|
| 0,60 | 37,5 | 57,0% | **−3,0** |
| 0,70 | 4,1 | 65,7% | **−4,3** |
| 0,75 | 2,1 | 70,4% | **−4,6** |
| 0,80 | 0,8 | 75,2% | **−4,8** |
| 0,85 | 12,2 | 81,3% | **−3,7** |

**Sempre para o mesmo lado**, ~4 pontos em todos os alvos. A causa provável: a curva foi medida sobre **todos** os reencontros, mas quem de fato volta perto do prazo é um subconjunto selecionado — e pior que a média.

Entra `MARGEM_ALVO = 0,04`: o prazo passa a ser calculado para `alvo + margem`, que é o que entrega o alvo pedido na prática. Com alvo 0,75, o prazo cai de 2,1 para 1,1 dia. `margem=0` devolve o prazo cru da curva.

### `gate.MIN_RESPOSTAS = 5` — confere, e é o cotovelo exato

| n no tópico | casos | AUC |
|---|---|---|
| 1–2 | 127.273 | **0,534** |
| 3–4 | 119.650 | 0,610 |
| **5–9** | 265.976 | **0,660** |
| 20–49 | 865.299 | 0,714 |
| 50+ | 3.801.801 | 0,738 |

Abaixo de 5 o sinal é quase acaso. O corte está no lugar certo — e não era escolha justificada até esta medida.

### `pre_avaliacao.divagacao` — o último, e confirma o padrão

O módulo devolve três campos auxiliares como porcentagem, e só a nota tinha sido validada. Spearman com a nota humana:

| campo | Spearman | |
|---|---|---|
| `cobertura` | **+0,462** | forte — quase o da nota inteira (+0,483) |
| `conceitos_faltando` | **−0,410** | forte |
| `divagacao` | **−0,114** | **fraco, e a escala engana** |

`cobertura` é monotônica e interpretável (nota real 2,76 → 3,45 → 3,76 → 3,96 por faixa). `conceitos_faltando` também (3,94 → 3,15).

**`divagacao` não é o que o nome diz.** É a fração dos tokens fora do gabarito e do enunciado — e em prosa a maioria das palavras está fora mesmo. A distribuição real:

| p10 | mediana | p90 |
|---|---|---|
| 0,57 | **0,72** | 0,83 |

**Divagar 72% é o normal.** O exemplo do módulo exibia *"divaga 22%"* como se fosse um valor comum; na prática 22% é excepcionalmente baixo. Um professor lendo "divaga 72%" sobre uma resposta de nota 4 entenderia o oposto.

Corrigido: o rótulo passa a ser `fora do gabarito 45% (típico ~72%)`, o campo está marcado no dataclass, e dois testes travam — que uma resposta boa também tem divagação alta, e que a string não chama mais isso de divagação.

### As que são arbitrárias e **não importam**

`dominio.PRIOR_ALUNO = 3` e `PRIOR_GLOBAL = 8` foram escolhidos sem medida. Varridos sobre 120.000 previsões, com IC95:

| `PRIOR_GLOBAL` | AUC | | `PRIOR_ALUNO` | AUC |
|---|---|---|---|---|
| 2 | 0,7238 | | 1 | 0,7239 |
| **8** | **0,7241** | | **3** | **0,7241** |
| 32 | 0,7236 | | 8 | 0,7240 |
| 64 | 0,7229 | | 16 | 0,7237 |

**A AUC varia na quarta casa** e todos os intervalos se sobrepõem. São arbitrários, e está medido que não importam nessa faixa — o que é diferente de "não tem base" e diferente de "tem base".

### As que têm um platô, e onde ele termina

`pre_avaliacao.JANELA = 4` e `DF_STOPWORD = 0,5`, com IC95 por bootstrap agrupado por aluno:

| configuração | Spearman | IC 95% |
|---|---|---|
| JANELA=3 | 0,472 | [0,426 – 0,516] |
| **JANELA=4 (padrão)** | **0,483** | [0,440 – 0,526] |
| JANELA=4, df=0,4 | 0,501 | [0,456 – 0,542] |
| JANELA=5 | 0,477 | [0,435 – 0,517] |
| **JANELA=8** | **0,388** | **[0,343 – 0,438]** |

> **Não mexi.** De 3 a 5 na janela, e de 0,3 a 0,6 no `df`, os intervalos se sobrepõem inteiramente — são indistinguíveis com 1.167 respostas. Trocar 0,483 por 0,501 seria ajustar a ruído, que é exatamente o erro que esta auditoria existe para não cometer.
>
> O que o dado **sustenta** é a fronteira: a partir de janela 8 a qualidade cai fora do intervalo.

---

## Parte 9: varredura adversarial — quatro defeitos meus

Atacando as afirmações mais frágeis que eu mesmo tinha escrito, em vez de repetir as checagens que já passavam.

### 1. "O ponto fixo é a taxa base" — falso

| | |
|---|---|
| ponto fixo analítico `a/(1−b)` | **0,6412** |
| taxa base do corpus | 0,6677 |
| o que eu afirmei (docstring, relatório, teste) | "é a taxa base" |

Errei duas vezes seguidas no mesmo ponto: primeiro disse 0,5, depois disse a taxa base. E o teste que escrevi para "travar" isso usava `abs=0,01` contra 0,646 — **passava por folga sobre o valor errado**.

Pior: a direção também estava errada no meu texto de correção. Acima do ponto fixo as previsões **sobem**; escrevi que desciam.

### 2. `prever_turma` derivava a variância do intervalo **clipado**

```python
var_post = ((base.maximo - base.minimo) / (2 * z)) ** 2   # errado
```

`estimar()` clampa os limites em [0, 1]. Numa questão de **30/30** o posterior sai truncado em 1,0, a largura aparente encolhe, e a variância inferida fica **pequena demais — justo onde a incerteza é maior**.

Corrigido para calcular a variância do posterior Beta diretamente, com teste que verifica que não há salto artificial de 29/30 para 30/30.

### 3. `gate.risco` tinha um parâmetro cujo nome invertia o sentido

O parâmetro se chamava `dificuldade_media_topico` e recebia **taxa de acerto** (facilidade). Medido:

| valor passado | risco devolvido |
|---|---|
| 0,9 (lido como "muito difícil") | **0,837** |
| 0,2 (lido como "muito fácil") | **0,890** |

Quem lesse só a assinatura passaria a dificuldade e **inverteria o termo**. Renomeado para `acerto_medio_topico`, com teste que fixa a direção.

### 4. `dominio.incerteza` ignorava `media_global`

Fixava `0,67`/`0,33` no prior mesmo quando o chamador declarava outra média global — a confiança devolvida era idêntica com `media_global=0,20` e `0,90`. Corrigido e propagado até `dominio()`.

---

## Parte 10: segunda varredura — interações e casos-limite

Atacando o que nenhuma rodada anterior tocou: o que acontece **entre** os módulos e nas bordas.

### 1. A recalibração quebrou o limiar de `tendencia` — e eu não tinha conferido

Corrigi `precisa_reforco` quando a recalibração mudou a escala. **Não olhei `tendencia`, que corta na mesma escala.**

| | variação de `p` para a mesma evidência |
|---|---|
| escala crua | +0,046 |
| recalibrada | **+0,091** |

O limiar de 0,03 passaria a disparar com **metade** da evidência. Corrigido calculando a tendência **antes** de expandir, com teste que exige que ela não dependa da recalibração.

E `LIMIAR_TENDENCIA = 0,03` virou constante nomeada, **declarada como escolha** — nenhum experimento a definiu.

### 2. `retencao` aceitava parâmetros contraditórios em silêncio

`retencao(7, acertou_antes=True, proporcao_acertos=0.0)` devolvia o mesmo que com `acertou_antes=False`. O parâmetro era aceito e ignorado. Agora levanta nos dois casos contraditórios.

### 3. `trajetoria`: o limiar depende de quantas janelas você passa

Para a **mesma queda de 8 para 1**:

| janelas | inclinação |
|---|---|
| `[8, 1]` | **−7,00** |
| `[8, 4, 1]` | −3,50 |
| `[8, 6, 4, 1]` | **−2,33** |

`incl` é a variação média **por janela**. Os números do módulo foram medidos com **três janelas de 10 dias**; com outra divisão o mesmo padrão muda de rótulo. Documentado, com recusa acima de 6 janelas.

### 4. `derivar_prior` devolvia força 210.000

| corpus | resultado antes |
|---|---|
| todas as questões com a mesma taxa | **Beta(146.999; 62.999)** — força 209.999 |
| bimodal (0,01 e 0,99) | **Beta(0,02; 0,02)** — força 0,04 |

O primeiro vem de `max(var − ruido, 1e-6)`: quando não há variação genuína, a divisão por 1e-6 explode. **Um prior de força 210 mil significa que 30 respostas não moveriam a estimativa em nada** — e a função devolvia isso sem avisar.

O segundo é um Beta **bimodal**, com massa em 0 e 1 — o oposto de "dificuldade típica".

Agora recusa nos dois casos, com `FORCA_MIN, FORCA_MAX = 0,5, 200` e mensagem que diz o que fazer.

### 5. `demorando` inverte com a imprecisão da mediana

Com `respostas=5` a mediana tem 20% de erro. Para **55 s**, o mesmo aluno:

| mediana da questão | `demorando(55s)` |
|---|---|
| 16 s (−20%) | **True** |
| 24 s (+20%) | **False** |

A decisão inverte dentro da margem de erro do próprio estimador. Já estava documentado que 5 não é preciso; aqui está a consequência concreta.

---

## Parte 11: fuzzing — 21 propagações de NaN e infinito

Classe que nenhuma rodada anterior tocou: **entrada hostil**. Toda função pública chamada com negativo, zero, NaN, infinito, vazio e valores absurdos. O contrato: ou levanta erro claro, ou devolve valor são — nunca NaN, nunca infinito.

**Primeira execução: 21 problemas.** Agora **3**, todos a sentinela intencional.

### O pior: `confirmar()` dizia "não suspeita" para questão não-mensurável

`discriminacao` devolve **NaN** como sentinela de `indeterminado` — quando n é baixo, ou quando **todo mundo acertou** (variância zero, que acontece em dado real).

O problema é que `NaN < limiar` é **False**. Então:

```python
confirmar(float('nan'), 0.1)  # -> False, "não suspeita"
```

Uma questão que **não deu para medir** era silenciosamente classificada como boa. Agora levanta com a mensagem: *"trate como 'sem evidência', não como 'questão ok'"*.

### As outras 20

| tipo | funções atingidas |
|---|---|
| NaN/infinito propagando até a saída | `estimar`, `prever_turma`, `ritmo`, `esperado`, `esperado_de_amostra`, `duracao_prevista`, `quao_lento`, `incerteza` |
| exceção obscura em vez de erro claro | `retencao(NaN)` → `IndexError`, `calibrar` sem campos → `KeyError` |
| valor sem sentido aceito | `dias_ate_revisar` com alvo 0 ou negativo devolvia infinito |
| lista com valores negativos | `duracao_prevista([-10, 20])` somava normalmente |

Todas passam a validar na entrada. `calibrar` agora **diz qual campo falta**, em vez de `KeyError: 'dias_recentes'`.

O fuzzer virou script permanente: [`80_fuzz.py`](scripts/80_fuzz.py). Roda sem dataset e falha se passar de 3 problemas.

---

## Parte 12: as faixas do SQL estavam mortas, e a curva não é monotônica

### As faixas de `evasao.sql` ficaram na escala antiga

Consertei os coeficientes e **deixei os limiares**. Terceira vez neste módulo — as outras foram `precisa_reforco` e o limiar de `tendencia`.

| faixa | limiar antigo | pegava, na escala nova |
|---|---|---|
| `alto` | ≥ 0,28 | **0,27%** dos alunos-semana |
| `atencao` | ≥ 0,15 | 0,63% |

**A precisão continuava alta** (25% de evasão entre os marcados) **e a cobertura era nula** — a função praticamente não classificava ninguém. É o modo de falha que passa despercebido: não há erro, só silêncio.

Distribuição real do risco na escala corrigida: média 0,0212, p70 **0,0213**, p90 **0,0428**.

| faixa nova | limiar | casos | evasão | lift |
|---|---|---|---|---|
| `alto` | ≥ 0,0428 | 10,0% | **9,96%** | 3,8× |
| `atencao` | ≥ 0,0213 | 20,0% | 4,24% | 1,6× |
| `normal` | — | 70,0% | 1,11% | 0,4× |

> **Detalhe que quase me pegou de novo:** arredondei 0,0428 para 0,043 e o teste falhou — 0,043 fica **acima** do p90, e a faixa passaria a pegar menos de 10%. Os limiares são os percentis exatos.

### A curva de esquecimento não é monotônica

Varredura de monotonicidade ao longo de **todo o domínio** (não em pontos isolados) achou quatro violações. Duas eram artefato da minha varredura; **duas são reais**:

```
_ACERTOU = [0,901, 0,907, 0,873, 0,859, 0,835, 0,819, 0,831]
             ↑ sobe de 0,04 para 1 dia          ↑ sobe de 60 para 180
```

`_ERROU` é estritamente decrescente. A de quem **acertou** sobe em dois pontos — compatível com consolidação no primeiro caso e com **seleção** no segundo (quem reencontra uma questão depois de 180 dias é quem continuou estudando). Não dá para separar com dado observacional, então a tabela fica como medida, **documentada**.

Consequência: `esquecimento` também não é monotônico para quem acertou, e `dias_ate_revisar` devolve o **primeiro** cruzamento — não o único.

### E `incerteza` não é função só de `n`

O docstring dizia *"é o que encolhe com n"*. Encolhe **para uma taxa fixa**; numa série real a taxa muda a cada resposta e a incerteza pode subir: `incerteza(10, 7) = 0,1233` contra `incerteza(11, 7) = 0,1237`.

O script de monotonia virou permanente: [`82_monotonia.py`](scripts/82_monotonia.py), com as duas exceções medidas declaradas.

---

## Mapa de cobertura da API, sem dataset

```bash
python3 docs/auditoria/scripts/79_cobertura.py
```

Enumera **toda função pública e todo campo de dataclass** do pacote e checa se existe teste. Roda sem dataset, e falha se sobrar símbolo descoberto.

**Foi escrito porque eu afirmei duas vezes que "está tudo auditado" sem conferir.** Na primeira execução apontou **16 símbolos sem teste**, incluindo `chute.foi_chute` — a função principal daquele módulo, usada em toda a auditoria e nunca testada diretamente.

Fechado: **0 símbolos descobertos**. O script sai com código 1 se algum voltar.

E o teste que escrevi para fechar a lacuna encontrou outra coisa: `ritmo.descrever(faixa, r)` recebe uma `dificuldade.Faixa` no primeiro parâmetro, não o rótulo do ritmo — e o parâmetro estava **sem anotação de tipo**, o que escondia isso de quem lesse a assinatura.

## Verificação de consistência, sem dataset

```bash
python3 docs/auditoria/scripts/70_consistencia.py
```

Compara o mesmo número citado em arquivos diferentes e falha se divergirem. Não precisa de dataset — lê só o repositório. Rodando depois da auditoria, apareceram **três divergências** que nenhum teste pegaria:

| | módulo dizia | documento dizia |
|---|---|---|
| `ritmo`, acerto com 5 | 97,3% | 96,4% (a primeira remedição, superada) |
| `evasao`, AUC | 0,783 (refeito) | 0,748 (anterior ao conserto) |
| `pre_avaliacao`, Spearman | 0,483 | 0,447 (versão de 8 features) |

Todas corrigidas. O valor antigo da evasão fica anotado de propósito no relatório histórico, e o verificador o conhece como exceção.

## Como reexecutar

```bash
python3 docs/auditoria/scripts/60_auditoria.py    # dificuldade, ritmo, tempo, dominio
python3 docs/auditoria/scripts/61_auditoria2.py   # revisao, chute, gate
python3 docs/auditoria/scripts/63_auditoria3.py   # pre_avaliacao, engajamento
```

Exige os datasets (ver [`docs/LEIA-ME_scripts.md`](../LEIA-ME_scripts.md)) e `pandas`/`numpy`. O pacote em si continua sem dependência.

> **Nota de execução:** o `61` é a versão vetorizada. A primeira versão usava `groupby.transform` com lambda sobre 6,5 M linhas e não terminava — o mesmo gargalo que já apareceu duas vezes neste projeto.
