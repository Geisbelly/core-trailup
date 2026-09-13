# Auditoria — cada número reconferido chamando o código

**Data:** 2026-09-13 · **Scripts:** [`scripts/`](scripts/)

Todo número afirmado nos cabeçalhos dos módulos foi remedido **chamando as funções do módulo**, não reimplementações. É o que pega divergência entre o que o código faz e o que a documentação diz.

Base: EdNet KT3, 6.504.124 respostas, acerto global 0,6677, split por aluno 70/30.

---

## Resultado das 7 partes

**27 verificações. 11 defeitos encontrados**, todos da mesma família: uma saída afirmando uma escala que ninguém mediu. Nenhum apareceria olhando AUC.

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

E quatro que **conferiram**: `perfil_chute` (p90 e p99), a tabela `REFERENCIA` do engajamento (diferença 0,000), o limite de 3× do `demorando`, e `ritmo.FRONTEIRA_SEG` — cujo 40 s é praticamente o corte ótimo por Otsu (39,8 s), com d de Cohen **maior** que o documentado (4,18 contra 3,30).

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

> **O ponto fixo é 0,646, não 0,5** — o corpus acerta 67%, e é em torno da taxa base que a expansão acontece. Um teste fixa isso, porque a intuição errada (expandir em torno de 0,5) passou primeiro.

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
