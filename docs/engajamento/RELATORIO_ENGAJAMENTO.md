# Engajamento — quem vai continuar estudando

**Base:** EdNet KT3 + OULAD (replicação em ARES) · **Módulo:** [`engajamento.py`](../../trailup_core/engajamento.py) · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê)

---

## 1. Pergunta

> **Olhando só o que o aluno fez no primeiro mês, dá para dizer quem ainda vai estar estudando no mês seguinte?**

Não é a mesma pergunta que "quem vai aprender mais". A §10 mostra que são perguntas diferentes e que este modelo responde só a primeira.

---

## 2. Dados

Duas plataformas deliberadamente distantes, mais uma terceira que serve de limite de escopo.

| | EdNet KT3 | OULAD | ARES |
|---|---|---|---|
| quem | adulto coreano | universitário britânico | universitário |
| o quê | TOEIC, autoestudo | disciplina a distância | leitura em L2 |
| turma? | não | sim | sim |
| continuar é | voluntário | voluntário com custo | **obrigatório** |
| unidade de atividade | resposta a questão | clique em material | leitura de texto |
| alunos na análise | **18.119** | **24.731** | 171 |
| retenção base | **30,5%** | **92,9%** | 87,7% |
| licença | CC BY-**NC** | CC BY 4.0 | CC BY 4.0 |

**Unidade de análise: o aluno.** Uma linha por aluno, não por evento.

**Filtros, nesta ordem:** ter 60 dias de histórico disponível na base (sem isso não dá para distinguir "abandonou" de "a base acabou" — censura à direita), e ≥30 eventos na janela de medida (abaixo disso é ruído amostral). É a origem do `MIN_EVENTOS = 30` no módulo.

---

## 3. Desenho

Para cada aluno, o dia 0 é o **da sua primeira atividade** — não uma data de calendário, porque alunos entram em momentos diferentes.

```
    dia 0                        dia 30                       dia 60
      |───── JANELA DE MEDIDA ─────|───── JANELA DE DESFECHO ────|
      |    calcula os eixos        |   observa se voltou ou não  |
      |    (só olha para cá)       |   (nunca entra nos eixos)   |
```

**Nada da segunda janela entra no cálculo dos eixos.** É isso que torna o resultado uma previsão, e não uma descrição.

No ARES a janela é 14/14, porque o curso dura seis semanas.

### O desfecho

```
voltou = 1 se houve ao menos um evento entre o dia 30 e o dia 59
```

Binário. O OULAD tem também **desmatrícula formal** registrada, mas só o desfecho de atividade é comparável entre as três bases; é ele que o estudo usa.

---

## 4. As medidas

Contagens sobre o que a plataforma já registra. Nada inferido.

| eixo | a conta | o que captura |
|---|---|---|
| **recência** | dias distintos com atividade nos **últimos 10** dias da janela | apareceu ultimamente? |
| **frequência** | dias distintos com atividade ÷ 30 | hábito ao longo do mês |
| **profundidade** | tempo mediano em material de conteúdo (log) | trata o erro como informação? |

Distribuição, em unidade crua:

| | EdNet | OULAD |
|---|---|---|
| dias ativos em 30 — mediana | **4** | **11** |
| profundidade — mediana | 10,8 s lendo explicação | 56% dos cliques em conteúdo |

No EdNet, **metade dos alunos aparece em 4 dias ou menos de 30**. É plataforma de estudo esporádico, e isso muda como ler todo o resto.

> **A tradução entre plataformas é aproximada, e isso é uma limitação real.** Frequência é a única medida com equivalência exata. Profundidade é "segundos lendo a explicação" no EdNet e "fração dos cliques em material de conteúdo" no OULAD — defensável, não canônica.

---

## 5. A métrica

**AUC** responde a uma pergunta de pares: *pego um aluno que voltou e um que não voltou, ao acaso — qual a chance de a medida dar valor mais alto para quem voltou?* 0,50 é cara ou coroa; **abaixo de 0,50 a medida prediz ao contrário**.

**ECE** (erro esperado de calibração): diferença média entre a probabilidade prevista e a frequência observada. Diz se o número significa o que diz.

**Acurácia balanceada**: média entre sensibilidade e especificidade. Existe aqui porque a acurácia simples é inútil quando a taxa base vai de 7% a 70% (§11).

Intervalos de confiança por bootstrap.

---

## 6. Resultado: o que prediz, e o que replica

| eixo | EdNet | IC 95% | OULAD | IC 95% | replica? |
|---|---|---|---|---|---|
| **recência** | **0,810** | [0,804 – 0,818] | **0,863** | [0,854 – 0,870] | **sim** |
| **frequência** | **0,799** | [0,792 – 0,806] | **0,811** | [0,800 – 0,820] | **sim** |
| tendência | 0,784 | [0,775 – 0,794] | 0,737 | [0,724 – 0,752] | sim |
| profundidade | 0,624 | [0,617 – 0,634] | 0,564 | [0,549 – 0,581] | sim, fraco |
| ~~volume por sessão~~ | 0,400 | [0,393 – 0,411] | 0,522 | [0,507 – 0,536] | **não** |
| *(referência)* taxa de acerto | 0,513 | [0,503 – 0,522] | — | — | *nada* |

Três leituras:

**Recência é o melhor eixo, nas duas bases.** Dias ativos nos últimos 10 batem dias ativos nos 30. "Apareceu ultimamente" vale mais que "aparece bastante".

**Desempenho não prevê permanência.** A taxa de acerto dá AUC 0,513, com intervalo quase tocando 0,50. O bom aluno abandona na mesma proporção que o ruim.

**O volume não sobreviveu à segunda base** e foi removido do módulo — ver §14.

### Onde nada funciona: o ARES

| eixo | AUC |
|---|---|
| frequência | 0,492 |
| volume | 0,522 |
| profundidade | 0,565 |

Tudo acaso, e a razão é estrutural: **a leitura no ARES é tarefa obrigatória**. 87,7% voltam e não há variação a prever.

> **Escopo declarado do módulo:** ele pressupõe **continuação voluntária**. Se o TrailUp for usado dentro da aula, com o professor mandando, estes eixos medem pouco. É pergunta de produto antes de ser de modelo.

---

## 7. A curva de recência, sem modelo nenhum

Retenção observada por dias ativos nos últimos 10:

| dias ativos | EdNet | OULAD |
|---|---|---|
| 0 | **12,4%** | **61,3%** |
| 1 | 47,3% | 88,6% |
| 2 | 55,5% | 93,5% |
| 3 | 68,1% | 96,3% |
| 4 | 75,9% | 98,2% |
| 6 | 88,7% | 99,2% |
| 8 | 93,3% | 99,9% |
| 10 | **98,2%** | **100,0%** |

**Monotônica nas duas**, com níveis completamente diferentes. É a **forma** que transfere; o nível, não.

E é uma tabela de consulta — sem modelo, sem dependência, auditável linha a linha. É o que o módulo usa como base.

---

## 8. Um modelo só para as duas bases

### Juntar as bases cruas deixa o modelo trapacear

**Teste:** prever, a partir dos eixos, **de qual base a linha veio**.

| representação | AUC para adivinhar a base |
|---|---|
| valor absoluto (3 eixos) | **0,994** |
| **z dentro da coorte** | **0,508** |
| percentil dentro da coorte | 0,493 |

Com valores absolutos as bases são quase perfeitamente separáveis: o modelo não precisa medir engajamento, basta identificar a plataforma e aplicar a taxa dela. E isso não é teórico — treinado assim, **o EdNet piora de 0,849 para 0,788**.

**Padronizar dentro de cada coorte antes de juntar leva a identificabilidade a 0,508** — indistinguível de cara ou coroa.

### Com isso, um modelo único empata com os específicos

| base de teste | modelo único | modelo só daquela base |
|---|---|---|
| EdNet | **0,856** | 0,854 |
| OULAD | **0,862** | 0,862 |

> **Correção de 2026-09-13.** Os valores 0,856 e 0,862 vinham de **uma** partição (semente 7). Em 12 partições independentes a média é **0,846 ± 0,006** no EdNet e **0,868 ± 0,004** no OULAD — o 0,856 fica **1,8 desvios acima** da média, era uma semente sortuda. Os pesos, esses, são indiferentes: a média das 12 dá a mesma AUC na quarta casa.


E ordena bem numa base que nunca viu: **EdNet → OULAD 0,865**, **OULAD → EdNet 0,842**.

Os pesos, sobre z da própria coorte:

| termo | peso |
|---|---|
| intercepto | +0,846 |
| **recência** | **+0,829** |
| frequência | +0,136 |

A recência pesa **6 vezes** mais que a frequência.

### A profundidade fica fora do ordenador

| eixos no modelo único | EdNet | OULAD |
|---|---|---|
| recência + frequência | **0,856** | 0,862 |
| + profundidade | 0,848 | 0,862 |

Ela **piora** o modelo único, porque significa coisas diferentes em cada plataforma. Dentro de uma base ajuda; entre bases, injeta ruído. Fica como eixo de **diagnóstico** (o que está acontecendo), fora do de **predição** (quem está em risco).

---

## 9. A ordem transfere; o nível não

Exportando o modelo de uma base para a outra, sem recalibrar:

| treino → teste | AUC | **ECE** |
|---|---|---|
| EdNet → OULAD | 0,869 | **0,359** |
| OULAD → EdNet | 0,832 | **0,571** |

AUC alto, calibração catastrófica: o modelo do OULAD aplicado ao EdNet erra **57 pontos percentuais** em média entre previsto e observado.

E os cortes absolutos também não transferem. Aplicando a régua de frequência do EdNet ao OULAD, **na mesma unidade**:

| | baixo | médio | alto |
|---|---|---|---|
| EdNet | 42,2% | 24,2% | 33,5% |
| **OULAD** | **6,0%** | 16,7% | **77,2%** |

O mesmo ordenador precisa de limiares diferentes para alertar os mesmos 10%:

| base | saem | limiar | precisão | cobertura |
|---|---|---|---|---|
| EdNet | 69,8% | **0,440** | 94,9% | 13,6% |
| OULAD | 7,1% | **0,582** | 38,7% | 54,4% |

**Daí a arquitetura:** ordenador comum compartilhado, e limiar, faixas e probabilidade esperada **por coorte**. É o que `ordenar()` e `calibrar()` implementam — e é por isso que `ordenar()` levanta erro sem calibração, em vez de devolver um número emprestado.

---

## 10. O limite: isto mede permanência, não aprendizado

Terceiro desfecho testado — **ganho de taxa de acerto** entre as duas janelas, entre os 5.523 do EdNet que voltaram:

| medida | Spearman com o ganho |
|---|---|
| recência / frequência / profundidade | **−0,04 a +0,03** |
| taxa de acerto na janela 1 | −0,241 |

**Nenhum eixo prevê aprender mais.** Em magnitude, são zero para qualquer uso prático. (O −0,241 é regressão à média, artefato aritmético: quem já estava alto tem menos espaço para subir.)

Ganho médio da coorte em 30 dias: **−0,003**.

> **Consequência de produto:** este módulo não pode ser usado como indicador de progresso, nem exibido ao aluno como "seu engajamento". Responde "vai continuar?", e só isso. Está em caixa alta no cabeçalho do arquivo porque o eixo chamado "profundidade" convida exatamente à leitura errada.

---

## 11. Acurácia não é a métrica deste problema

Alvo invertido para "vai sair", que é sobre o que se age.

### OULAD — onde 92,9% de acurácia não vale nada

| limiar | alerta em | acurácia | acc. balanceada | precisão | cobertura |
|---|---|---|---|---|---|
| **0,5 (padrão)** | **0,0%** | **92,9%** | **50,0%** | — | 0,0% |
| *nunca alertar* | *0,0%* | *92,9%* | *50,0%* | — | *0,0%* |
| top 5% de risco | 5,0% | 92,1% | 63,5% | **42,6%** | 30,1% |
| **top 10% de risco** | 10,0% | 90,8% | **74,6%** | 39,7% | 55,8% |
| top 20% de risco | 20,0% | 82,9% | 77,0% | 24,9% | 70,1% |

**Com limiar 0,5 o modelo não alerta ninguém e acerta 92,9% — exatamente a acurácia de não ter modelo.** A acurácia balanceada de 50,0% denuncia. E o AUC é 0,863: ele ordena muito bem, só nunca cruza 0,5.

### EdNet — a base é o oposto

| limiar | alerta em | acurácia | acc. balanceada | precisão | cobertura |
|---|---|---|---|---|---|
| 0,5 (padrão) | 81,2% | 81,5% | 72,7% | 81,6% | 95,0% |
| top 10% de risco | 10,0% | 39,3% | 56,0% | **95,1%** | 13,6% |
| melhor acc. balanceada | 67,0% | 81,0% | **78,7%** | 87,9% | 84,4% |
| *sempre alertar* | *100%* | *69,8%* | *50,0%* | *69,8%* | *100%* |

Aqui 69,8% saem, então "sempre alertar" já dá 69,8% de acurácia. Alertar os 10% de maior risco **derruba** a acurácia para 39,3% — e é o melhor ponto de operação se a intervenção for cara, porque a precisão é 95,1%.

**O que reportar:** AUC (ordena?), acurácia balanceada (distingue dos dois lados?), e precisão e cobertura no ponto de operação escolhido. Por isso `risco()` recebe **taxa de alerta**, não limiar fixo.

---

## 12. As medidas são traço, não ruído

Correlação entre metades aleatórias e independentes dos eventos do mesmo aluno (EdNet, 16.762 alunos):

| frequência | volume | profundidade |
|---|---|---|
| +0,989 | +0,892 | +0,881 |

Todas acima de 0,88. Medem propriedade do aluno.

---

## 13. O que o módulo entrega

| sem calibração | com `calibrar()` (≥300 alunos com desfecho) |
|---|---|
| faixa de recência (0 / 1–3 / 4+ dias) | as três faixas, por percentil da coorte |
| `retencao_esperada = None` | retenção observada na sua coorte |
| `ordenar()` levanta erro | score comum sobre z da coorte |
| — | `risco(taxa_alerta=0.10)` com limiar da coorte |

A recência funciona sem calibração porque seus cortes são **contagem de dias**, não escala. Os demais eixos exigem a régua da própria coorte.

---

## 15. Trajetória: "caiu / estável / subiu"

A pergunta natural depois de medir o nível: a **forma** da atividade importa? Dois alunos com o mesmo esforço, um crescendo e outro despencando, terminam igual?

**Desenho.** A janela de 30 dias dividida em três sub-janelas de 10. Trajetória = inclinação dos dias ativos entre a primeira e a última. O limiar de 0,25 dia por janela é estrito — `[5, 5, 4]` já conta como "caiu".

### Para o modelo, ela não acrescenta

| modelo (AUC fora do treino) | EdNet | OULAD |
|---|---|---|
| só recência | 0,815 | 0,854 |
| recência + frequência | 0,843 | 0,861 |
| recência + tendência | 0,833 | 0,855 |
| **recência + frequência + tendência** | **0,846** | **0,864** |

**+0,003 nas duas bases.** Recência e frequência já contêm a trajetória: saber quanto o aluno fez no total e quanto fez ultimamente **determina** a forma. Não é feature nova, é a mesma informação.

### Mas como explicação ela é forte — se comparada do jeito certo

Fixando o **total de dias ativos** na janela:

| total de dias | caiu | estável | subiu |
|---|---|---|---|
| **EdNet** | | | |
| 3 a 6 | **18,6%** | 43,5% | **60,0%** |
| 7 a 12 | 38,5% | 71,9% | 81,6% |
| 13 a 20 | 59,4% | 83,8% | 89,1% |
| **OULAD** | | | |
| 3 a 6 | **74,3%** | 87,2% | **91,6%** |
| 7 a 12 | 89,9% | 96,4% | 97,7% |
| 13 a 20 | 96,9% | 99,0% | 99,4% |

Monotônico nos **seis estratos, nas duas bases**, com até **41 pontos** de diferença. Mesmo esforço total, formas diferentes, destinos diferentes.

---

## 16. O controle errado inverte o sinal

Este é o achado que mais importa para quem for construir o alerta.

Fixando a **recência** em vez do total, a mesma tabela dá o contrário:

| recência | caiu | estável | subiu |
|---|---|---|---|
| EdNet, 1–3 dias | **55,4%** | **46,8%** | 57,6% |
| OULAD, 1–3 dias | **93,4%** | **91,9%** | 92,3% |

**"Caiu" aparece com retenção maior que "estável"** — o oposto da intuição.

O mecanismo é aritmético. `tendência = (j3 − j1) / 2`. Fixando j3, "caiu" só pode significar **j1 alto**:

| recência 1–3 dias (EdNet) | dias ativos na 1ª janela | total na janela | retenção |
|---|---|---|---|
| caiu | **5,1** | 9,4 | 55,4% |
| estável | 1,6 | 4,4 | 46,8% |
| subiu | 1,2 | 5,0 | 57,6% |

Entre alunos **igualmente ativos agora**, quem "caiu" é quem era **muito mais ativo antes** — e atividade acumulada prediz retenção.

> **Consequência prática:** um alerta de "engajamento caiu" que compare alunos pela atividade recente marca justamente os **mais engajados** da coorte. Para funcionar, a comparação tem de ser entre alunos de **mesmo total de atividade**.

As duas tabelas não se contradizem — respondem a perguntas diferentes. *"Entre alunos igualmente ativos hoje, o passado importa?"* Não, e um pouco ao contrário. *"Entre alunos de mesmo esforço total, a forma importa?"* Enormemente.

---

## 17. Contra o abandono formal (OULAD)

O OULAD registra **desmatrícula**, não só ausência de cliques. Contra esse alvo (3,5% da base):

| medida | AUC |
|---|---|
| recência | 0,618 |
| profundidade | **0,611** |
| frequência | 0,604 |
| tendência | 0,553 |
| instabilidade | 0,424 *(invertida: instável → mais abandono)* |

Tudo **muito mais fraco** que contra atividade (0,863). E a ordem muda: a profundidade quase empata com a recência, enquanto no alvo de atividade ela era a mais fraca das três.

**Parar de clicar e cancelar a matrícula são fenômenos diferentes.** O módulo prediz o primeiro; nada aqui autoriza usá-lo para o segundo.

---

## 18. O que isso muda no módulo

Entra [`trajetoria()`](../../trailup_core/engajamento.py) — **descrição, não predição**. Devolve `'caiu' | 'estavel' | 'subiu'` e está documentada como não-somável ao score, com um teste que verifica que `ordenar()` não a usa.

Serve para dizer ao professor **por que** o aluno está em risco, numa forma sobre a qual ele consegue agir — e o risco em si continua saindo de recência + frequência.

---

## 19. Limites

- **O volume por sessão foi removido.** Ele tinha o sinal mais interessante do estudo (AUC 0,400 no EdNet, efeito sobrevivendo dentro de cada quintil de frequência) e **virou ruído no OULAD** (0,522, IC quase tocando o acaso), com a correlação com frequência invertendo de sinal (−0,30 → +0,23). Duas explicações possíveis e indistinguíveis com este dado: a tradução não é equivalente (respostas por sessão ≠ cliques por dia), ou o efeito é do EdNet. Nos dois casos o eixo sai — um sinal forte que some ao mudar de plataforma é propriedade de uma base, não medida de engajamento.
- **Duas bases não provam generalização.** O volume caiu na segunda; não há garantia de que recência e frequência sobrevivam a uma terceira com contexto muito diferente.
- **A AUC do teste *junto* (0,702) não é métrica útil** e não deve ser citada: misturar populações com prevalências de 30,5% e 92,9%, tendo padronizado *dentro* de cada uma, faz alunos de z alto em bases diferentes receberem o mesmo score com probabilidades reais muito distintas. A AUC que importa é a de dentro de cada coorte.
- **Os cortes e a curva de referência são das bases citadas.** Recalibrar com `calibrar()` quando houver ~2.000 alunos com 60 dias de histórico. Se o ciclo do TrailUp for semanal, mudar `JANELA_DIAS` **antes** de recalibrar — a janela muda o significado de todos os eixos.
- **`risco()` dispara um pouco acima da taxa pedida** (7,9% quando se pede 5%), porque `dias_recentes` é contagem discreta e há empates no limiar.
- **A recência é adjacente à janela do desfecho** (olha os dias 20–29 para prever 30–59). Não há vazamento — é informação disponível na hora da decisão — mas parte do ganho vem de estar mais perto no tempo, não de medir melhor.
- **Falta instrumentação para a profundidade.** Exige registrar tempo por questão, que o TrailUp não coleta.
- **Licença.** EdNet é CC BY-NC. O módulo carrega constantes medidas, não pesos treinados — a decisão sobre uso comercial é de quem publica.
- **Tudo observacional.** "Recência alta prevê retenção" não autoriza concluir que *fazer* o aluno aparecer o faria ficar. Para isso é preciso experimento.

---

- **A trajetória foi medida com limiar estrito** (0,25 dia por sub-janela). Isso deixa "estável" como faixa estreita — no EdNet, 633 alunos contra 2.640 em "caiu" dentro do mesmo estrato de recência. Um limiar mais folgado muda os tamanhos dos grupos e provavelmente atenua os contrastes.
- **Três sub-janelas de 10 dias é uma escolha.** Uma trajetória em escala semanal, ou por sessão, não foi testada.
- **O alvo de desmatrícula só existe no OULAD**, então o achado da §17 não foi replicado.

## Apêndice — o que mudou, e por quê

O estudo teve três rodadas. As duas primeiras publicaram números que as seguintes derrubaram. Ficam registrados aqui porque o que caiu explica o desenho que ficou.

### Rodada 1 — só EdNet

**Desfecho tautológico.** "Voltou depois da metade do próprio período" é verdadeiro para **100% dos alunos** — o ponto médio do período de alguém sempre tem atividade depois. O AUC medido contra um alvo constante não media nada. **Corrigido** pela janela fixa 0–29 / 30–59 da §3.

**Componente circular.** `persistencia = 1 − quits/(por_sessao+1)` está dividido pelo volume por sessão, que era outro componente. Correlacionava 0,94 com ele **por construção algébrica**, não por achado. **Removido.**

### Rodada 2 — replicação no OULAD e no ARES

**O volume caiu.** Era o achado que eu mais defendi — "maratona é sinal de abandono", com o efeito sobrevivendo ao condicionamento por quintil de frequência. Sumiu na segunda base. **Removido do módulo** (§14).

**A recência apareceu.** Ganho de +4,1 e +4,6 pontos de AUC sobre o desenho de três eixos, nas duas bases.

**Uma hipótese minha, errada.** Eu esperava que usar **percentil** em vez de valor absoluto consertasse a transferência entre bases. Testei: AUC praticamente igual (0,866 / 0,844) e **ECE ainda pior** (0,596 / 0,655). Normalizar a *feature* não conserta uma diferença que está na *prevalência do desfecho*.

**AUC e calibração andaram em direções opostas.** A logística linear sobre os novos eixos ganhou AUC e **piorou o ECE** (0,049 contra 0,031 do desenho anterior) — a relação entre dias recentes e retenção não é linear no logito. Tratar a recência como **categoria** recuperou: ECE 0,015. Otimizar só AUC teria piorado o produto, porque o número exibido é o que o professor lê.

### Rodada 3 — treinar nas duas juntas

**Juntar as bases cruas era armadilha**, e o teste de identificabilidade (§8) mostrou por quê: AUC 0,994 para adivinhar a plataforma. Treinar assim piorava o EdNet de 0,849 para 0,788. **Corrigido** pela padronização dentro da coorte antes de juntar.

### Números que mudaram entre versões

| | rodada 1 | final |
|---|---|---|
| eixos | frequência, volume, profundidade | recência, frequência, profundidade |
| AUC | 0,802 (uma base) | **0,846 / 0,868** (duas bases, um modelo, média de 12 partições) |
| ECE | 0,031 | 0,015 / 0,008 |
| cortes | absolutos, do EdNet, embutidos | por coorte, via `calibrar()` |
| sem calibração | três faixas + retenção esperada | só recência; `None` na probabilidade |
