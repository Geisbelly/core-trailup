# Trajetória — como cada modelo chegou ao estado atual

Um documento por modelo existe em `docs/`. Este aqui é o caminho: o que foi tentado, o que caiu, e por quê. Serve para não repetir tentativa já refutada, e para auditar de onde vem cada número.

**Estado em 2026-09-13:** 10 módulos em uso, 7 hipóteses descartadas, 1 na fila, 136 testes, 5 verificadores automáticos.

Este documento tem duas partes: **como cada modelo chegou ao estado atual** (§1 a §10) e **a auditoria que veio depois** — 97 verificações, 49 defeitos, todos corrigidos.

---

## O fio condutor

Onze experimentos independentes, em quatro datasets, chegaram à mesma conclusão:

> **O sinal está no conteúdo e no acumulado. Não está no estado momentâneo do aluno.**

| experimento | o que ganhou | o que perdeu |
|---|---|---|
| domínio (M2) | a questão (0,703) | histórico comportamental (0,641) |
| tempo esperado | a questão (R² 0,578) | o aluno (R² −0,037) |
| dificuldade | a questão | agrupar alunos (ARI 0,18) |
| avaliador discursivo | cobertura do gabarito | relação entre conceitos (+0,006) |
| engajamento | recência acumulada | volume por sessão (não replica) |
| empenho | — | as três operacionalizações |
| estado do aluno (M1) | — | **tudo** (AUC 0,505) |

Esse padrão não foi hipótese inicial. Foi o que sobrou.

---

## 1. M1 — estado de estudo

**Hipótese.** Substituir os seis "analisadores" de regra da `linear_analysis_pipeline.py` por um modelo que infere *fluindo / travado / disperso* a partir de telemetria.

**Caminho.**

1. Primeiro corte quebrava a leitura a cada modal de palavra → "episódios" de 25 s que não são leitura. Refeito.
2. Nota de estrela não carrega `assignmentId` → atribuída por posição temporal, consolidada pelo último valor.
3. Modelo comportamental: **AUC 0,505**, IC [0,474 – 0,535]. Três famílias, três alvos, nenhum passa de 0,53.
4. O que prediz: base pessoal do aluno (0,784) + dificuldade do texto (0,581).
5. **Reviravolta.** A base pessoal tem **AUC intra-aluno de 0,379** — abaixo do acaso. Ela separa quem reclama de quem não reclama; não diz qual leitura foi difícil para um aluno. O que sobrevive por dentro é a dificuldade do texto (0,654).

**Onde parou.** M1 não existe como estava desenhado. O slot vira duas quantidades declaradas: `base_do_aluno` (para calibrar a escala do check-in) e `dificuldade_do_conteudo`. O check-in sai da Fase 3 e vira o mecanismo — satura em 3 a 5 respostas.

**Consequência de LGPD.** A *categoria* de sinal que frames de câmera e `touch_samples` representam não prediz estado. Coletar dado sensível de menor para alimentar inferência que não funciona não tem defesa técnica nem jurídica.

---

## 2. M2 — domínio por tópico

**Hipótese.** A regra em produção (`base ± 0,08·acertos − 0,07·erros`, confiança fixa 0,66) é ruim e um modelo tabular a supera.

**Caminho.**

1. Regra medida: AUC 0,587, **ECE 0,201**, e **0,486 nas 5 primeiras respostas** — IC inteiro abaixo de 0,50, ou seja anti-informativa justo onde o TrailUp opera. Satura: o ponto de equilíbrio é 46,7% de acerto.
2. Boosting com 25 features: 0,753.
3. **Erro de alvo.** Eu usava `1 − p(acerta a próxima)` como sinal do gate. Medido contra o alvo real ("vai travar no tópico"), o proxy dá 0,698 **contra 0,761 da regra que ia substituir**. Teria ido a produção piorando o gate. Viraram dois modelos.
4. **Rótulo contaminado.** 21,7% das linhas eram tentativas intermediárias contadas como resposta final. Acurácia medida 0,577, real 0,668. A contaminação **inflava o comportamento**, porque tentativa intermediária *é* comportamento. Corrigido, a leitura inverte: só-comportamento cai de 0,683 para 0,641, e a questão sozinha sobe de 0,667 para 0,706.
5. **Vazamento por aspas de shell.** `for a in "proxima base"` fazia `ALVO="proxima base"`, o guard nunca disparava, e latência da própria questão entrava para prever aquela resposta. AUC 0,764, contaminado. Blindado com lista `POSTERIOR` e `assert`.
6. Três apostas de feature renderam quase nada: 189 tags de habilidade +0,001, latência e hesitação +0,003, janela recente +0,007.

**Onde parou.** Fórmula fechada `0,7 × questão + 0,3 × aluno`, AUC 0,721 contra 0,586 da regra. Uma linha de aritmética captura 95% do ganho do boosting.

**Tentativa de melhoria (2026-09-13).** Primeira rodada: peso ótimo, encolhimento e logito não rendem nada (melhor 0,722, que já era o que o módulo fazia; logito é pior, 0,715).

**Segunda rodada, com um terceiro termo.** O acerto do aluno em **todos** os tópicos, e não só naquele:

| fórmula | AUC |
|---|---|
| 0,70 questão + 0,30 tópico | 0,722 |
| 0,70 questão + 0,30 global | 0,721 |
| **0,60 questão + 0,15 tópico + 0,25 global** | **0,727** |

Testadas 14 combinações. O global sozinho não ajuda; junto com o tópico, sim. E **o peso ótimo do global (0,25) é maior que o do tópico (0,15)** — o acerto geral é mais estável que o acerto naquele tópico, que quase sempre tem poucas observações.

O módulo mantém **dois conjuntos de pesos**, porque os ótimos diferem: sem histórico global, 0,70/0,30; com ele, 0,60/0,15/0,25.

**E o gap para o boosting não é de não-linearidade.** Testado exportando a interação entre as quatro features como **tabela de consulta** (1.646 células):

| | AUC |
|---|---|
| só o acumulado, linear | 0,746 |
| tabela de 1.646 células | **0,746** |
| tabela + linear | 0,748 |
| boosting com 30 features | **0,779** |

A tabela captura toda interação possível entre essas quatro variáveis e **não ganha nada**. O que separa 0,748 de 0,779 são as outras 26 features — não a forma da função. Fechar o gap exige instrumentação nova, não aritmética mais esperta.

É a **quinta aposta de engenharia de feature que não paga**, depois das 189 tags (+0,001), latência e hesitação (+0,003), janela recente (+0,007) e tipo de recurso na evasão (+0,003).

**Lacuna fechada em 2026-09-13: o gate não estava no módulo.** Era o modelo mais relevante para o produto — a decisão de quando abrir a LLM — e só existia como boosting. Medido em 1.721.188 pontos de decisão (taxa base 12,2%):

| critério | AUC | prec@5% | prec@10% | lift@10% |
|---|---|---|---|---|
| regra do TrailUp | 0,687 | **40,8%** | 32,9% | 2,7× |
| acerto acumulado no tópico | 0,720 | 36,3% | 31,0% | 2,5× |
| forma fechada (3 termos) | 0,725 | 37,0% | 33,0% | 2,7× |
| **fechada + regra** | **0,732** | 40,0% | **33,9%** | **2,8×** |
| boosting com 30 features | 0,779 | — | 41,7% | 3,4× |

**A forma fechada sozinha tem AUC melhor que a regra e perde para ela no ponto de 5%** (37,0% contra 40,8%) — melhoria de ordenação global que não se traduz onde a decisão é tomada.

A razão está na saturação. A regra prende 27,7% das estimativas no teto, o que a torna péssima como estimativa. Mas no extremo inferior a saturação **vira informação**: os 0,9% de alunos presos no piso têm **57,3%** de taxa de alvo contra 12,2% da base. A forma suave não marca esses casos.

Daí o módulo combinar as duas, meio a meio. E daí a ressalva honesta: o boosting continua 0,047 de AUC acima, e 8 pontos de precisão acima no ponto de 10%.

---

## 3. Dificuldade da questão

**Hipótese inicial (errada).** Classificar questões em categorias de dificuldade.

**Caminho.**

1. Categorias fixas → substituídas por **faixa de valores**, a pedido: cravar corte inventa fronteira que o dado não tem.
2. **Prior escolhido a olho** (força 15) quebrava a cobertura: o intervalo de 90% cobria 64,5%. `derivar_prior()` por momentos dá Beta(5,07 / 2,08), força **7,0**.
3. **Validação no objeto errado.** A tabela publicada media a cobertura de um intervalo **preditivo** (sobre a contagem futura), mas `estimar()` devolve um **posterior** (sobre a taxa latente). Objetos diferentes.
4. **Reencontro quebra independência.** 11,6% das respostas do EdNet são reencontro; passar `alunos` corrige de 88,9% para 90,1% em n=100.
5. `k=4` imposto às questões → HDBSCAN, DP-GMM e Mean Shift encontram **2 grupos**, separados pelo **tempo** (d de Cohen 3,30), e o eixo do acerto é contínuo. Daí a regra de desenho: **categoria no que é discreto, intervalo no que é contínuo**.

**Revisão de 2026-09-13.** O relatório afirmava que a cobertura degradava em n alto (77,6% em n=200). **Estava errado, e o erro era meu de novo:** o alvo da validação (taxa numa amostra de validação) tem ruído próprio comparável à largura do intervalo. Contabilizando esse ruído, a cobertura é **88 a 91% de n=50 a n=400+** — estável. O estimador estava certo.

**O defeito real, esse sim.** O intervalo de `estimar()` responde *"qual é a dificuldade desta questão"*. A pergunta do professor é *"quanto a minha turma de 30 vai acertar"* — e para essa, o intervalo cobre **39,7%** das turmas.

| turma | `estimar()` | `prever_turma()` |
|---|---|---|
| 15 alunos | 30,7% | **90,1%** |
| 30 alunos | 39,7% | **88,8%** |
| 60 alunos | 49,7% | **88,2%** |

Entra `prever_turma()`. A largura triplica (0,097 → 0,279 numa turma de 30) porque a incerteza é real.

**Variação entre coortes reais** (alunos antigos vs recentes no EdNet): σ = 0,0153, correlação 0,953 entre coortes. Pequena, mas não encolhe com n — entra no preditivo.

---

## 4. Dificuldade por turma (o painel do professor)

**Caminho.**

1. Testar numa turma nova não responde nada — não há dificuldade-de-turma para consultar; tudo cai para o global. O teste tem de caminhar **dentro da questão, ao longo dos alunos**.
2. A taxa crua da turma é **pior que o global** abaixo de 10 alunos (0,657 contra 0,688). O estimador hierárquico nunca perde.
3. **A referência global precisa ser *leave-one-out*.** Incluindo a própria turma, `|z| ≥ 2` dispara em 3% dos casos em vez de 15% — a funcionalidade parece inútil numa turma de 30.
4. Percentil é bom para ler, ruim para calcular: como feature dá AUC 0,744 contra 0,792 da taxa absoluta.

**Onde parou.** Encolhimento hierárquico sempre; exibir o desvio da turma só quando `|z| ≥ 2` — numa turma de 30, aparece em ~15% das questões com 94% de acerto na direção, contra 72% de "exibir sempre".

---

## 5. Engajamento

**Caminho.**

1. Primeira rodada, só EdNet, quatro componentes. **Dois defeitos graves:** `persistencia` era circular (dividida pelo próprio volume, r = 0,94 por álgebra), e o desfecho "voltou depois da metade do próprio período" dava positivo para **100% dos alunos**.
2. Refeito com janela fixa 0–29 / 30–59. Três eixos: frequência, volume, profundidade. AUC 0,802.
3. **Replicação no OULAD derruba o volume.** Era o achado mais interessante — "maratona é sinal de abandono", sobrevivendo ao condicionamento por quintil. Vira 0,522 na segunda base, e a correlação com frequência inverte de sinal. Removido.
4. **ARES não tem o fenômeno:** leitura obrigatória, 87,7% voltam, AUC 0,49–0,57. Vira escopo declarado — o módulo pressupõe continuação **voluntária**.
5. **Recência aparece** (dias ativos nos últimos 10): 0,810 / 0,863, melhor que frequência nas duas bases. Recência + frequência = 0,843 / 0,861.
6. **Hipótese minha, errada:** percentil consertaria a transferência entre bases. Não conserta — AUC igual, ECE pior (0,596 / 0,655). Normalizar a feature não corrige diferença de *prevalência do desfecho*.
7. **AUC e calibração andaram em direções opostas.** A logística linear ganhou AUC e piorou o ECE (0,049 contra 0,031). Recência como **categoria** recupera: ECE 0,015.
8. **Juntar as bases cruas era armadilha:** prever de qual base a linha veio dá **AUC 0,994**, e treinar assim piora o EdNet de 0,849 para 0,788. Padronizando dentro da coorte, cai para 0,508 e um modelo único empata com os específicos.
9. **Trajetória:** não acrescenta ao modelo (+0,003), mas explica bem — fixando o total de atividade, a retenção vai de 18,6% ("caiu") a 60,0% ("subiu") no mesmo esforço total. **Fixando a recência em vez do total, o sinal inverte** e o alerta marcaria os mais engajados.

**Onde parou.** Recência + frequência, ordenador comum às duas bases, calibração por coorte obrigatória, `trajetoria()` como descrição.

---

## 6. Empenho — três tentativas, três nulos

**Hipótese.** O engajamento prediz permanência mas **não prediz aprendizado** (±0,04). Se o esforço predisser, mede algo que nenhum módulo mede.

| tentativa | melhor AUC | por que caiu |
|---|---|---|
| nível de esforço (resíduo de tempo) | 0,516 | a dificuldade da questão prediz mais (0,584) |
| comportamento **após o erro** | 0,524 | **o placebo prediz mais** (0,527) |
| **trajetória** do próprio aluno | 0,504 | nulo com medida confiável |

A trajetória era o desenho certo — cada aluno é seu próprio controle, eliminando o confundimento que derrubou a base pessoal do M1. E o nulo **não é de medida**: confiabilidade 0,705 entre metades, teto de 0,839.

**Sobra:** a diferença entre **ler e não ler** a explicação (59,2% contra 75,3%), sem dose-resposta, provavelmente confundida com sessão abandonada.

---

## 7. Avaliador discursivo

**Caminho.** Quatro tentativas: similaridade TF-IDF → grafo de co-ocorrência (+0,10) → enriquecer a referência com conteúdo do domínio (**piora**: 0,486 → 0,396) → encoder multilíngue (+0,057).

**Teto corrigido.** O "0,899" era a concordância de uma execução do LLM com a **média das três, que a contém**. Entre execuções independentes: **≈0,88**.

**Melhoria de 2026-09-13.** Ampliando de 8 para 10 features: Spearman 0,451 → **0,463**, QWK 0,318 → **0,350**.

Todo o ganho vem de `n_nos` e `n_arestas` — **contagem pura** do tamanho do grafo da resposta (+0,014 e +0,015 cada). E a **camada semântica não acrescenta nada**: 13 features sem LSA dão 0,459; as 15 com ela, 0,458.

Escolhido pela **cauda baixa**, que é o uso declarado: as 20 piores previstas passam de nota real 2,37 para 2,03. A cauda alta piora um pouco (as 10 melhores, 4,37 → 4,13).

> **Um vazamento evitado.** A primeira versão do teste pegou todas as colunas numéricas do arquivo — o que incluía `Run1_AI Evaluation`, `Run2_…` e `Style_Mean`, que são a avaliação do próprio LLM. Seria vazamento direto do alvo. O script agora tem lista explícita de features e um `assert` contra as colunas proibidas.

**Onde parou.** Linear sem dependência, Spearman **0,501** ponta a ponta (era 0,463 nas features do pipeline; boosting 0,429 no mesmo conjunto, encoder 0,532). **Só para ordenar**, nunca para nota — os pesos foram ajustados em alemão. Como triagem funciona: as 10 piores previstas têm nota real média 1,87 contra 3,50 do geral.

**Bug que quase passou:** `stopwords()` pegava as 60 palavras mais frequentes de dois textos curtos, o que zerava o grafo de referência e fazia a resposta puramente divagante tirar 5,0. Corrigido para frequência de documento com guarda de 10 textos.

---

## 8. Traços e grupos

**Resultado.** Alunos são **altamente mensuráveis e nada categóricos**: oito traços replicam entre 0,70 e 0,98, mas os grupos não replicam (ARI 0,18) e a silhueta do HDBSCAN é idêntica à do k-means com k=2 — os dois bissectam uma nuvem contínua.

**Como é fácil fabricar grupos falsos.** Trocando a transformação por quantis por `StandardScaler`, o resultado vira ARI 0,917 e silhueta 0,821 — grupos "sólidos". São a cauda de outliers separada do resto: d = 3,17 num eixo e **0,09 em outro**.

**Bug corrigido.** "Zero questões com discriminação negativa" era comparação contra −1 em vez de 0. Real: 2,96% numa metade dos alunos, **0,53% confirmadas nas duas**.

**Melhoria de 2026-09-13 — como marcar.** Com três partições de alunos (duas para marcar, uma nunca vista para conferir):

| regra | marca | precisão | lift |
|---|---|---|---|
| negativa numa partição | 239 | 13,8% | 3,9× |
| negativa em p0 **ou** p1 | 549 | 12,2% | 3,5× |
| **média das duas < 0** | 132 | **25,0%** | **7,1×** |
| negativa nas **duas** | 39 | 28,2% | 8,0× |
| média das duas < −0,05 | 27 | 29,6% | 8,4× |

Dividir os alunos em duas metades e exigir que a **média** seja negativa quase **dobra a precisão** marcando metade das questões. Entra `confirmar()`.

A razão de funcionar: a discriminação correlaciona apenas **0,33** entre partições independentes. Uma medida isolada é, em boa parte, ruído amostral.

---

## 8b. Revisão — validada como preditor

A curva de esquecimento foi medida em 594 mil reencontros desde o início, mas **nunca tinha sido testada como preditor** — só o formato havia sido descrito.

| preditor | AUC | ECE |
|---|---|---|
| **curva de esquecimento** | **0,665** | **0,021** |
| só "acertou antes" | 0,586 | — |
| só a dificuldade da questão | 0,618 | — |
| curva + dificuldade da questão | 0,661 | 0,077 |

Bate os dois componentes isolados, e **somar a dificuldade da questão piora** — tanto a ordenação quanto a calibração. A curva já carrega o que precisa.

**Melhoria de 2026-09-13 — a proporção, não o último resultado.** Em vez de escolher entre as duas tabelas pelo último acerto, interpolar entre elas pela **fração dos encontros anteriores em que o aluno acertou**:

| variante | AUC | ECE |
|---|---|---|
| só o último resultado (binário) | 0,665 | 0,021 |
| **interpolado pela proporção** | **0,669** | **0,015** |

Melhora ordenação **e** calibração sem tabela nova. Só faz diferença a partir do terceiro encontro.

**E o efeito de espaçamento não aparece como a literatura prevê.** Média de acerto de quem acertou no encontro anterior, por número do reencontro:

| intervalo | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| <1h | 0,901 | 0,939 | 0,943 | 0,937 |
| 7d | 0,859 | 0,812 | 0,865 | 0,897 |
| **60d** | **0,819** | **0,745** | **0,749** | **0,725** |
| **60d+** | **0,831** | **0,733** | **0,739** | **0,708** |

Em intervalo curto a curva achata, como esperado. Em intervalo longo **a direção inverte**. A explicação provável é **seleção, não memória**: questão reencontrada pela quarta vez depois de 60 dias é questão que o aluno vem errando, ou que o sistema insiste em trazer de volta.

Por isso o número do reencontro **não entra** no módulo: renderia +0,003 e carregaria um viés de seleção que não sei separar.

---

## 8c. Chute — o limiar, medido em toda a faixa

O limiar de 30% da mediana era escolha não justificada. Validação em 127.554 casos (errou na 1ª vez, reencontrou):

| limiar | marca | acerta ao reencontrar | não-marcados | diferença |
|---|---|---|---|---|
| 0,15 | 0,6% | 58,7% | 72,3% | **−13,6 pts** |
| 0,20 | 0,8% | 60,8% | 72,3% | −11,5 pts |
| **0,30** | 1,9% | 64,7% | 72,3% | −7,6 pts |
| 0,40 | 3,7% | 66,4% | 72,4% | −6,0 pts |
| 0,60 | 10,8% | 69,1% | 72,6% | −3,5 pts |

A diferença encolhe **monotonicamente** conforme o limiar afrouxa — o construto é real e graduado. 0,30 fica como padrão; para sinalizar só o caso gritante, `fracao=0,15` dobra a separação marcando um terço.

---

## 9. Evasão

AUC 0,748 com split **por coorte** (treina em 2013B/2013J/2014B, testa em 2014J) — mais severo que o split aleatório da literatura. **Não bateu a meta** que eu mesmo pus (0,80 na semana 4; obtive 0,746).

A semana 2 dá 0,638 — o alerta não funciona no início do curso, que é quando mais se quereria. Alertando os 10%: precisão 12,3% contra base 3,6%, **lift 3,4×**. Sem gap de cobertura por `disability`.

**Minha explicação para o resultado fraco estava errada.** Eu escrevi que a causa provável era *"conjunto de features magro — cliques agregados, sem quebra por tipo de recurso"*. Testado (2026-09-13), quebrando os cliques em conteúdo / social / prova / navegação / apoio, mais a diversidade de recursos tocados:

| conjunto | AUC | AP | lift@10% |
|---|---|---|---|
| magro (cliques agregados) | 0,763 | 0,100 | 3,6× |
| **rico (+ tipo de recurso)** | **0,766** | 0,100 | 3,6× |

**+0,003.** A quebra por tipo de recurso não explica o teto. É a quarta aposta de engenharia de feature que não paga, depois das 189 tags de habilidade (+0,001), da latência e hesitação (+0,003) e da janela recente (+0,007).

**Armadilha de métrica registrada.** O alvo "evade em algum momento" tem AP 0,279 contra 0,106 — parece melhor, mas a taxa base é 15,1% contra 3,6%. **AP não compara entre alvos de prevalência diferente.** O AUC mostra o contrário: 0,701 contra 0,748.

---

## 10. Tempo esperado

| variante | R² |
|---|---|
| média global | −0,045 |
| mediana bruta da questão | 0,574 |
| **média do log da questão** | **0,578** |
| mediana encolhida para a global (k=5) | 0,575 |
| boosting com 14 features | 0,651 |

**O aluno não prediz o próprio tempo.** Sob split por aluno a mediana dele sequer existe — o preditor cai para o global, R² −0,037. Ajustar pelo ritmo do aluno **piora** (0,571 → 0,303 com expoente 1,0).

Entra `esperado_de_amostra()` com média do log, +0,004. É menos robusto a outlier que a mediana — aplicar corte antes.

---

## A auditoria de 2026-09-13

Depois das melhorias, veio uma auditoria em **13 rodadas**. Cada número dos cabeçalhos foi remedido **chamando as funções do módulo**; depois vieram fuzzing, monotonicidade, cobertura de API e consistência entre documentos.

**97 verificações. 49 defeitos.** Nenhum deles apareceria olhando AUC.

### As rodadas, e o que cada uma atacou

| rodada | ataque | defeitos |
|---|---|---|
| 1–3 | os números dos cabeçalhos, remedidos chamando o código | 3 |
| 4 | `dominio`: probabilidade e confiança nunca calibradas | 2 |
| 5 | limiares que dependem da escala recalibrada | 1 |
| 6 | somas e agregações (`duracao_prevista`) | 1 |
| 7 | promessas implícitas nos nomes (`prioridade_revisao`) | 1 |
| 8 | constantes estabelecidas sem base | 2 |
| 9 | **as minhas próprias afirmações da auditoria** | 4 |
| 10 | interações entre módulos e bordas | 5 |
| 11 | fuzzing com entrada hostil | 21 |
| 12 | faixas do SQL e monotonicidade no domínio inteiro | 3 |
| 13 | convenção de sinal do pacote | 2 |

### Os quatro padrões

**1. Unidade é promessa.** As saídas que só **ordenam** passaram sem defeito — `discriminacao`, `trajetoria`, `perfil_chute`, `cobertura`. Os defeitos estão todos em saídas com **unidade**: confiança, probabilidade, prazo em dias, soma em segundos, taxa de disparo, porcentagem. Um score que ordena não pode mentir sobre magnitude; um número com unidade mente por omissão se ninguém mediu a magnitude.

| onde | afirmava | era |
|---|---|---|
| `ritmo._CONF` | confiança 1,00 em n=21 | teto real 0,997 |
| `evasao.sql` | probabilidade de evasão | **11× maior que a real** |
| `dominio.p` | probabilidade | ECE 0,056, comprimida |
| `dominio.confianca` | cresce com n | o erro real é plano |
| `duracao_prevista` | soma de medianas | subestima 13% |
| `dias_ate_revisar` | retém X% no dia N | entrega X−4 pontos |
| `pre_avaliacao.divagacao` | % de divagação | mediana é 0,72 |

**2. Consertar uma escala deixa órfão quem corta nela.** Aconteceu **três vezes**: `precisa_reforco` (2,3% → 12,2% de disparo), o limiar de `tendencia` (passaria a disparar com metade da evidência) e as faixas do `evasao.sql` (que ficaram pegando **0,27%** dos casos — precisão alta, cobertura nula).

**3. Corrigir um sintoma sem varrer os vizinhos.** O parâmetro com nome invertido apareceu no `gate`; eu corrigi e **não olhei o `dominio`**, que tinha o mesmo defeito no parâmetro principal. Passar 0,9 como "questão difícil" devolvia `p = 0,890`. A raiz era uma convenção do pacote inteiro — dificuldade é sempre **taxa de acerto** — agora declarada no `__init__.py`.

**4b. Usar um teste que não testa.** Rejeitei uma melhoria real do `pre_avaliacao` porque os intervalos de confiança se sobrepunham — o que não é um teste para medidas correlacionadas. A diferença **pareada** era +0,019, positiva em 100% das reamostras. Descartei sinal achando que estava sendo rigoroso.

**4. Afirmar cobertura sem enumerar.** Disse três vezes que "está tudo auditado". Nas três, escrever um script que checasse me desmentiu: o mapa de cobertura achou **16 símbolos sem teste**, incluindo `chute.foi_chute`, a função principal daquele módulo.

### Os defeitos que só o fuzzing acharia

**21 propagações de NaN e infinito.** A pior: `discriminacao` devolve NaN como sentinela de "não deu para medir" — inclusive quando **todo mundo acertou**, que é comum em dado real. E `NaN < limiar` é `False`, então `confirmar()` classificava a questão não-mensurável como **"não suspeita"**, em silêncio.

### O que a auditoria confirmou

Oito medidas passaram sem ajuste: `perfil_chute` (p90 0,0196 contra 0,020 declarado), a tabela `REFERENCIA` do engajamento (diferença 0,000), o limite de 3× do `demorando`, `gate.MIN_RESPOSTAS = 5` (é o cotovelo exato), a aproximação normal do `dificuldade` (cobertura 75,3% contra 74,8% do Beta), `cobertura` e `conceitos_faltando` do `pre_avaliacao`, e `ritmo.FRONTEIRA_SEG = 40 s` — que é praticamente o corte ótimo por Otsu (**39,8 s**), com d de Cohen **maior** que o documentado (4,18 contra 3,30).

### Uma melhoria que teria piorado o sistema

`JANELA_RECENTE = 10` parecia sub-ótimo: o eixo isolado vai de 0,810 para **0,824** com 14 dias. Recomputei tudo para trocar — e o **modelo compartilhado desabou** de 0,856 para 0,812 (ambos na mesma partição, então a comparação vale), com o peso da frequência ficando **negativo**. Otimizar a parte degradava o todo em 0,044. Mantido em 10, agora com base medida.

### Os cinco verificadores permanentes

Rodam **sem dataset** e falham se alguém quebrar o contrato:

| script | pega |
|---|---|
| `70_consistencia.py` | o mesmo número citado com valores diferentes (10 métricas) |
| `79_cobertura.py` | símbolo público sem teste |
| `80_fuzz.py` | NaN ou infinito vazando |
| `82_monotonia.py` | direção quebrada ao longo do domínio |
| `pytest` | 136 invariantes |

Detalhes em [`auditoria/AUDITORIA.md`](auditoria/AUDITORIA.md).


## Os erros, agrupados por tipo

Sete afirmações publicadas estavam erradas. O padrão importa mais que a lista:

| tipo | casos |
|---|---|
| **validar um objeto e publicar outro** | cobertura preditiva vs posterior; teto do LLM contra a média que o contém |
| **bug de código** | contagem contra −1; conclusão obsoleta impressa pelo script de evasão |
| **circularidade por construção** | `persistencia` dividida pelo próprio volume; roteador de intenção lendo o template da ferramenta |
| **alvo errado** | desfecho verdadeiro para 100%; proxy do gate pior que a regra |
| **contaminação de dado** | 21,7% do rótulo; vazamento por aspas de shell |
| **controle ausente** | base pessoal sem teste intra-aluno; trajetória sem fixar o total |
| **escala não medida** | confiança, probabilidade, prazo, soma, taxa, porcentagem — 7 casos |
| **escala consertada, limiar órfão** | `precisa_reforco`, `tendencia`, faixas do `evasao.sql` — 3 casos |
| **nome que inverte o sentido** | `gate.risco` e `dominio.dominio` recebendo facilidade como "dificuldade" |
| **entrada hostil** | 21 propagações de NaN e infinito |

Quatro dos sete só apareceram porque **a segunda base derrubou o achado da primeira**, ou porque um **placebo** foi incluído. Nenhum apareceria com mais ajuste de modelo.

E os 49 da auditoria só apareceram porque alguém **chamou o código em vez de ler a documentação**, e depois o atacou com entrada hostil, varreu o domínio inteiro e enumerou a API símbolo por símbolo. Há um verificador de consistência permanente em [`auditoria/scripts/70_consistencia.py`](auditoria/scripts/70_consistencia.py) que roda sem dataset e compara os números entre módulo e documentos — foi assim que três divergências restantes apareceram depois da auditoria.
