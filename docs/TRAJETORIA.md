# Trajetória — como cada modelo chegou ao estado atual

Um documento por modelo existe em `docs/`. Este aqui é o caminho: o que foi tentado, o que caiu, e por quê. Serve para não repetir tentativa já refutada, e para auditar de onde vem cada número.

**Estado em 2026-09-13:** 9 módulos em uso, 7 hipóteses descartadas, 1 na fila.

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

**Onde parou.** Linear sem dependência, Spearman 0,463 (boosting 0,429 no mesmo conjunto, encoder 0,532). **Só para ordenar**, nunca para nota — os pesos foram ajustados em alemão. Como triagem funciona: as 10 piores previstas têm nota real média 1,87 contra 3,50 do geral.

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

Quatro dos sete só apareceram porque **a segunda base derrubou o achado da primeira**, ou porque um **placebo** foi incluído. Nenhum apareceria com mais ajuste de modelo.
