# Engajamento — eixos separados, validados em duas realidades

**Data:** 2026-09-13 · **Bases:** EdNet KT3, OULAD, ARES · **Módulo:** [`engajamento.py`](../../trailup_core/engajamento.py)

> **Segunda rodada — o que mudou.** A primeira versão deste estudo mediu tudo só no EdNet e entregou três eixos: frequência, volume e profundidade. Replicando no OULAD (universitário britânico, EAD) e no ARES (leitura obrigatória em L2), **um dos três não sobreviveu** e um eixo novo apareceu e é melhor que todos. O desenho entregue mudou:
>
> | | antes | agora |
> |---|---|---|
> | eixos | frequência, volume, profundidade | **recência**, frequência, profundidade |
> | AUC | 0,802 (EdNet) | **0,843** (EdNet) / **0,861** (OULAD) |
> | calibração | ECE 0,031 | **ECE 0,015 / 0,008** |
> | cortes | absolutos, do EdNet | por coorte, via `calibrar()` |
>
> A §15 e a §16 são novas e carregam o argumento. As seções sobre o EdNet seguem válidas, com a ressalva do volume (§16).

---

## 1. Pergunta de negócio

O TrailUp quer saber **quem está prestes a abandonar**, para agir antes. Hoje não há medida disso: a API infere "estado emocional" a cada lote de telemetria, o que já foi testado e não funciona (M1, AUC 0,53).

A pergunta que este estudo responde é mais estreita e mais respondível:

> **Olhando só o que o aluno fez no primeiro mês, dá para dizer quem ainda vai estar estudando no mês seguinte?**

Não é a mesma pergunta que "quem vai aprender mais". A §8 mostra que são perguntas diferentes e que este modelo responde só a primeira.

---

## 2. Dados

**Fonte:** EdNet KT3 — registro de interação de uma plataforma coreana de preparação para o TOEIC. Adultos, estudo autodirigido, sem turma e sem professor. Licença **CC BY-NC** (não comercial).

**Unidade de análise:** o aluno. Uma linha por aluno, não por resposta.

**População final: 18.119 alunos.** Do total do EdNet, restaram esses após três filtros, nesta ordem:

| filtro | por quê | sobraram |
|---|---|---|
| aluno teve ≥60 dias de histórico disponível na base | sem isso não dá para saber se ele voltou ou se a base simplesmente acabou (censura à direita) | 25.845 |
| ≥30 respostas nos primeiros 30 dias | abaixo disso as medidas são ruído amostral | 18.119 |

O segundo filtro é a razão de `MIN_RESPOSTAS = 30` no módulo: quem respondeu menos que isso não recebe faixa, recebe `confiavel=False`.

---

## 3. Desenho: como se evita medir o próprio umbigo

Este é o ponto onde a primeira versão deste estudo errou, então vale explicitar.

Para cada aluno, o dia 0 é o **dia da sua primeira resposta** (não uma data de calendário — alunos entram em momentos diferentes).

```
    dia 0                        dia 30                       dia 60
      |───── JANELA DE MEDIDA ─────|───── JANELA DE DESFECHO ────|
      |   calcula os três eixos    |   observa se voltou ou não  |
      |   (só olha para cá)        |   (nunca entra nos eixos)   |
```

**Nada da segunda janela entra no cálculo dos eixos.** É isso que torna o resultado uma previsão e não uma descrição.

### O que a versão anterior fazia de errado

Ela definia o desfecho como *"o aluno voltou depois da metade do próprio período de atividade?"*. O ponto médio do período de alguém **sempre** tem atividade depois dele — é a definição de ponto médio. O desfecho dava positivo para **100% dos alunos**. Um modelo pode ter qualquer AUC contra um alvo constante; aquele número não media nada.

Um segundo erro, no mesmo lugar: havia um quarto componente, `persistencia`, definido como `1 − quits/(por_sessao+1)`. Está dividido pelo volume por sessão, que era outro componente. Correlacionava 0,94 com ele **por construção algébrica**, não por achado. Foi removido.

---

## 4. As três medidas: definição operacional

Cada eixo é uma conta sobre o que a plataforma já registra. Nenhum é inferido, estimado ou modelado — são contagens.

### Eixo 1 — frequência

> **Em quantos dias diferentes o aluno apareceu, dentro dos 30.**

```
frequencia = dias_distintos_com_ao_menos_uma_resposta / 30
```

Varia de 0 a 1. Vale 0,10 se apareceu em 3 dias; 0,50 se apareceu em 15.

**O que captura:** hábito. Não quanto estudou — com que regularidade voltou.

| | dias ativos em 30 |
|---|---|
| 5% dos alunos ficam abaixo de | 1 dia |
| 25% abaixo de | 2 dias |
| **mediana** | **4 dias** |
| 75% abaixo de | 9 dias |
| 95% abaixo de | 20 dias |

A distribuição é muito assimétrica: **metade dos alunos aparece em 4 dias ou menos de 30.** É uma plataforma de estudo esporádico.

### Eixo 2 — volume por sessão

> **Quantas respostas o aluno dá de uma vez, quando senta para estudar.**

```
sessao = bloco de atividade separado do seguinte por mais de 30 min de inatividade
volume = total_de_respostas / numero_de_sessoes
```

O corte de 30 minutos é uma convenção (é o padrão em analytics de aprendizagem), não uma medida — está exposto como `GAP_SESSAO_SEG` no módulo para ser mudado.

**O que captura:** intensidade da sessão. Note que é uma **média por sessão**, não um total — quem faz 300 respostas em 30 sessões tem volume 10, e quem faz 300 em 3 sessões tem volume 100.

| | respostas por sessão |
|---|---|
| p5 | 6,8 |
| p25 | 11,2 |
| **mediana** | **15,6** |
| p75 | 22,7 |
| p95 | 42,0 |

### Eixo 3 — profundidade

> **Quanto tempo o aluno passa lendo a explicação depois de errar.**

```
profundidade = log(1 + mediana_dos_segundos_na_tela_de_explicacao)
```

Mediana e não média porque a distribuição tem cauda longa — o aluno que deixou a tela aberta e foi almoçar não pode dominar a medida. O `log` comprime a cauda restante; é transformação de escala, não muda a ordem dos alunos.

**O que captura:** se o aluno trata o erro como informação ou passa batido.

| | segundos na explicação (mediana do aluno) |
|---|---|
| p5 | 4,3 s |
| p25 | 6,8 s |
| **mediana** | **10,8 s** |
| p75 | 20,6 s |
| p95 | 45,4 s |

---

## 5. O desfecho: o que "voltou" quer dizer

```
ret = 1  se o aluno deu ao menos uma resposta entre o dia 30 e o dia 59
ret = 0  caso contrário
```

Binário, sem meio-termo. **5.523 dos 18.119 alunos voltaram — 30,5%.**

Esses 30,5% são a **taxa base**, e é contra ela que todo número deste relatório tem que ser lido. Um método que chutasse "todo mundo abandona" acertaria 69,5% das vezes e seria inútil. Por isso a métrica não é acurácia.

---

## 6. A métrica: o que é AUC, em português

**AUC** (área sob a curva ROC) responde a uma pergunta de comparação aos pares:

> Pego **um aluno que voltou** e **um que não voltou**, ao acaso. Qual a chance de que a medida dê um valor mais alto para quem voltou?

- **AUC = 0,50** → a medida não distingue nada. É cara ou coroa.
- **AUC = 0,80** → em 80% dos pares, acerta quem é quem.
- **AUC = 1,00** → separação perfeita (não existe em dado real de comportamento).
- **AUC abaixo de 0,50** → a medida distingue, **mas ao contrário**: valor alto indica quem *não* voltou.

A vantagem do AUC aqui é que ele **não depende da taxa base** nem de escolher um ponto de corte. Ele mede ordenação pura.

### Resultado

Cada eixo sozinho, contra o desfecho, com intervalo de confiança de 95% por bootstrap (400 reamostragens de alunos):

| medida | AUC | IC 95% | leitura |
|---|---|---|---|
| **frequência** | **0,799** | [0,792 – 0,805] | forte, e no sentido esperado: mais dias ativos → mais volta |
| **volume por sessão** | **0,400** | [0,392 – 0,409] | **abaixo de 0,50 — sentido invertido**: mais respostas por sessão → volta **menos** |
| **profundidade** | **0,624** | [0,616 – 0,633] | moderada: mais tempo lendo a explicação → mais volta |
| volume bruto de respostas | 0,721 | [0,713 – 0,729] | referência de comparação |
| taxa de acerto | 0,513 | [0,503 – 0,522] | **nada**: o IC quase toca 0,50; acertar muito não diz se o aluno fica |

O último resultado merece atenção: **desempenho não prevê permanência.** O bom aluno abandona na mesma proporção que o ruim.

---

## 7. O achado que exige explicação: o volume aponta ao contrário

AUC 0,400 significa que, tomando um par de alunos ao acaso, o que **fez mais respostas por sessão** é o que tem 60% de chance de ter abandonado.

Dito em taxas observadas, com intervalo de confiança de Wilson:

| faixa de volume | alunos | voltaram | taxa | IC 95% |
|---|---|---|---|---|
| baixo (≤ 12,5 resp/sessão) | 6.063 | 2.335 | **38,5%** | [37,3% – 39,7%] |
| médio (12,5 – 19,6) | 6.016 | 1.792 | 29,8% | [28,6% – 31,0%] |
| alto (> 19,6) | 6.040 | 1.396 | **23,1%** | [22,1% – 24,2%] |

Os intervalos não se sobrepõem. A diferença entre a faixa baixa e a alta é de **15 pontos percentuais**.

### Isso não é a frequência disfarçada

A objeção óbvia: quem aparece pouco precisa concentrar tudo numa sessão só, então talvez o "volume alto" seja só um sintoma de "frequência baixa", e não um sinal próprio. É o **paradoxo de Simpson** — a relação marginal inverter ou desaparecer ao condicionar.

Testado abrindo a tabela por dentro de cada faixa de frequência:

| | volume baixo | volume médio | volume alto |
|---|---|---|---|
| **frequência baixa** | 13,1% (n=1.627) | 11,4% (n=2.583) | **8,6%** (n=3.443) |
| **frequência média** | 28,6% (n=1.887) | 26,1% (n=1.402) | **23,1%** (n=1.103) |
| **frequência alta** | 62,1% (n=2.549) | 55,7% (n=2.031) | **56,5%** (n=1.494) |

A taxa cai da esquerda para a direita em **todas as três linhas**. O efeito sobrevive ao condicionamento — é sinal próprio, não composição. A magnitude condicional é menor (4 a 6 pontos) do que a marginal (15 pontos), o que é esperado: parte do efeito bruto *era* frequência.

**Interpretação:** sessão longa e isolada é comportamento de maratona — de quem tenta compensar o atraso de uma vez. É sinal de abandono, não de dedicação. Um produto que premia "muitas respostas numa sessão" está premiando o padrão errado.

O mesmo teste para a profundidade (sobe da esquerda para a direita em todas as linhas: 8,8→12,4 / 22,5→29,9 / 53,3→63,4) confirma que ela também é sinal próprio.

---

## 8. Por que três números e não um

A tentação natural é fazer média dos três e chamar de "engajamento de 0 a 100". Duas razões para não fazer.

**Primeira: eles não medem a mesma coisa.** Correlação de Spearman entre os eixos (Spearman e não Pearson porque as distribuições são assimétricas e o que interessa é a ordem, não a escala):

| | frequência | volume | profundidade |
|---|---|---|---|
| **frequência** | 1,00 | −0,30 | +0,33 |
| **volume** | −0,30 | 1,00 | −0,23 |
| **profundidade** | +0,33 | −0,23 | 1,00 |

Todos abaixo de 0,35 em módulo. Se fossem faces de um mesmo construto, estariam em 0,6 ou mais. São três coisas.

**Segunda, e mais grave: a média cancelaria sinais opostos.** O volume é o único eixo em que *mais é pior*. Somar os três faz o aluno que estuda 200 questões numa sessão única empatar com o que aparece todo dia com calma:

| perfil | frequência | volume | escore médio | retenção real |
|---|---|---|---|---|
| maratonista | baixa | alto | ~médio | **19,5%** |
| constante | alta | baixo | ~médio | **81,4%** |

Mesmo escore, quatro vezes a diferença no desfecho. Um número só apagaria exatamente a informação que serve para agir.

---

## 9. Os eixos substituem o volume bruto de respostas

Comparação honesta: será que tudo isso é uma maneira complicada de dizer "quem respondeu mais, fica"?

Regressão logística, treinada em 70% dos alunos e avaliada nos 30% restantes (5.455 alunos que o modelo nunca viu):

| o que entra no modelo | AUC fora do treino |
|---|---|
| só o total de respostas | 0,719 |
| **só os três eixos** | **0,803** |
| os três eixos + o total | 0,805 |

Os três eixos **substituem** o total de respostas e ainda ganham 8 pontos de AUC. O total, somado a eles, acrescenta 0,002 — ou seja, nada.

### Onde o mérito está concentrado — e isto é uma ressalva

| modelo | AUC fora do treino |
|---|---|
| só frequência | 0,800 |
| frequência + volume | 0,802 |
| frequência + profundidade | 0,802 |
| os três | 0,803 |

**Para prever quem volta, a frequência faz praticamente todo o trabalho.** Os outros dois somam 0,003 de AUC.

Isso não contradiz a §7 — os efeitos deles são reais e sobrevivem ao condicionamento — mas obriga a ser preciso sobre para que serve cada um:

- **frequência** é o eixo de **predição**. É dele que sai o alerta de risco.
- **volume** e **profundidade** são eixos de **diagnóstico**. Acrescentam pouco a *quem* está em risco, mas dizem *o que está acontecendo* com ele — maratona ou desatenção ao erro — que é o que decide qual intervenção faz sentido.

Os coeficientes do modelo confirmam a hierarquia (log-odds por desvio-padrão, features padronizadas):

| | coeficiente |
|---|---|
| frequência | **+1,158** |
| volume | −0,183 |
| profundidade | +0,144 |

Frequência pesa 6 a 8 vezes mais que os outros dois.

---

## 10. O modelo está calibrado

AUC mede ordenação, não se o número faz sentido como probabilidade. Para exibir "45% de chance de continuar" é preciso que, entre os alunos a quem o modelo dá 45%, cerca de 45% de fato continuem.

Nos 5.455 alunos fora do treino, agrupados em decis de risco previsto:

| decil | previsto | observado | n |
|---|---|---|---|
| 1 (menor risco de sair) | 7,3% | 5,1% | 546 |
| 2 | 10,4% | 8,6% | 545 |
| 3 | 12,7% | 10,3% | 546 |
| 4 | 15,4% | 17,2% | 545 |
| 5 | 18,7% | 17,8% | 546 |
| 6 | 23,6% | 29,0% | 545 |
| 7 | 30,5% | 36,3% | 545 |
| 8 | 42,0% | 46,5% | 546 |
| 9 | 61,4% | 57,8% | 545 |
| 10 | 84,8% | 82,1% | 546 |

**ECE = 0,031** — o erro médio de calibração é de 3 pontos percentuais. Para comparação, a regra de domínio hoje em produção no TrailUp tem ECE de 0,222.

O modelo é monotônico e razoavelmente calibrado, com um viés leve de subestimar no meio da distribuição.

---

## 11. As medidas são estáveis — é traço, não ruído

Se os eixos oscilassem conforme a amostra de respostas, não seriam propriedade do aluno. Teste: dividir as respostas de cada aluno **ao acaso** em duas metades e calcular os eixos em cada metade separadamente. Se medem algo real, as duas metades concordam.

Correlação entre as metades, 16.762 alunos:

| eixo | correlação entre metades |
|---|---|
| frequência | **+0,989** |
| volume | **+0,892** |
| profundidade | **+0,881** |

Todos acima de 0,88. São traços do aluno.

---

## 12. O limite: isto mede permanência, não aprendizado

Terceiro desfecho testado, para não confundir as duas coisas. Entre os 5.523 alunos que voltaram, quanto a taxa de acerto **melhorou** do primeiro para o segundo mês:

```
ganho = acerto_medio(dias 30-59) − acerto_medio(dias 0-29)
```

| medida | Spearman com o ganho |
|---|---|
| frequência | −0,035 |
| volume | −0,036 |
| profundidade | +0,027 |
| taxa de acerto no 1º mês | −0,241 |

**Nenhum dos três eixos prevê aprender mais.** Os valores são estatisticamente distinguíveis de zero só pelo tamanho da amostra; em magnitude, são zero para qualquer uso prático.

(O −0,241 da taxa de acerto é **regressão à média**, não achado: quem já estava alto tem menos espaço para subir e mais chance de cair. É artefato aritmético.)

Ganho médio geral: **−0,003** — a coorte não melhorou em 30 dias.

**Consequência de produto:** este módulo não pode ser usado como indicador de progresso, nem exibido ao aluno como "seu engajamento". Ele responde "vai continuar?", e só isso. Está escrito em caixa alta no cabeçalho do arquivo porque é a confusão mais provável — o eixo chamado "profundidade" convida justamente à leitura errada.

---

## 13. O que se entrega

Três faixas por eixo, cortadas nos terços da distribuição do EdNet, com a taxa de retenção observada em cada uma e intervalo de Wilson a 95%:

### frequência — cortes em 0,100 e 0,233 (3 e 7 dias ativos em 30)

| faixa | alunos | voltaram | taxa | IC 95% |
|---|---|---|---|---|
| baixa | 7.653 | 804 | **10,5%** | [9,8% – 11,2%] |
| média | 4.392 | 1.161 | 26,4% | [25,2% – 27,8%] |
| alta | 6.074 | 3.558 | **58,6%** | [57,3% – 59,8%] |

### volume — cortes em 12,5 e 19,6 respostas por sessão

| faixa | alunos | voltaram | taxa | IC 95% |
|---|---|---|---|---|
| baixo | 6.063 | 2.335 | **38,5%** | [37,3% – 39,7%] |
| médio | 6.016 | 1.792 | 29,8% | [28,6% – 31,0%] |
| alto | 6.040 | 1.396 | **23,1%** | [22,1% – 24,2%] |

### profundidade — cortes em 2,175 e 2,865 (≈ 8 s e ≈ 17 s de leitura)

| faixa | alunos | voltaram | taxa | IC 95% |
|---|---|---|---|---|
| baixa | 6.046 | 1.211 | **20,0%** | [19,0% – 21,1%] |
| média | 6.034 | 1.850 | 30,7% | [29,5% – 31,8%] |
| alta | 6.039 | 2.462 | **40,8%** | [39,5% – 42,0%] |

Monotônico nos três, com intervalos que não se sobrepõem entre faixas adjacentes.

No módulo, `Eixo.bom` já inverte o sentido no volume (lá, *baixo* é o favorável), e `alertas()` devolve só os eixos em faixa desfavorável — lista vazia significa nada a fazer.

---

## 14. Ressalvas antes de usar em produção

**Os cortes são do EdNet, não do TrailUp.** Adultos coreanos estudando TOEIC sozinhos. O que transfere é a **forma** do achado — quais sinais importam e em que direção. Os valores numéricos, não. Recalibrar com `recalibrar()` quando houver ~2.000 alunos com 60 dias de histórico.

**A janela de 30 dias é uma escolha, não uma medida.** Se o ciclo do TrailUp for semanal (aula, tarefa, prazo), a janela certa pode ser 7 ou 14 dias. Refazer a janela **antes** de recalibrar os cortes — mudar a janela muda o significado de todos os três eixos.

**O corte de sessão de 30 minutos é convenção.** Padrão em learning analytics, não medido aqui.

**Falta instrumentação para um dos três eixos.** A profundidade exige registrar o tempo na tela de explicação após o erro. O TrailUp não coleta tempo por questão hoje — é a mesma lacuna que bloqueia `chute.py` e `tempo.py`.

**Licença.** EdNet é CC BY-NC. O módulo carrega constantes medidas, não pesos treinados, mas a decisão sobre uso comercial é de quem publica.

**Sem grupo de controle.** Isto é observacional. "Frequência alta prevê retenção" não autoriza concluir que *fazer* o aluno aparecer mais dias o faria ficar. Para isso é preciso experimento.

---

# Segunda rodada: replicação em outras realidades

## 15. Por que replicar, e em quê

Tudo acima foi medido numa base só: **EdNet KT3** — adultos coreanos estudando TOEIC por conta própria, sem turma, sem professor, sem matrícula. Continuar é 100% voluntário.

Um achado medido numa base só não distingue **propriedade do aprendizado** de **propriedade daquela plataforma**. A única forma de separar é repetir onde o contexto é diferente.

### As três bases

| | EdNet KT3 | OULAD | ARES |
|---|---|---|---|
| quem | adulto coreano | universitário britânico | universitário |
| o quê | TOEIC, autoestudo | disciplina a distância | leitura em L2 |
| turma? | não | **sim** | sim |
| matrícula? | não | **sim, com abandono formal** | sim |
| continuar é | voluntário | voluntário com custo | **obrigatório** |
| unidade de atividade | resposta a questão | clique em material | leitura de texto |
| alunos na análise | 18.119 | **24.731** | 171 |
| **retenção base** | **30,5%** | **92,9%** | **87,7%** |

**O desenho é o mesmo nas três:** eixos nos dias 0–29, desfecho nos dias 30–59 (14/14 no ARES, cujo curso dura 6 semanas). Nada da segunda janela entra no cálculo.

### As medidas precisaram de tradução — e isso é uma limitação

Nenhuma das três plataformas registra a mesma coisa. O mapeamento foi:

| eixo | EdNet | OULAD | ARES |
|---|---|---|---|
| frequência | dias com resposta ÷ 30 | dias com clique ÷ 30 | dias com leitura ÷ 14 |
| volume | respostas ÷ sessão | cliques ÷ dia ativo | leituras ÷ sessão |
| profundidade | seg. lendo a explicação | **fração dos cliques em material de conteúdo** | duração mediana da leitura |

**Frequência é a única com tradução exata.** As outras duas são análogos, e a §16 mostra que isso importa.

---

## 16. O que replicou e o que não

AUC contra retenção, com IC 95% por bootstrap:

| eixo | EdNet | IC 95% | OULAD | IC 95% | replica? |
|---|---|---|---|---|---|
| **recência** (novo, §17) | **0,810** | [0,804 – 0,818] | **0,863** | [0,854 – 0,870] | **sim** |
| **frequência** | **0,799** | [0,792 – 0,806] | **0,811** | [0,800 – 0,820] | **sim** |
| tendência | 0,784 | [0,775 – 0,794] | 0,737 | [0,724 – 0,752] | sim |
| profundidade | 0,624 | [0,617 – 0,634] | 0,564 | [0,549 – 0,581] | sim, fraco |
| **volume** | **0,400** | [0,393 – 0,411] | **0,522** | [0,507 – 0,536] | **NÃO** |

### O volume não sobreviveu

Era o achado mais interessante da primeira rodada: mais respostas por sessão prediziam **menos** retenção, o efeito se mantinha dentro de cada quintil de frequência, e a leitura era boa — *maratona é sinal de abandono*.

No OULAD isso vira **0,522**, com intervalo [0,507 – 0,536] que quase toca o acaso. E a correlação com a frequência **inverte de sinal**: −0,30 no EdNet, **+0,23** no OULAD.

Duas explicações possíveis, e não dá para escolher entre elas com o dado que tenho:

1. **A tradução não é equivalente.** "Respostas por sessão" é quanto o aluno produz numa sentada; "cliques por dia ativo" mistura produzir com navegar. Pode ser que o construto não tenha sido testado de verdade.
2. **O efeito é do EdNet.** Num app de autoestudo sem prazo, maratonar é sinal de quem está tentando compensar. Numa disciplina com cronograma, a semana dita o ritmo e maratonar é normal.

**De qualquer modo, o volume sai do módulo.** Um eixo cujo sinal forte some quando muda a plataforma não é medida de engajamento — é propriedade de uma base. Mantê-lo seria exportar uma regra do EdNet como se fosse lei.

### O ARES: onde nada funciona, e por quê

| eixo | AUC |
|---|---|
| frequência | 0,492 |
| volume | 0,522 |
| profundidade | 0,565 |

Tudo acaso. A razão é estrutural e vale como aviso de escopo: **a leitura no ARES é tarefa obrigatória de disciplina**. 87,7% voltam, a mediana é de 2 dias ativos em 14, e não há variação a prever. Onde continuar não é escolha, não há engajamento a medir.

**Consequência para o TrailUp:** se a plataforma for usada dentro de aula, com o professor mandando, estes eixos medem pouco. Se for usada em casa, por escolha, medem bem. É uma pergunta de produto antes de ser de modelo.

---

## 17. O eixo novo: recência bate frequência nas duas bases

A hipótese: "com que frequência apareceu nos últimos 30 dias" perde informação, porque trata igual quem apareceu 10 vezes no começo e sumiu, e quem apareceu 10 vezes distribuídas.

**Recência** = dias ativos nos **últimos 10 dias** da janela.

| medida | EdNet | OULAD |
|---|---|---|
| frequência (30 dias) | 0,799 | 0,811 |
| **recência (10 dias)** | **0,810** | **0,863** |

Ganha nas duas, e por mais no OULAD. E o combinado ganha muito mais:

| modelo (AUC fora do treino) | EdNet | OULAD |
|---|---|---|
| 3 eixos antigos (freq, volume, prof) | 0,802 | 0,815 |
| **recência + frequência** | **0,843** | **0,861** |
| recência + frequência + profundidade | 0,841 | 0,860 |

**+4,1 e +4,6 pontos de AUC sobre o desenho anterior, nas duas bases.**

A profundidade não acrescenta nada depois das outras duas (−0,002 e −0,001), mas fica no módulo porque é o único eixo que diz **o que** está acontecendo, e não só *quem* está em risco.

### A curva, sem modelo nenhum

Retenção observada por dias ativos nos últimos 10:

| dias ativos | EdNet | OULAD |
|---|---|---|
| 0 | **12,4%** | **61,3%** |
| 1 | 47,3% | 88,6% |
| 2 | 55,5% | 93,5% |
| 3 | 68,1% | 96,3% |
| 4 | 75,9% | 98,2% |
| 5 | 80,8% | 98,6% |
| 6 | 88,7% | 99,2% |
| 8 | 93,3% | 99,9% |
| 10 | **98,2%** | **100,0%** |

**Monotônica nas duas**, com níveis completamente diferentes. É a forma que transfere; o nível, não.

E é uma **tabela de consulta** — não precisa de modelo, não precisa de dependência, e é auditável linha a linha. É o que o módulo usa.

> **Uma leitura de produto:** o eixo que mais prevê é também o mais barato de calcular e o que mais depende de **quando** você calcula. "Apareceu ultimamente" vale mais que "aparece bastante" — então recalcular perto da decisão vale mais que refinar o modelo.

---

## 18. Calibração: a régua não transfere

Aplicando os cortes do EdNet ao OULAD, **na mesma unidade**:

| eixo | base | baixo | médio | alto |
|---|---|---|---|---|
| frequência | EdNet | 42,2% | 24,2% | 33,5% |
| frequência | **OULAD** | **6,0%** | **16,7%** | **77,2%** |
| profundidade | EdNet | 33,4% | 33,3% | 33,3% |
| profundidade | **OULAD** | **100%** | **0%** | **0%** |

A profundidade colapsa porque a unidade nem é a mesma (log-segundos contra fração). Mas a **frequência tem unidade idêntica** — fração de dias ativos — e mesmo assim 77% do OULAD cai em "alto".

**Nenhum corte absoluto medido numa base pode ser exportado para outra.** É por isso que o módulo:

- entrega **só a recência** sem calibração — é contagem de dias, não escala, e os cortes (0 / 1–3 / 4+) são estruturais;
- exige `calibrar()` com ≥300 alunos da própria coorte para liberar os outros eixos;
- devolve `retencao_esperada = None` enquanto não houver calibração confiável, em vez de um número emprestado.

### E a forma de modelo também importa para calibrar

Testei três formas de combinar os eixos:

| base | forma | AUC | ECE |
|---|---|---|---|
| EdNet | logística linear | 0,841 | **0,049** |
| EdNet | + termo quadrático | 0,842 | 0,031 |
| EdNet | **recência como categoria** | 0,843 | **0,015** |
| OULAD | logística linear | 0,860 | 0,009 |
| OULAD | **recência como categoria** | 0,861 | **0,008** |

A logística linear **ganha AUC e perde calibração** no EdNet (ECE 0,049 contra 0,031 do modelo antigo) — a relação entre dias recentes e retenção não é linear no logito. Tratar a recência como **categoria** recupera tudo: ECE 0,015, três vezes melhor que o desenho anterior.

**AUC e calibração podem andar em direções opostas.** Otimizar só o primeiro teria piorado o produto — o número exibido é o que o professor lê.

---

## 19. O que mudou no módulo

| | antes | agora |
|---|---|---|
| eixos | frequência, **volume**, profundidade | **recência**, frequência, profundidade |
| volume | eixo principal, sinal negativo | **removido** — não replica |
| cortes | absolutos, do EdNet, embutidos | por coorte, via `calibrar()` |
| sem calibração | devolvia três faixas e uma retenção esperada | devolve **só a recência**, e `None` na probabilidade |
| AUC | 0,802 (uma base) | 0,843 / 0,861 (duas bases) |
| ECE | 0,031 | 0,015 / 0,008 |

---

## 20. Limites da segunda rodada

- **Duas bases não são muitas.** O volume caiu na segunda; não há garantia de que recência e frequência sobrevivam a uma terceira com contexto muito diferente. O que se pode dizer é que sobreviveram a **uma** mudança grande de contexto, e o volume não.
- **A tradução das medidas é minha escolha.** "Fração dos cliques em material de conteúdo" como profundidade no OULAD é defensável, não canônica. Um mapeamento diferente poderia dar outro resultado — e a queda do volume pode ser em parte isso.
- **O ARES é pequeno demais** (171 alunos) para distinguir "não replica" de "sem poder estatístico". O que ele mostra com clareza é o **teto**: com 87,7% de retenção não há o que prever.
- **Os desfechos não são idênticos.** No EdNet e no ARES, "voltou" é atividade. No OULAD existe também a **desmatrícula formal**, e contra ela a profundidade (0,611) é melhor que contra atividade (0,564) — sinal de que os dois desfechos não são a mesma coisa. Só o de atividade é comparável entre as bases.
- **A recência é adjacente à janela do desfecho.** Ela olha os dias 20–29 para prever os dias 30–59. Não há vazamento — é tudo informação disponível na hora da decisão — mas parte do ganho vem de estar mais perto no tempo, não de medir melhor.
- **Tudo observacional.** Nada aqui autoriza dizer que fazer o aluno aparecer o faria ficar.
