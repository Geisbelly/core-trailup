# Dificuldade da questão como faixa de valores

**Base:** EdNet KT3 · **Módulos:** [`dificuldade.py`](../../trailup_core/dificuldade.py), [`ritmo.py`](../../trailup_core/ritmo.py) · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê)

---

## 1. Pergunta de negócio

O TrailUp precisa saber quão difícil é cada questão — para calibrar trilha, para avisar o professor e para alimentar o modelo de domínio (que é onde está 0,704 dos 0,750 de AUC, §3).

A pergunta tem duas partes, e a segunda é a que costuma ser ignorada:

> **Qual a taxa de acerto desta questão — e quanta confiança eu tenho nisso?**

Uma questão respondida por 6 alunos e uma respondida por 600 não sabem a mesma coisa. Devolver "72% de acerto" nos dois casos é esconder a diferença que mais importa para quem decide.

**Decisão de desenho:** devolver um **intervalo**, não uma categoria. Quem consome aplica o próprio limiar. A afirmação categórica ("esta é difícil") vira um teste sobre o intervalo, e só é verdadeira quando o intervalo **inteiro** sustenta.

---

## 2. Dados

**Fonte:** EdNet KT3 (TOEIC, adultos, autoestudo, licença **CC BY-NC**), sobre o rótulo corrigido — 6.504.124 respostas finais (ver [M2 §2](../m2_knowledge_tracing/RELATORIO_M2.md) sobre a contaminação de 21,7% que foi consertada).

**Unidade de análise: a questão.** Uma linha por questão.

| | |
|---|---|
| questões com verdade confiável | **11.338** |
| respostas por questão — p5 | 88 |
| p25 | 208 |
| **mediana** | **372** |
| p75 | 651 |
| p95 | 1.755 |

**Distribuição da dificuldade** (taxa de acerto por questão):

| p10 | p25 | mediana | p75 | p90 | média | desvio |
|---|---|---|---|---|---|---|
| 0,483 | 0,608 | **0,730** | 0,832 | 0,905 | 0,709 | 0,163 |

A distribuição é assimétrica à esquerda: a maioria das questões é respondida corretamente pela maior parte dos alunos, e a cauda difícil é fina. Isso tem consequência direta na §9 — poucas questões conseguem ser cravadas como "difícil".

> **Atenção ao regime.** Mediana de 372 respostas por questão é um corpus **rico**. Uma turma de 30 alunos do TrailUp vive na ponta esquerda de todas as tabelas deste documento. É por isso que a §10 existe.

---

## 3. Por que a questão, e não o aluno

Antes de investir na medida da questão, vale checar se ela é mesmo onde está o sinal. Decomposição aditiva, prevendo a probabilidade de acerto:

| modelo | AUC | ECE |
|---|---|---|
| **só a questão** (mesma previsão para todos os alunos) | **0,704** | 0,007 |
| aditivo: questão + habilidade geral do aluno | 0,713 | 0,004 |
| + habilidade do aluno no tópico | 0,713 | 0,003 |
| modelo completo (não-linear, 19 features) | 0,750 | 0,003 |

A questão sozinha entrega **0,704 dos 0,750**. Saber quem é o aluno acrescenta 0,046, e metade disso só aparece com modelo não-linear.

No modelo aditivo, o coeficiente da dificuldade é **1,076** e o da habilidade do aluno **0,218**. O Rasch clássico (`logit p = habilidade − dificuldade`, coeficientes simétricos) **não se sustenta neste corpus**: a média corrente de acerto do aluno é proxy fraco de habilidade.

É o quarto experimento independente a apontar na mesma direção: **o sinal está no conteúdo.**

---

## 4. Como se estima: Beta-Binomial com prior do próprio corpus

A taxa bruta `acertos / respostas` é péssima com poucos dados: 1 de 1 vira "100% de acerto". A correção padrão é encolher para a média do corpus, com peso proporcional à evidência:

```
taxa estimada = (acertos + a) / (respostas + a + b)
```

onde `Beta(a, b)` é o **prior** — o que se sabe sobre uma questão antes de ela receber qualquer resposta.

### O prior sai do dado

`derivar_prior()` faz a conta: casa os momentos da distribuição real de dificuldade do corpus e **desconta o ruído binomial** (parte da variância observada entre questões é só amostragem, não diferença real de dificuldade).

Resultado no EdNet: **Beta(4,94 , 2,02) — força 7,0**, média 0,709.

O efeito de errar o prior, medido (cobertura nominal de 90%, n=30, contra a verdade medida em ≥200 respostas independentes):

| força do prior | cobertura | IC 95% | largura |
|---|---|---|---|
| 0 (sem encolhimento) | 74,6% | [73,0% – 76,2%] | 0,237 |
| 3 | 73,6% | [72,0% – 75,1%] | 0,233 |
| **7,2 (derivado)** | 72,2% | [70,6% – 73,8%] | 0,224 |
| 15 | 64,5% | [62,8% – 66,2%] | 0,211 |
| 30 | 53,0% | [51,2% – 54,8%] | 0,187 |
| 60 | 40,4% | [38,6% – 42,1%] | 0,155 |

*(Esta tabela usa reamostragem binomial, que subestima a cobertura — ver §7. O que ela mede corretamente é o **efeito relativo** do prior.)*

Prior forte estreita o intervalo e desloca o centro: é mentira confortável. Força 60 devolve um intervalo de 90% que acerta 40% das vezes.

**`derivar_prior()` não é parâmetro de ajuste — é propriedade do banco de questões**, e deve ser refeito quando o banco mudar.

---

## 5. O que `n` conta — e o que ele não conta

`n` é **quantos alunos responderam àquela questão**. Não é quantas respostas um aluno deu.

A confusão é fácil porque este estudo tem três `n` diferentes:

| onde | o que `n` conta |
|---|---|
| **dificuldade da questão** (aqui) | quantos **alunos** responderam **àquela questão** |
| M2, `n_prev_part` | quantas respostas **aquele aluno** deu **naquele tópico** |
| M1, "3–5 check-ins" | quantos check-ins **aquele aluno** fez |

E há uma correção embutida: no EdNet **9 a 11% das respostas são o mesmo aluno reencontrando a questão**. Essas respostas não são independentes; contá-las como se fossem estreita o intervalo indevidamente. O parâmetro `alunos` do `estimar()` reponderá pelo número de alunos distintos.

Efeito medido:

| n | reencontro | cobertura sem correção | **com `alunos`** | largura sem | com |
|---|---|---|---|---|---|
| 30 | 9,4% | 89,3% [88,6 – 90,0] | **90,8%** [90,2 – 91,4] | 0,229 | 0,240 |
| 50 | 9,9% | 88,2% [87,5 – 88,9] | **89,8%** [89,1 – 90,5] | 0,185 | 0,195 |
| 100 | 10,1% | 84,1% [83,3 – 85,0] | **86,1%** [85,3 – 86,9] | 0,136 | 0,143 |
| 200 | 11,1% | 77,6% [76,4 – 78,7] | **80,0%** [78,9 – 81,1] | 0,098 | 0,104 |

A correção vale 1,5 a 2,4 pontos de cobertura ao custo de ~5% de largura. **Usar sempre que o número de alunos distintos estiver disponível.**

---

## 6. As métricas, definidas

**Cobertura** — a fração das questões em que o intervalo de 90% de fato contém a verdade. Se o intervalo for honesto, dá 90%. Abaixo disso, ele é estreito demais (excesso de confiança); acima, largo demais (desperdício).

**Verdade** — aqui, a taxa de acerto medida nas respostas **seguintes** àquelas usadas na estimativa, exigindo pelo menos 200 delas (erro-padrão abaixo de 0,035). É verdade medida, não latente.

**Largura** — tamanho do intervalo de 90%. É o que diz se o número serve para alguma coisa.

**Erro absoluto** — distância média entre a estimativa pontual e a verdade.

**AUC** — probabilidade de ordenar corretamente um par acerto/erro. 0,50 é cara ou coroa.

**Silhueta** — quão separados estão os grupos de um agrupamento (−1 a 1). **ARI** e **AMI** — concordância entre dois agrupamentos, corrigida por acaso; 0 é acaso. **d de Cohen** — separação entre duas médias em desvios-padrão; acima de 0,8 é grande.

---

## 7. Cobertura: o intervalo é honesto até ~50 respostas

`estimar()` devolve um intervalo **posterior sobre a taxa latente** — não um intervalo preditivo sobre a contagem de uma amostra futura. São objetos distintos, com coberturas distintas, e a medida abaixo é sobre o que o código de fato devolve.

Medido com respostas reais (as *n* primeiras de cada questão, na ordem em que chegaram), verdade nas ≥200 seguintes:

| n | questões | **cobertura posterior (nominal 90%)** | IC 95% | largura | erro absoluto | *(cobertura preditiva)* |
|---|---|---|---|---|---|---|
| 5 | 8.568 | **92,2%** | [91,6 – 92,8] | 0,403 | 0,095 | *91,8%* |
| 10 | 8.472 | **91,7%** | [91,1 – 92,3] | 0,338 | 0,079 | *91,9%* |
| 21 | 8.276 | **90,7%** | [90,1 – 91,4] | 0,264 | 0,064 | *91,8%* |
| 30 | 8.101 | **89,3%** | [88,6 – 90,0] | 0,229 | 0,057 | *90,9%* |
| 50 | 7.771 | 88,2% | [87,5 – 88,9] | 0,185 | 0,047 | *90,8%* |
| 100 | 7.089 | **84,1%** | [83,3 – 85,0] | 0,136 | 0,039 | *88,9%* |
| 200 | 5.090 | **77,6%** | [76,4 – 78,7] | 0,098 | 0,033 | *85,9%* |

**Até n=30 o intervalo é nominal** (89,3% a 92,2% para um nominal de 90%), e com a correção por alunos distintos (§5) fica nominal até n=50.

**Acima de n=100 ele é estreito demais.** A razão é conhecida: o modelo binomial assume que todas as respostas vêm da mesma taxa fixa. Com muita evidência, o posterior encolhe proporcionalmente a 1/√n — mas a incerteza real **não** encolhe do mesmo jeito, porque a dificuldade de uma questão não é constante: muda com a coorte, com o momento do curso, com quem chega primeiro.

**Consequência prática:** o regime do TrailUp (turma de 30, talvez 60) está exatamente na faixa em que o intervalo é honesto. Quem for usar isto com milhares de respostas por questão precisa de um modelo hierárquico, não deste.

*(A coluna em itálico mostra a cobertura preditiva, para comparação. Ela se mantém próxima do nominal por mais tempo — mas não é o que o código devolve.)*

### Sem dependência

A aproximação normal do posterior bate o Beta exato dentro de **1 ponto de cobertura** em toda a faixa testada. A implementação usa a aproximação e não precisa de `scipy`.

---

## 8. Largura: quando o número serve para alguma coisa

| respostas | largura do intervalo de 90% | o que se sabe |
|---|---|---|
| 5 | 0,403 | quase nada |
| 21 | 0,264 | ±13 pontos |
| **30** | **0,229** | **±11 pontos** |
| 50 | 0,185 | ±9 pontos |
| 100 | 0,136 | ±7 pontos |
| 200 | 0,098 | ±5 pontos |

Com 30 respostas você sabe a dificuldade **±11 pontos percentuais**. Antes de ~50 o intervalo é largo, e é honesto que seja: a informação não existe ainda.

**Questão nova devolve o prior.** `estimar(0, 0)` dá `71% (entre 45% e 97%)` — a distribuição do banco de questões, com toda a largura. É a resposta correta para uma questão que ninguém respondeu.

---

## 9. A afirmação categórica vira teste sobre o intervalo

`afirmar(faixa, 0.50, 'abaixo')` só é verdadeiro quando o intervalo **inteiro** fica abaixo do limiar. Medido (simulação a partir da verdade de cada questão, 3.000 sorteios por linha):

| n | crava ("difícil" ou "fácil") | acerta quando crava | IC 95% |
|---|---|---|---|
| 10 | 0,3% | 100% | — (n=8) |
| 21 | 5,6% | 99,4% | [98,2 – 100] |
| **30** | **14,2%** | **96,2%** | [94,4 – 98,0] |
| 50 | 20,1% | 98,5% | [97,5 – 99,5] |
| 100 | 23,8% | 99,0% | [98,3 – 99,7] |
| 200 | 28,0% | 98,9% | [98,2 – 99,6] |

Duas leituras:

**Quando crava, acerta.** Acima de 96% em toda a faixa utilizável.

**E crava pouco — satura em ~28%.** Mesmo com 200 respostas, três quartos das questões nunca ganham rótulo. Não é falha do método: é a distribuição da §2. A maioria das questões é média, e o intervalo honesto sobre elas cobre os dois lados do corte. **A maioria das questões devolve só o intervalo, que é o que de fato se sabe sobre elas.**

### Questão nova não dá para classificar a frio

Tentativa de prever a faixa **sem nenhuma resposta**, só por metadado (validação agrupada por bundle):

| features | acerto |
|---|---|
| chutar a faixa mais comum | 55,4% |
| metadado: `part`, n de tags, tamanho e posição no bundle | 57,7% |
| + 60 tags de assunto | 58,0% |

**Metadado não prediz dificuldade** — ganha 2,5 pontos sobre chutar, e a matriz de confusão mostra que ele praticamente nunca arrisca "difícil" (acerta 12 de 1.282). Em respostas observadas isso vale ~7 alunos.

Consequência para o banco vazio: **a questão nasce indeterminada e tem de ganhar a classificação.** Não há atalho por metadado.

> **A lacuna mais promissora, não testada:** o EdNet **não traz o texto** das questões, só ids e tags. O TrailUp tem enunciado, alternativas e gabarito — um *embedding* do enunciado pode predizer dificuldade onde a tag não prediz. É o experimento a fazer quando houver questões reais.

---

## 10. Existem grupos de dificuldade? O `k=4` era imposto

Duas objeções, nesta ordem, e as duas procedem. **Primeira:** os cortes 50/80 eram escolha minha. **Segunda:** ao aplicar k-means eu ainda escolhia `k`. Um algoritmo que **recebe** a quantidade de grupos não **descobre** a quantidade de grupos.

### Três métodos que descobrem a quantidade

Sobre 10.526 questões com ≥100 respostas, em quatro dimensões (taxa de acerto, latência mediana, taxa de troca de alternativa, latência da primeira resposta):

| método | como funciona | grupos | estabilidade |
|---|---|---|---|
| **HDBSCAN** | densidade; só pede o tamanho mínimo de grupo | **2** + 14,3% de ruído | ARI **1,000** variando o parâmetro de 50 a 400 |
| Dirichlet Process GMM | teto de 15 componentes, poda o que não usa | 13 com peso >2% | fragmenta em vez de podar |
| Mean Shift | largura estimada do próprio dado | 4 → 3 → 1 | colapsa conforme a largura |

Só o HDBSCAN é estável, e dá a melhor silhueta: **0,470**, contra 0,446 do k-means com k=2 e **0,366 do meu k=4**.

### O que ele encontrou: a separação é no tempo, não no acerto

| feature | grupo rápido | grupo lento | separação (d de Cohen) |
|---|---|---|---|
| **latência mediana** | 19 s | 69 s | **3,30** |
| acerto | 0,70 | 0,77 | 0,50 |
| taxa de troca de alternativa | 0,15 | 0,18 | 0,57 |

E o "ruído" (14,3%) não é um terceiro grupo: é a **cauda** — desvio de acerto 0,22 contra 0,15 dos grupos, latência mediana de 80 s.

**Conclusão:** o eixo do **tempo** tem estrutura discreta real (d = 3,30 é separação enorme); o eixo do **acerto** é contínuo (d = 0,50). Um `k=4` impõe uma grade 2×2 e batiza os quadrantes (`armadilha`, `travamento`) com nomes que sugerem entidades que o dado não separa.

Isso explica um resultado que antes parecia arbitrário: o classificador acertava **99% no eixo do tempo e 83% no eixo do acerto**. Não era um eixo mais fácil — era estrutura real contra grade imposta.

### A fronteira, e quão barata ela é

O grupo rápido vai até 40 s (p95 = 27 s); o lento começa em 36 s (p5 = 52 s). Fronteira em **~40 s de latência mediana**.

| respostas | acerto do eixo tempo |
|---|---|
| 5 | **98%** |
| 10 | 99% |
| 21 | 100% |

**Cinco respostas classificam o ritmo.** É o eixo barato; o caro é o acerto, que precisa de 30 para ±11 pontos.

### O desenho final

**Categoria no que é discreto, intervalo no que é contínuo:**

```
     24/30    15s  rapida, acerto entre 67% e 89% (90%, n=30)
     20/100   75s  lenta,  acerto entre 17% e 30% (90%, n=100)   -> dificil
      4/6     12s  rapida, acerto entre 49% e 89% (90%, n=6)
```

O ritmo vira rótulo porque o dado separa. O acerto vira intervalo porque não separa — e fingir um corte ali seria inventar fronteira onde há contínuo.

---

## 11. Categorizar pelo padrão de quem acerta — e por que não deu

Objeção correta à §10: agrupei as questões por **estatísticas agregadas**, não pelo que define dificuldade de verdade — **quem** acerta cada uma. Duas questões com 60% de acerto podem ser acertadas por alunos completamente diferentes; aí não são a mesma dificuldade, são habilidades diferentes.

### O teste

Bloco denso: **1.000 questões × 3.668 alunos**, 1,04 M respostas, densidade 28,4%, mediana de 312 alunos respondendo qualquer par de questões.

**Passo crítico: remover habilidade do aluno e dificuldade da questão antes de comparar.** Sem isso tudo correlaciona com tudo — aluno bom acerta tudo — e o agrupamento só redescobre a habilidade geral. Ajuste aditivo iterativo; habilidade + dificuldade explicam 15,1% da variância (o resto é o ruído irredutível de uma variável binária).

Depois disso, correlação entre questões sobre o **resíduo**.

### Sobrou estrutura — mas fraca e sem dimensão dominante

| | valor |
|---|---|
| desvio da correlação residual | **0,0646** |
| esperado por ruído puro (≈1/√n) | 0,0310 |
| pares com \|r\| > 0,1 | 11,4% |
| maior autovalor explica | **1,5%** |
| cinco maiores explicam | 5,1% |

O sinal é **2,1× o ruído** — existe. Mas o espectro é achatado: **não há poucos eixos de habilidade dominantes.**

### E ele corresponde a assunto, na direção certa

| pares de questões | correlação residual média |
|---|---|
| nenhuma tag em comum | −0,0051 |
| até 1/3 das tags em comum | +0,0065 |
| mais de 1/3 das tags | **+0,0128** |
| mesma `part` | +0,0015 |
| `part` diferente | −0,0072 |

Monotônico na sobreposição de tags (correlação de *r* com a sobreposição: +0,0505). O efeito é real e interpretável — só que minúsculo.

### Mas não sustenta categorias

| método | resultado |
|---|---|
| **HDBSCAN** sobre a distância de correlação | **0 grupos, 100% ruído** |
| espectral k=5 | AMI com `part` **+0,106** |
| espectral k=10 | +0,101 |
| espectral k=20 | +0,080 |
| *(referência: AMI entre `part` e a tag principal)* | *+0,303* |

O HDBSCAN, que na §10 encontrou estrutura clara no tempo, aqui não encontra **nada**. O espectral só acha grupos porque é obrigado a achar, e eles capturam um terço do alinhamento que os metadados já têm entre si.

### A conclusão

**Neste corpus a dificuldade é unidimensional.** Existe uma habilidade latente só; as questões diferem em *quanto* exigem dela, não em *qual* habilidade exigem. Não há categorias a descobrir — há um contínuo, que é exatamente o que o intervalo da §7 modela.

Isso fecha dois resultados que estavam soltos:

- as **189 tags** deram +0,001 de AUC ao modelo M2: a tag codifica assunto, mas assunto não cria dimensão de habilidade separada aqui;
- o eixo do **acerto** não formava grupos na §10: não é limitação do método de agrupamento, é ausência de estrutura multidimensional.

> **Ressalva que importa para o TrailUp.** EdNet é TOEIC — **uma área, um tipo de prova**. É o cenário menos favorável possível para encontrar múltiplas habilidades: tudo mede inglês para negócios. Uma plataforma escolar cobrindo matemática, história e biologia tem muito mais chance de ter estrutura multidimensional real — um aluno forte em interpretação e fraco em cálculo produz exatamente o padrão que aqui não apareceu. **"Unidimensional" é propriedade deste corpus, não lei geral.** O teste é barato e está pronto (`scripts/01_estrutura.py`, `02_valida_tags.py`): roda sobre qualquer matriz aluno × questão.

---

## 12. Manual de operação: tudo em função de *n*

Consolidação no mesmo recorte. `n` = quantos **alunos** responderam à questão. Verdade = taxa medida nas respostas seguintes (≥200).

| n | ritmo | largura | erro absoluto | cobertura | crava | acerta ao cravar |
|---|---|---|---|---|---|---|
| 5 | 98% | 0,403 | 0,095 | 92,2% | 0% | — |
| 10 | 99% | 0,338 | 0,079 | 91,7% | 0,3% | 100% |
| 21 | 100% | 0,264 | 0,064 | 90,7% | 5,6% | 99,4% |
| **30** | **100%** | **0,229** | 0,057 | **89,3%** | **14,2%** | 96,2% |
| 50 | 100% | 0,185 | 0,047 | 88,2% | 20,1% | 98,5% |
| 100 | 100% | 0,136 | 0,039 | 84,1% ⚠️ | 23,8% | 99,0% |
| 200 | 100% | 0,098 | 0,033 | 77,6% ⚠️ | 28,0% | 98,9% |

⚠️ = intervalo estreito demais; ver §7.

### Traduzido para turma real

| cenário | n | ritmo | acerto | crava em |
|---|---|---|---|---|
| **turma de 30, 1 passada** | 30 | 100% | **±11%** | 14% |
| turma de 30, 2 passadas | 60 | 100% | ±9% | ~21% |
| turma de 30, 3 passadas | 90 | 100% | ±7% | ~23% |
| turma de 25, 2 passadas | 50 | 100% | ±9% | 20% |

### Três leituras

**O ritmo é praticamente grátis.** 98% com 5 respostas. Meia dúzia de alunos já diz se a questão é de reconhecimento ou de elaboração — e é o eixo que o dado separa de verdade.

**O acerto custa caro e satura devagar.** De n=30 para n=200 — quase 7× mais dado — a largura cai de ±11% para ±5%. Não existe atalho: a incerteza é binomial.

**A afirmação categórica satura em ~28%**, pelo motivo da §9.

### Dois vieses checados

**Quem responde primeiro difere de quem responde depois?** Praticamente não. Acerto 0,699 nas 30 primeiras contra 0,701 no resto; latência 22,5 s contra 22,6 s; correlação entre metades de 0,894 (acerto) e 0,987 (latência). Usar as 30 primeiras em vez de 30 aleatórias custa **+0,005** de erro. Isso importa porque **numa turma as *n* respostas *são* a turma inteira** — a tabela transfere.

**As linhas de *n* alto rodam sobre questões diferentes?** Sim — questões muito respondidas são mais difíceis (acerto 0,777 com <100 respostas, 0,633 com 1000+). Refazendo a tabela com **conjunto fixo** (as mesmas 8.101 questões em todas as linhas), a diferença em "crava" é no máximo **1,5 ponto** e as demais métricas são idênticas. **O crescimento com *n* é evidência, não composição.**

---

## 13. O que isso implica para o produto

Com **uma turma passando uma vez** pela atividade você já tem: o ritmo de toda questão (100%), a dificuldade de cada uma a ±11 pontos, e afirmação categórica sobre 14% delas — as que realmente destoam.

É pouco para um ranking fino de dificuldade e é **suficiente** para as duas decisões que importam: separar questão de reconhecimento de questão de elaboração, e apontar ao professor as poucas que estão claramente fora da curva.

---

## 14. Limites

- **EdNet é TOEIC**, múltipla escolha, adulto, autoestudo. A *mecânica* transfere; os números são daquele corpus e devem ser refeitos com dado próprio (`derivar_prior()` no banco novo).
- **O intervalo é estreito demais acima de ~100 respostas** (§7). Para o regime de turma isso não morde; para uso com corpus grande, seria preciso modelo hierárquico.
- **Dificuldade aqui é taxa de acerto**, que mistura dificuldade intrínseca com quem respondeu. Numa turma homogênea os dois se confundem; é o problema que a [dificuldade por turma](DIFICULDADE_TURMA.md) endereça.
- **A tabela de confiança não extrapola** acima de 200 respostas.
- **Questão personalizada não grava em `questao_aluno`** (`QuestionActivity.tsx:1281`) — sem corrigir isso, a maior parte das questões do fluxo adaptativo nunca acumula resposta e fica indeterminada para sempre.
- **A fronteira de 40 s é do EdNet**, onde a latência é do `enter` do bundle até responder. O TrailUp tem `time_metrics.activities[].active_sec`, que já desconta ociosidade acima de 15 s — mais limpo, mas a fronteira teria de ser reencontrada.
- **A taxa de troca de alternativa não existe no TrailUp** (exigiria emitir `answer_change`). Ela entrou no agrupamento, mas não é o que separa — sua ausência não invalida a estrutura.

---

## Apêndice — o que mudou, e por quê

**A tabela de cobertura validava o objeto errado.** A versão anterior media a cobertura de um intervalo **preditivo** — construído com `betabinom` para conter a *contagem* de uma amostra futura. Mas `estimar()` devolve um intervalo **posterior sobre a taxa latente**. O preditivo é mais largo, porque inclui o ruído de amostragem do futuro; validar um e publicar o outro faz o documento afirmar mais do que o código entrega.

Refeita sobre o objeto certo (§7), a cobertura é **nominal até n≈30** e cai depois: 84,1% em n=100 e 77,6% em n=200. O intervalo é estreito demais com muita evidência, porque o modelo binomial assume taxa fixa e a dificuldade de uma questão não é constante entre coortes.

**Não morde no regime do TrailUp** (turma de 30 a 60), mas estava afirmado errado.

**O prior de força 15 vinha de escolha no olho**, não do corpus. `derivar_prior()` dá **força 7,0** no EdNet, e o efeito de errar isso está medido na §4: com força 15 o intervalo de 90% cobre 64,5%.

**O `k=4` das questões era grade imposta.** A versão anterior nomeava quadrantes (`armadilha`, `travamento`) de uma grade 2×2. Os métodos que descobrem a quantidade de grupos encontram **2**, separados pelo tempo — e o eixo do acerto é contínuo (§10).

### Números que mudaram entre versões

| | antes | final |
|---|---|---|
| cobertura em n=30 | "89,4%" (preditiva) | **89,3%** (posterior) |
| cobertura em n=200 | "87,9%" (preditiva) | **77,6%** (posterior) |
| prior | força 15, no olho | **força 7,0**, derivada do corpus |
| grupos de questões | 4, nomeados | **2**, separados pelo tempo |
