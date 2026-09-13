# Traços medem, grupos não existem

**Base:** EdNet KT3 · **Módulos:** [`chute.py`](../../trailup_core/chute.py), [`discriminacao.py`](../../trailup_core/discriminacao.py) · **Histórico de correções:** [apêndice](#apêndice--o-que-mudou-e-por-quê)

---

## 1. Pergunta de negócio

Duas perguntas que o TrailUp faz hoje sem instrumento:

> **Este aluno errou porque não sabe, ou porque não tentou?** — o `agente_notificacao` só enxerga erro.
>
> **Dá para agrupar alunos em perfis de comportamento?** — os 7 perfis BrainHex são atribuídos por quiz declarado no signup, sem verificação.

E uma terceira, de qualidade de banco: **esta questão está quebrada?**

---

## 2. Dados

**Fonte:** EdNet KT3 (TOEIC, adultos, autoestudo, **CC BY-NC**), rótulo corrigido.

| | |
|---|---|
| respostas | 6.504.124 |
| alunos | 27.792 |
| questões | 11.555 |
| alunos com ≥100 respostas (usados no agrupamento) | **11.236** |
| respostas por aluno — p25 / mediana / p75 / p95 | 153 / **263** / 546 / 1.749 |

**Unidade de análise:** varia por seção — a questão na §3, o aluno nas §4 e §5.

---

## 3. Questão defeituosa: o método está certo, o corpus quase não tem o fenômeno

### A medida

**Discriminação** = correlação entre acertar aquela questão e a habilidade geral do aluno, **calculada excluindo aquela questão** (senão a questão entra nos dois lados da correlação e o valor infla sozinho).

Interpretação: **negativa significa que aluno bom erra** — gabarito trocado, enunciado ambíguo, distrator melhor que a resposta certa. É o sinal clássico de item quebrado na teoria de resposta ao item.

### O que o corpus tem

10.636 questões com discriminação estimável em duas metades independentes de alunos:

| | metade A | metade B |
|---|---|---|
| média | +0,209 | +0,208 |
| p10 | +0,076 | +0,073 |
| mediana | +0,208 | +0,206 |
| p90 | +0,344 | +0,344 |

Média de 0,21 é normal para múltipla escolha.

| | fração | questões |
|---|---|---|
| discriminação **negativa** na metade A | 2,96% | 315 |
| negativa na metade B | 2,85% | 303 |
| **negativa nas duas metades** | **0,53%** | **56** |
| quase nula (<0,05) na metade A | 6,9% | 738 |
| baixa (<0,15) | 25,2% | — |

**A queda de 2,96% para 0,53% é o resultado.** A maior parte do que aparece como negativo numa metade **não aparece na outra** — é ruído amostral, não defeito de item.

### Quão bem dá para marcar

Replicação entre as metades: correlação **+0,396** [IC 95%: +0,379 – +0,413].

| marcadas na metade A | confirmam na metade B | base | lift |
|---|---|---|---|
| discriminação < 0 (315 questões) | **17,8%** | 2,85% | **6,2×** |
| discriminação < 0,05 (738 questões) | **27,1%** | 6,8% | 4,0× |

**Lift de 4 a 6× sobre a base, mas precisão de 18 a 27%.** Marcar 315 questões para o professor sabendo que 4 em cada 5 estão boas não é um produto — é ruído com aparência de diagnóstico.

> **Para o TrailUp isso inverte de valor.** O EdNet é banco comercial curado; item quebrado não sobrevive ali, e é por isso que a taxa é tão baixa que o sinal se perde no ruído. No TrailUp as questões são escritas por professor e **geradas por IA** — item com gabarito errado ou distrator ambíguo é esperado, não excepcional. Com prevalência maior, a mesma precisão de marcação sobe. O método (`discriminacao.py`) está pronto, e é dos poucos que deve valer **mais** no TrailUp do que valeu no corpus de referência.

---

## 4. Chute: validado, e é traço

### A definição

> **Chute** = resposta **rápida demais para aquela questão** (menos de 30% da latência mediana dela) **e** errada.

O limiar é **relativo à questão** porque uma questão de 60 s e uma de 15 s não têm o mesmo "rápido". Exige ≥50 respostas na questão para a mediana ser estável.

Frequência: **1,01%** das respostas.

### A validação: a definição tem consequência medível

Uma definição arbitrária não vale nada se não prevê algo independente. O teste: quando o mesmo aluno **reencontra a mesma questão**, ele acerta?

594.398 reencontros analisados:

| na primeira vez, o aluno | acerta ao reencontrar | IC 95% | n |
|---|---|---|---|
| errou **chutando** | **65,1%** | [64,0% – 66,1%] | 7.816 |
| errou trabalhando | **72,5%** | [72,4% – 72,7%] | 417.957 |
| tinha acertado | 86,1% | [86,0% – 86,3%] | 168.625 |

**Quem chutou aprendeu menos** — 7,4 pontos abaixo de quem errou trabalhando, com intervalos que não se sobrepõem. É exatamente o que o conceito prevê: errar tentando deixa alguma marca; errar sem tentar não deixa nenhuma.

### É traço estável do aluno

Correlação da taxa de chute entre duas metades aleatórias e independentes das respostas: **+0,922**.

Distribuição: mediana 0,4%, p90 2,0%, **p99 7,9%** — uma cauda pequena de alunos que chuta muito. Concentra em questão difícil (+0,374) e lenta (+0,331).

**Uso:** é a distinção "não sabe" × "não tentou". E é uma afirmação sobre a **resposta**, não uma inferência de estado emocional de menor.

---

## 5. Agrupar alunos: eles medem bem, mas não formam grupos

A metade que faltava — agrupei questões e não achei estrutura (ver [FAIXA_DIFICULDADE §11](../dificuldade/FAIXA_DIFICULDADE.md)); alunos podiam ter.

### Os oito traços

11.236 alunos com ≥100 respostas, em oito dimensões:

| traço | o que mede | p10 | mediana | p90 |
|---|---|---|---|---|
| acerto | taxa de acerto | 0,544 | 0,662 | 0,763 |
| taxa de chute | § 4 | 0,000 | 0,004 | 0,019 |
| ritmo relativo aos pares | latência ÷ mediana da questão | 0,80 | 1,01 | 1,23 |
| constância de ritmo | desvio do ritmo relativo | 0,35 | 0,57 | 1,19 |
| trocar de alternativa | fração das respostas em que trocou | 0,054 | 0,131 | 0,262 |
| dificuldade que enfrenta | dificuldade média das questões que escolhe | 0,272 | 0,334 | 0,405 |
| tempo lendo explicação | segundos, mediana | 6,1 | 13,8 | 37,7 |
| respostas por sessão | volume por sessão | 8,3 | 15,7 | 34,3 |

### Cada traço, isolado, replica muito

Correlação entre duas metades aleatórias das respostas do mesmo aluno:

| traço | confiabilidade |
|---|---|
| respostas por sessão | **+0,980** |
| tempo lendo explicação | +0,950 |
| ritmo relativo aos pares | +0,948 |
| dificuldade que enfrenta | +0,934 |
| taxa de chute | +0,922 |
| trocar de alternativa | +0,897 |
| acerto | +0,799 |
| constância de ritmo | +0,706 |

**Todos acima de 0,70.** São propriedades do aluno, não ruído de amostragem.

### Mas os grupos não replicam

Método que **descobre** a quantidade de grupos (HDBSCAN, `min_cluster_size=200`), sobre as features transformadas por quantis para distribuição normal — passo que importa, ver a caixa abaixo.

Resultado aparente: **2 grupos**, estáveis, silhueta 0,491, zero ruído. Parecia achado.

Dois testes derrubam:

**Teste 1 — o grupo replica?** Dividir as respostas de cada aluno em duas metades independentes e reagrupar. Se o grupo for traço, o aluno cai no mesmo dos dois lados.

> **ARI = +0,176** (10.716 alunos com ≥50 respostas em cada metade).

ARI (índice Rand ajustado) vale 1 quando as duas partições coincidem e 0 quando a concordância é a do acaso. 0,176 é quase acaso: **o aluno cai em grupos diferentes conforme a metade usada.**

**Teste 2 — a silhueta era de quê?** A silhueta mede quão separados estão os grupos (−1 a 1).

| | silhueta |
|---|---|
| HDBSCAN, 2 grupos | **0,491** |
| k-means com k=2 no mesmo espaço | **0,491** |
| divisão puramente aleatória | −0,000 |

**Valores idênticos.** Os dois métodos estão bissectando uma nuvem contínua — é o que se obtém cortando uma gaussiana ao meio. Uma silhueta de 0,49 não prova estrutura; prova que o algoritmo cortou alguma coisa.

> ### Por que a transformação por quantis não é detalhe
>
> Trocando um passo do método — `StandardScaler` em vez de quantis, k-means em vez de HDBSCAN — o resultado vira **ARI +0,917 e silhueta 0,821**: grupos aparentemente sólidos e altamente replicáveis.
>
> São falsos. As features têm cauda pesada (constância de ritmo vai de 0,35 a 1,19 entre p10 e p90, com cauda muito além), e o k-means sobre escala bruta **separa a cauda de outliers do resto**. A separação entre os dois "grupos" é d = 3,17 em constância de ritmo e **d = 0,09 em trocar de alternativa** — um único eixo carregando tudo. E replica bem justamente porque ser outlier é traço estável.
>
> A transformação por quantis elimina o artefato, e aí a estrutura desaparece. Vale registrar porque é o caminho que produz exatamente os números que alguém gostaria de ver.

### A conclusão

**Alunos são altamente mensuráveis e nada categóricos** — contínuos em oito dimensões estáveis.

---

## 6. A síntese das três investigações

**Momento não informa; acumulado informa muito.**

| | resultado |
|---|---|
| estado momentâneo ([M1](../m1_estado_de_estudo/RELATORIO_M1.md): o aluno está travado agora?) | AUC ~0,51 — **não mede** |
| traço agregado (chuta? é lento? lê explicação?) | confiabilidade **0,70–0,98** |

Não é que comportamento não sirva. É que **a janela errada o destrói.** O M1 falhou porque perguntava do instante; estes traços funcionam porque perguntam do acumulado.

E o padrão se repete nos dois lados do dado:

| | estrutura discreta? | o que existe |
|---|---|---|
| questões | não (dificuldade unidimensional) | posição num contínuo |
| alunos | não (ARI 0,18) | vetor de 8 traços estáveis |

**Em nenhum dos dois casos as categorias existem.** O que existe são medidas contínuas confiáveis. As duas tentativas de impor categorias — o `k=4` das questões e os "2 grupos" de alunos — não sobreviveram ao teste de replicação.

---

## 7. Implicação para o TrailUp

Os **7 perfis BrainHex são categorias**, atribuídas por quiz declarado no signup. O dado diz que comportamento de aluno é contínuo e multidimensional.

Um vetor de traços medidos teria lastro empírico que o rótulo discreto não tem — e endereça a ameaça que o próprio TCC nomeia (`docs/tcc/02:84`): *"perfil declarado vs perfil efetivo de uso"*.

**Não é proposta de remover o BrainHex.** Ele organiza identidade visual, tom e narrativa do produto, e isso não depende de ter lastro preditivo. É proposta de **não usá-lo como se fosse medida** quando há medidas melhores disponíveis.

---

## 8. Limites

- **EdNet é TOEIC, autoestudo adulto, sem turma nem professor.** Traços como "lê explicação" e "chuta" devem existir em qualquer plataforma; suas **distribuições** não transferem.
- **A ausência de grupos pode ser do corpus.** Uma população escolar heterogênea (idades, matérias, contextos) tem mais chance de ter subgrupos reais do que usuários adultos autosselecionados de um app de TOEIC. O teste é barato e está pronto.
- **A raridade de questões defeituosas é do corpus** — e o TrailUp deve ter mais. Mas a **precisão de marcação** medida aqui (18–27%) é o que se deve esperar em prevalência baixa; com prevalência maior ela sobe, e o quanto é questão empírica.
- **`n_trocas` e a latência por questão não existem no TrailUp** hoje — dois dos oito traços exigiriam instrumentação (`answer_change` e tempo por questão).
- **Tudo observacional.** "Quem chutou aprende menos" é associação medida no reencontro, não efeito causal do chute.

---

## Apêndice — o que mudou, e por quê

**A afirmação "zero questões com discriminação negativa no EdNet" era bug de código.** A linha que contava era:

```python
for lim, lab in [(-1, 'NEGATIVA (aluno bom erra)'), (0.05, '...'), (0.15, '...')]:
    m = D.A < lim if lim < 0 else D.A < lim      # os dois ramos são idênticos
```

Com `lim = -1`, ela contava questões com discriminação **abaixo de −1**, que não existem por definição. Daí o "zero".

O número correto está na §3: **2,96%** numa metade dos alunos, **0,53% confirmadas nas duas**.

A conclusão de fundo não muda — o fenômeno é raro no EdNet e a marcação tem precisão de 18 a 27% — mas o relatório anterior afirmava "zero" num parágrafo e, no seguinte, que 18% das marcadas confirmavam. Eram duas afirmações incompatíveis lado a lado.

O cabeçalho de [`discriminacao.py`](../../trailup_core/discriminacao.py) carregava a mesma afirmação e foi corrigido junto.
