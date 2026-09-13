# Lista de modelos — construídos, descartados e na fila

Estado em 2026-09-13. Tudo medido sobre os datasets de referência; o banco do TrailUp está vazio.

---

## Construídos e no módulo

Implementações em [`modulo/`](../README.md), sem dependência.

| Modelo | Medida | Custo |
|---|---|---|
| dificuldade da questão (intervalo) | cobertura nominal; ±11 pts com 30 respostas | 30 alunos |
| ritmo da questão | 98% de acerto | 5 alunos |
| tempo esperado de resposta | R² 0,571 | 5 alunos |
| domínio no tópico | AUC 0,721 (regra atual 0,586) | histórico |
| detecção de chute | traço, +0,922 | 50 respostas |
| curva de esquecimento / revisão | 585 mil reencontros | nenhum |
| discriminação (questão quebrada) | pronto; ver ressalva | 40 respostas |
| risco de evasão (SQL) | lift 3,4× no top 10% | semanas |
| engajamento (3 eixos) | AUC 0,805 para retenção em 30 dias | 30 respostas |
| pré-avaliação rápida de resposta | Spearman 0,447 (teto ≈0,88) | nenhum |

### Nível de engajamento → [`engajamento.py`](../trailup_core/engajamento.py)

**Pergunta:** este aluno ainda vai estar estudando daqui a um mês? (Não: "vai aprender mais" — ver o limite abaixo.)

**Desenho:** EdNet KT3, 18.119 alunos. Eixos calculados nos **dias 0–29** desde a primeira resposta do aluno; desfecho observado nos **dias 30–59** — deu ao menos uma resposta, sim ou não. Nada da segunda janela entra no cálculo. Taxa base: 30,5% voltaram.

**As medidas** (contagens sobre o que a plataforma já registra, nada inferido):

| eixo | conta | AUC | IC 95% | direção |
|---|---|---|---|---|
| frequência | dias ativos ÷ 30 | **0,799** | [0,792–0,805] | mais dias → **mais** volta |
| volume | respostas ÷ sessões (gap > 30 min) | **0,400** | [0,392–0,409] | mais por sessão → **menos** volta |
| profundidade | log(1 + s lendo explicação após errar) | **0,624** | [0,616–0,633] | mais tempo → **mais** volta |

AUC = chance de ordenar corretamente um par voltou/não-voltou; 0,50 é cara ou coroa, abaixo de 0,50 prediz ao contrário. Referências: volume bruto de respostas 0,721; taxa de acerto 0,513 (**desempenho não prevê permanência**).

**Três números e não um** porque os eixos correlacionam |r| ≤ 0,33 entre si e o volume aponta ao contrário: uma média empataria o maratonista (retenção 19,5%) com o constante (81,4%).

**O volume negativo não é composição.** Retenção por faixa de volume dentro de cada faixa de frequência cai em todas as linhas (13,1→8,6 / 28,6→23,1 / 62,1→56,5). Sessão longa e isolada é maratona de quem tenta compensar atraso.

**Predição vs diagnóstico:** só a frequência já dá AUC 0,800 fora do treino; os três dão 0,803. Volume e profundidade quase não ajudam a saber *quem* está em risco — servem para saber *o que* está acontecendo, que é o que decide a intervenção. Calibração: ECE 0,031. Estabilidade entre metades: +0,88 a +0,99.

**Limite medido:** nenhum eixo prevê ganho de acerto (Spearman −0,04 a +0,03). Mede permanência, não aprendizado — não serve como indicador de progresso.

Corrige duas falhas da rodada anterior: `persistencia` era circular (dividida pelo próprio volume, r=0,94 por construção algébrica) e o desfecho "voltou depois da metade do próprio período" dava positivo para **100%** dos alunos. Relatório: [`engajamento/RELATORIO_ENGAJAMENTO.md`](engajamento/RELATORIO_ENGAJAMENTO.md).

### Pré-avaliação rápida de resposta → [`pre_avaliacao.py`](../trailup_core/pre_avaliacao.py)

Medido em [AVALIADOR_DISCURSIVO.md](discursiva/AVALIADOR_DISCURSIVO.md):

| Versão | Spearman | Latência | Dependência |
|---|---|---|---|
| superfície (grafo + similaridade) | 0,488 | microssegundos | **nenhuma** |
| + encoder multilíngue | 0,532 | ~50 ms + carga do modelo | torch (~900 MB) |
| *(teto: execuções independentes do LLM)* | *≈0,88* | segundos | API externa |

Para "rápida", a versão sem dependência é a certa: roda no mesmo ciclo da requisição e dá **feedback instantâneo enquanto a LLM trabalha**.

Ponto de operação medido: cortar em 10%/90% dispensa a LLM em 20% dos casos, errando 8% na ponta alta e 12% na baixa. E a triagem funciona melhor que a pontuação — as 10 respostas de menor previsão têm nota real média 2,33 contra 3,50 do geral.

Enviada a versão **linear, sem dependência**: Spearman 0,447 contra 0,482 do boosting sobre as mesmas features. Perde pouco em ordenação, que é o uso; perde mais em nota absoluta (QWK 0,325 contra 0,397), que não é.

Dois cuidados embutidos no código:

- **Os coeficientes foram ajustados em alemão.** As features são agnósticas de idioma (operações de conjunto sobre tokens); os pesos não. Até haver ~100 respostas em português com nota humana, usar só para **ordenar** — ordenação é bem mais robusta a peso errado que nota absoluta.
- **`stopwords()` exige corpus.** A primeira versão derivava das 60 palavras mais frequentes de dois textos curtos — o que transformava quase todo o vocabulário em stopword e deixava o grafo de referência **vazio**, fazendo tudo pontuar igual (a resposta que só divagava tirava 5,0). Agora usa frequência de documento e devolve conjunto vazio com menos de 10 textos.

---

## Descartados, com o motivo

### Empenho do aluno (regressão) — **testado, não passou** → [relatório](empenho/RELATORIO_EMPENHO.md)

Esforço medido como **resíduo** de `log(tempo na resposta) ~ tempo esperado da questão + dificuldade + habilidade do aluno`. Sem inventar rótulo.

A medida é boa: **traço estável (+0,920)**, não é capacidade disfarçada (correlação −0,144 com acerto) e correlaciona +0,456 com ler a explicação após errar.

**Mas não prediz aprendizado**, que era a única razão para construí-lo — o engajamento já mede permanência:

| desfecho | resultado |
|---|---|
| acertar ao reencontrar a mesma questão | AUC **0,516** (dificuldade da questão: 0,584) |
| ganho de acerto entre metades | +0,044 bruto, **−0,025** controlando o acerto inicial |

O +0,044 era regressão à média. E exigiria tempo por questão, que o TrailUp não coleta.


| Descartado | Por quê |
|---|---|
| inferência de estado emocional (M1) | AUC 0,505 (IC contém 0,50); e a "base pessoal" que parecia salvar o slot é estilo de resposta — AUC intra-aluno 0,379 |
| agrupamento de alunos em perfis | grupos não replicam (ARI 0,18); traços individuais sim (0,70–0,98) |
| categorias de dificuldade por habilidade | unidimensional no corpus (HDBSCAN: 0 grupos) |
| roteador de intenção do chat | circular — 88% viraram 38% ao remover o template |
| análise de frames de câmera | zero consumidores, zero dataset, biometria de menor |

---

## Na fila

### 1. Sugestão de pontuação — **não testado; provavelmente não é modelo**

Quanto vale uma questão na gamificação. Hoje o TrailUp tem `eventos_pontuacao` (tipo → pontos), fixo e ajustável sem migration.

Os insumos já existem e estão medidos: **dificuldade** (raridade do acerto) e **tempo esperado** (esforço). Combiná-los em pontos é uma fórmula de produto, não algo a aprender — não há rótulo de "pontuação correta" em dataset nenhum.

O que **daria** para medir, se houver dado próprio: se a pontuação afeta comportamento (o aluno busca questão que vale mais?). Isso exige experimento no produto, não dataset externo.

> **Recomendação:** fórmula explícita `pontos ∝ f(dificuldade, tempo)` com os valores que já medi, e A/B depois. Tratar como modelo dá falsa impressão de que foi aprendido de dado.

## Ainda abertos, de listas anteriores

| Item | Estado |
|---|---|
| dificuldade pelo texto do enunciado | inconclusivo — r = −0,929 mas com n=8 tarefas e 10 features |
| avaliador discursivo como substituto da LLM | descartado; serve como pré-filtro (item 3 acima) |
| questão defeituosa | construído, mas EdNet não tem o fenômeno — validar no TrailUp |

## Duas lacunas de instrumentação que destravam quatro módulos

1. **Tempo por questão** — `chute.py` e `tempo.py` dependem, e o TrailUp não coleta.
2. **Questões personalizadas gravarem em `questao_aluno`** — `discriminacao.py` e o corpus de dificuldade dependem; hoje não gravam (`QuestionActivity.tsx:1281`).
