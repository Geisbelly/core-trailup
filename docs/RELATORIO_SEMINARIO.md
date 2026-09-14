# Relatório do processo — TrailUp Core e avaliação discursiva

## 1. Objetivo e alcance deste relatório

Este documento serve de apoio à apresentação do seminário: explica o problema,
os dados, os algoritmos, os experimentos, os resultados e o que foi entregue.
Apresenta o contexto do núcleo TrailUp, com detalhamento do avaliador de
respostas discursivas, que foi o último componente solicitado pelo grupo.

A referência de código desta revisão é o commit `9a6bed8`. Os experimentos
anteriores são descritos a partir dos relatórios e scripts versionados; não
foram repetidos durante a elaboração deste documento. Foram executados nesta
revisão o exemplo do avaliador e a suíte de testes, com **139 testes aprovados**.

É necessário distinguir três evidências:

| Evidência | O que demonstra |
|---|---|
| Resultado registrado em relatório de pesquisa | Desempenho relatado sob determinado protocolo e dataset |
| Execução de exemplo e teste de software | Comportamento concreto da implementação e dos casos testados |
| Avaliação com respostas reais do TrailUp | Qualidade no público e no conteúdo do produto; ainda pendente |

O PDF do seminário exige repositório público, apresentação de 15 minutos,
origem dos dados, comparação com baseline, diário e reprodução. Ele não exige
uma acurácia mínima. A enumeração “1 a 8” mencionada na conversa do grupo não
identifica, por si só, os módulos correspondentes; não atribuímos essa
numeração aos componentes sem uma lista explícita.

## 2. Problema real e motivação

O TrailUp organiza trilhas de aprendizagem adaptativas. Para decidir quando
oferecer reforço ou analisar uma resposta, precisa interpretar evidências
como respostas anteriores, propriedades das questões e atividade do aluno.

O objetivo do núcleo é calcular medidas locais que ajudem nessas decisões.
Uma parte consiste em fórmulas estatísticas; outra foi investigada com modelos
treinados. Ter um módulo com nome de modelo não significa que ele implemente
uma rede neural: o algoritmo precisa ser identificado pelo código.

Na parte discursiva, a pergunta foi: **é possível estimar a qualidade de uma
resposta comparando-a com o gabarito por relações linguísticas e semânticas?**
Uma estimativa rápida pode ajudar o professor a escolher o que revisar primeiro.
Reduzir chamadas à LLM é uma hipótese operacional; não houve medição de economia
em uma integração real do produto nesta entrega.

## 3. Contexto dos demais componentes

O [README do núcleo](../README.md) descreve as seguintes responsabilidades.
Esta tabela identifica funções, não uma sequência obrigatória de execução.

| Componente | Pergunta respondida | Abordagem principal |
|---|---|---|
| Dificuldade | Qual a taxa de acerto esperada da questão? | Estimativa estatística com intervalo |
| Ritmo | A questão costuma ser respondida rapidamente ou com elaboração? | Classificação apoiada no tempo observado |
| Tempo | Quanto deve demorar uma resposta? | Referência temporal por questão |
| Domínio | Qual a chance de acertar? | Combinação de evidência da questão e histórico do aluno |
| Chute | Há indício de tentativa rápida sem acerto? | Tempo relativo e quantidade de evidência |
| Revisão | Quando propor um reencontro? | Relação empírica entre intervalo e retenção |
| Discriminação | A questão diferencia desempenho de forma esperada? | Estatísticas e confirmação com evidência adicional |
| Gate | Há risco de dificuldade persistente no tópico? | Histórico e estimativa de risco |
| Engajamento | O aluno tende a continuar? | Recência e frequência calibradas por coorte |
| Pré-avaliação | Quais respostas discursivas revisar primeiro? | Comparação com o grafo do gabarito |
| Evasão | Quem pode abandonar? | Agregação longitudinal em SQL |

O módulo leve de domínio e os arquivos de gradient boosting do M2 são
implementações diferentes. Seus resultados não devem ser misturados. Da mesma
forma, prever acerto e prever permanência são tarefas distintas.

Os resultados e correções do restante do núcleo estão na
[trajetória do projeto](TRAJETORIA.md), na [auditoria](auditoria/AUDITORIA.md)
e no [relatório M2](m2_knowledge_tracing/RELATORIO_M2.md).

## 4. Dados usados no avaliador discursivo

Segundo o [relatório do experimento](discursiva/AVALIADOR_DISCURSIVO.md), foi
usado o classEx, distribuído no registro Zenodo `19467052`, arquivo
`GPT4FeedbackIncreasesStudentActivation.xlsx`, aba `Data`.

| Característica | Registro do experimento |
|---|---|
| Respostas | 1.167 |
| Agrupamentos de aluno relatados | 249; identificação exige a ressalva da seção 8 |
| Tarefas | 8 |
| Idioma e domínio | Alemão, macroeconomia, universitários |
| Entradas | Enunciado, gabarito e resposta |
| Alvo | Média de três notas de conteúdo atribuídas pelo GPT-4 |
| Escala original | 1 a 5 |
| Média e desvio da nota | Aproximadamente 3,50 e 0,70 |

O campo-alvo é `Content_Mean_Run1_2_3`. Aproximadamente 87% das notas
arredondadas estão em 3 ou 4. Há poucos exemplos nas pontas, o que dificulta
avaliar com precisão respostas excelentes ou muito ruins.

**A nota-alvo não é uma correção humana.** O treino busca aproximar o julgamento
registrado da LLM. Isso não demonstra que reproduz uma rubrica de professor.
O dataset é descrito nos documentos do projeto como CC BY; a ficha da fonte e
os termos do arquivo devem acompanhar qualquer redistribuição dos dados.

## 5. Preparação dos textos e construção do grafo

### 5.1 Seleção e transformação dos dados

No pipeline experimental, os campos são renomeados para `q`, `gab`, `resp` e
`nota`. Linhas sem resposta, gabarito ou nota são retiradas. Os textos são
convertidos em tokens: sequências de caracteres tratadas como palavras.

No módulo entregue, `_tokens()` usa `\w{3,}` e minúsculas. Isso seleciona tokens
com pelo menos três caracteres, incluindo caracteres de palavra Unicode.
Não há lematização, análise de dependência sintática ou identificação de
proposições. “Aumentar” e “aumenta” continuam sendo tokens diferentes.

Palavras de até dois caracteres não entram. Isso pode apagar distinções de
sentido; a tokenização não foi validada como análise linguística completa.

### 5.2 Stopwords

São palavras retiradas por aparecerem frequentemente no corpus. A intenção é
diminuir a influência de termos pouco discriminativos.

O módulo deriva stopwords por frequência de documento: por padrão, uma palavra
presente em mais de 40% dos textos é retirada. Com menos de dez documentos,
devolve um conjunto vazio. O corpus recomendado é o material do tópico.

Os scripts iniciais usam outra regra: retiram as 60 palavras mais frequentes.
Essa diferença é uma das razões pelas quais pipeline experimental e módulo
leve podem produzir resultados distintos.

### 5.3 Nós e arestas

Cada token mantido vira um nó, com peso igual à sua frequência. Pares de tokens
próximos viram arestas não direcionadas, ponderadas pelo número de ocorrências.
Com `JANELA=4`, o código compara cada token com os três seguintes: uma janela
de quatro posições contando o token inicial.

Exemplo conceitual, após tokenização e filtragem:

```text
Gabarito: juros → reduzem → consumo
Resposta: juros → aumentam → consumo
```

Os grafos compartilham “juros” e “consumo”, mas as afirmações têm sentidos
opostos. Como as arestas representam proximidade e não causalidade formal,
compartilhar uma relação no grafo não prova que a resposta está correta.

`preparar()` calcula o grafo do gabarito uma vez. Depois, `avaliar()` reutiliza
essa referência para cada resposta. Os conceitos centrais são escolhidos pelo
grau ponderado: palavras ligadas a mais ocorrências de relações recebem destaque.

## 6. Como o módulo calcula a estimativa

O módulo usa dez características numéricas. “Conceito” aqui significa token
selecionado pelo procedimento, não uma entidade semântica identificada por um
especialista.

| Característica | Cálculo ou significado |
|---|---|
| `no_peso` | Cobertura dos tokens do gabarito ponderada por frequência |
| `no_cobertura` | Fração dos tokens distintos do gabarito encontrados na resposta |
| `cob_conceitos_centrais` | Fração dos tokens centrais encontrados |
| `divagacao` | Fração dos tokens distintos da resposta ausentes do gabarito e do enunciado |
| `razao_tam` | Quantidade de tokens distintos da resposta dividida pela do gabarito |
| `aresta_cobertura` | Fração das arestas do gabarito também presentes na resposta |
| `aresta_peso` | Sobreposição ponderada das arestas |
| `log_len` | Logaritmo de um mais o comprimento da resposta em caracteres |
| `n_nos` | Quantidade de tokens distintos da resposta |
| `n_arestas` | Quantidade de pares distintos ligados na resposta |

O campo `divagacao` tem nome mais forte que sua interpretação matemática:
sinônimos, exemplos válidos e reformulações também aumentam esse valor.
Portanto, ele não mede literalmente quanto o aluno fugiu do assunto.

Os coeficientes estão fixados em `_PESOS`, com origem descrita como regressão
ridge nos comentários do módulo. A ridge ajusta uma soma ponderada para
aproximar notas, penalizando coeficientes muito grandes. Na inferência:

```text
nota_bruta = intercepto + soma(peso_j × característica_j)
nota = limitar(nota_bruta, mínimo=1, máximo=5)
```

As características têm correlação entre si. Um coeficiente positivo isolado
não implica que aumentar aquela propriedade seja pedagogicamente melhor.

O módulo [pre_avaliacao.py](../trailup_core/pre_avaliacao.py) usa apenas a
biblioteca padrão. Ele não carrega um encoder nem treina uma regressão durante
`avaliar()`.

## 7. Hipóteses e experimentos anteriores

Os resultados abaixo são **valores históricos registrados** no
[relatório discursivo](discursiva/AVALIADOR_DISCURSIVO.md). Não são novas
medições produzidas para este relatório.

### 7.1 Linha de base e comparação de representações

Baseline é uma solução simples usada para verificar se a complexidade adicional
traz ganho. Foram comparados comprimento, similaridade textual e grafos.

TF-IDF representa textos por pesos de palavras: termos frequentes num texto,
mas menos comuns no corpus, ganham importância. O cosseno compara a direção
desses vetores. LSA reduz a dimensionalidade por decomposição da matriz e tenta
capturar associações entre palavras.

| Variante experimental | Spearman | QWK | MAE na escala 1–5 |
|---|---:|---:|---:|
| Comprimento | 0,253 | 0,219 | 0,554 |
| Similaridade e comprimento | 0,401 | 0,302 | 0,511 |
| Grafo: nós | 0,482 | 0,387 | 0,481 |
| Grafo: nós e arestas | 0,488 | 0,388 | 0,478 |
| Grafo e semântica LSA | 0,488 | 0,391 | 0,479 |
| Grafo e similaridade | 0,501 | 0,415 | 0,471 |

Adicionar arestas trouxe diferença pequena, de 0,006 de Spearman. A maior
parte do sinal estava na cobertura dos tokens do gabarito. Isso sustenta a
utilidade da representação, mas enfraquece a hipótese de que relações de
coocorrência bastariam para julgar a correção do argumento.

### 7.2 Enriquecimento do gabarito

Foram incorporadas respostas bem avaliadas do treino à referência. Na rodada
correspondente, usar todas essas respostas reduziu Spearman de 0,486 para
0,396. A interpretação proposta é que a referência ficou genérica demais.
Esse resultado se aplica à estratégia testada, não a toda forma de usar
conhecimento externo.

### 7.3 Mais exemplos

A curva de aprendizagem registrada mostra ganho de aproximadamente 0,030 ao
dobrar de 466 para 933 exemplos. Isso indica retorno modesto nessa faixa;
não prova que dados novos, de maior diversidade ou com rubrica humana, seriam
incapazes de melhorar o resultado.

### 7.4 Encoder semântico

Foi testado `paraphrase-multilingual-MiniLM-L12-v2`. O encoder transforma textos
em vetores que permitem comparar formulações semanticamente semelhantes.
Além do cosseno dos textos inteiros, o script divide resposta e gabarito em
frases e calcula, para cada frase do gabarito, a melhor correspondência na
resposta. Produz médias, mínimos e frações acima de um corte de similaridade.

Um `HistGradientBoostingRegressor` combina essas características. No script
`06_encoder.py`, são configuradas até 300 iterações, taxa de aprendizado 0,05,
profundidade máxima 3, mínimo de 30 exemplos por folha e regularização L2 de 1.
O boosting combina árvores que corrigem progressivamente erros da previsão.

| Variante | Spearman | QWK | MAE |
|---|---:|---:|---:|
| Grafo de superfície | 0,488 | 0,388 | 0,478 |
| Encoder e cobertura por frase | 0,476 | 0,400 | 0,480 |
| Grafo e encoder | 0,532 | 0,433 | 0,450 |

As representações foram complementares. O melhor valor relatado, 0,532,
pertence ao experimento com encoder, não ao módulo leve entregue.

## 8. Validação e ressalvas encontradas nos scripts

### 8.1 Protocolo pretendido

`GroupKFold(5)` separa cinco grupos de avaliação. Em cada rodada, o estimador
aprende com quatro partes e prevê a restante. O objetivo de agrupar por aluno
é impedir que o mesmo aluno esteja nos dois lados.

`LeaveOneGroupOut`, usando `Task`, testa uma tarefa de cada vez sem suas
respostas no ajuste do regressor. O relatório registra queda para Spearman
0,429 na combinação de grafo e encoder. Esse teste é mais próximo da pergunta
“funciona para uma questão nova?”, embora tenha as ressalvas abaixo.

### 8.2 Limites concretos da implementação experimental

A inspeção de `01_grafos.py` e `02_avaliar.py` mostrou que:

1. Stopwords, TF-IDF e LSA são ajustados no corpus completo antes da validação
   do regressor. Isso permite que textos reservados à avaliação influenciem a
   representação. Não usa suas notas diretamente, mas impede descrever o
   protocolo como uma avaliação completamente isolada de textos inéditos.
2. A chave de aluno concatena dois pseudônimos após `astype(str)`. Ausências
   podem virar a string `nan_nan`; o `fillna()` posterior não corrige strings.
   A contagem de grupos não equivale automaticamente a alunos identificados
   de maneira confiável, e colisões de pseudônimos precisam ser verificadas.
3. Alguns resultados são calculados sobre as mesmas tarefas usadas em outras
   rodadas de investigação. Escolher configurações a partir dessas medições
   exige um conjunto final reservado para uma estimativa independente.

Assim, o desenho de validação é relevante, mas os números históricos devem
ser apresentados como exploratórios. Uma reprodução mais rigorosa precisa
ajustar todo pré-processamento dentro de cada dobra e auditar os agrupamentos.
Este relatório registra esses problemas; não altera os scripts nem afirma
que essa nova validação já foi executada.

### 8.3 Significado das métricas

**Spearman** mede concordância da ordenação: respostas melhores recebem scores
maiores? 0,532 não significa 53,2% de acertos nem 53,2% de notas corretas.

**QWK** mede concordância entre notas discretizadas, penalizando diferenças
grandes. **MAE** é a média dos erros absolutos na unidade da nota: 0,450 é um
erro médio de 0,45 ponto na escala 1–5.

O relatório chama a concordância de aproximadamente 0,88 entre execuções da
LLM de “teto”. A descrição mais precisa é **referência de concordância entre
avaliadores automáticos**. Ela não estabelece um limite matemático universal
para prever a média das avaliações ou uma futura nota humana.

## 9. O que foi implementado na última entrega

O commit `9a6bed8` adicionou percentual e faixa, mudou a representação textual,
acrescentou um teste e documentação. Não houve novo treino discursivo nem
alteração dos pesos nesse commit.

```text
percentual_estimado = arredondar((nota − 1) / 4 × 100, uma casa decimal)
```

| Nota | Percentual da escala | Faixa retornada |
|---|---:|---|
| 1 | 0% | baixo |
| 2 | 25% | baixo |
| 3 | 50% | medio |
| 4 | 75% | alto |
| 5 | 100% | alto |

Os cortes são aplicados ao percentual arredondado: abaixo de 50, de 50 até
menos de 75, e a partir de 75. São convenções de exibição, não fronteiras de
aprovação validadas pedagogicamente. `faixa_percentual` não é um intervalo de
confiança. Não foi implementada uma estimativa de confiança estatística.

Essa transformação preserva essencialmente a ordenação, com possíveis empates
pelo arredondamento. Não melhora o erro do modelo. Também não permite
interpretar “75%” como três quartos dos argumentos corretos.

`conceitos_faltando` já existia e lista tokens centrais não encontrados por
igualdade textual. Um sinônimo válido pode aparecer como ausência. `triar()`
ordena respostas da menor nota prevista para a maior; não chama a LLM e não
implementa o roteamento de respostas para revisão externa.

## 10. Demonstração executada nesta revisão

Na raiz do repositório:

```bash
python3 -m trailup_core.pre_avaliacao
python3 -m pytest tests/ -q
```

O exemplo trata da relação entre juros, consumo, investimento e inflação.
Resultado observado, arredondado pela saída textual:

| Resposta do exemplo | Percentual da escala | Cobertura literal do gabarito |
|---|---:|---:|
| Completa | 88% | 70% |
| Parcial | 42% | 25% |
| Fora do tema, identificada como “divagando” | 48% | 10% |
| “Não sei responder” | 4% | 0% |

O modelo coloca a resposta completa acima da vazia, mas também coloca a
resposta fora do tema acima da parcial. Esse erro deve ser mostrado: o modelo
é sensível a características de superfície, como comprimento, e pode produzir
uma ordenação inadequada mesmo num exemplo simples.

Os **139 testes passaram**. Eles verificam comportamento de software; não
demonstram validade de uma nota automática em português.

## 11. Histórico de trabalho e correções de interpretação

O repositório já continha investigações e módulos antes das últimas entregas.
O [diário discursivo](discursiva/DIARIO.md) é uma síntese retrospectiva desses
experimentos. Não deve ser apresentado como evidência de que cada experimento
foi executado novamente na sessão que adicionou o percentual.

No M2, primeiro foi executado um piloto com 2.000 arquivos selecionados,
448 alunos elegíveis e 103.119 respostas finais. A conversa registrou AUC
0,7001 para próximo acerto e 0,7074 para o gate. Depois houve uma execução
maior, selecionando 120.000 arquivos, que gerou dois artefatos e motivou o
commit `5379965`.

**Correção necessária sobre os resultados do M2:** os números 0,7538 e 0,7795
informados na conversa após esse treino também constavam no relatório
histórico. As saídas de métricas da execução maior não foram preservadas no
registro visível, e o script imprime essas métricas sem salvá-las em JSON.
Portanto, não podemos atribuir esses valores aos novos arquivos apenas porque
eles foram gerados. É necessário reavaliar os artefatos com o split
correspondente e guardar os resultados vinculados aos seus hashes.

O mesmo cuidado vale para chamar o material de “produção”: gerar um `.pkl`,
versioná-lo e passar testes não comprova integração, calibração local,
desempenho operacional ou validação em alunos do TrailUp.

## 12. Reprodução: o que já roda e o que falta

A demonstração do módulo leve pode ser executada pelos comandos da seção 10.
O projeto declara as dependências de desenvolvimento em
[pyproject.toml](../pyproject.toml). Para instalar o pacote com os testes,
pode-se usar `python3 -m pip install -e '.[dev]'` em um ambiente virtual.
Nesta revisão, o ambiente já tinha pytest e não foi reinstalado.

Para reproduzir a pesquisa discursiva, a cadeia principal é:

```text
Planilha classEx, aba Data
  → 01_grafos.py → classex_grafo.parquet
  → 02_avaliar.py → classex_pred.parquet
  → 06_encoder.py → classex_emb.parquet
  → 07_operacao.py → métricas de triagem
```

Os scripts de enriquecimento e curva de aprendizagem registram experimentos
adicionais. A cadeia acima descreve as dependências de arquivos, não garante
execução imediata após o clone.

As [instruções dos scripts](LEIA-ME_scripts.md) avisam que existem caminhos
fictícios. A pesquisa discursiva requer NumPy, pandas, scikit-learn, SciPy,
PyArrow, leitor Excel como openpyxl e, no experimento semântico,
sentence-transformers com suas dependências e o modelo baixado.

Para atender integralmente ao requisito de reprodução científica do seminário,
ainda é preciso parametrizar os caminhos discursivos, registrar versões exatas,
origem e checksum do dataset, corrigir o protocolo descrito na seção 8 e salvar
métricas e configurações da execução. O módulo demonstrável está disponível;
a reprodução completa da pesquisa não foi concluída pela conversão para percentual.

## 13. Limitações e próximos passos verificáveis

| Limitação | Próxima verificação necessária |
|---|---|
| Alvo gerado por GPT-4 | Comparar com notas humanas e rubrica explícita |
| Alemão e macroeconomia | Avaliar respostas reais em português e novos conteúdos |
| Poucas notas extremas | Ampliar diversidade e reportar erro por faixa de nota |
| Grafo perde sentido e direção | Testar negação, contradição, paráfrase e causalidade invertida |
| Pré-processamento observa corpus inteiro | Ajustá-lo apenas nos dados de treino de cada dobra |
| Agrupamento por pseudônimo | Auditar ausências e colisões antes de dividir os dados |
| Percentual pode sugerir precisão inexistente | Explicar a escala e medir erro antes de usá-la como nota |
| Falta de confiança estatística | Calibrar intervalos ou abstinência em conjunto separado |
| Limiares de triagem sem validação local | Medir erros e custo de revisão com dados do TrailUp |

A sugestão anterior de coletar cerca de 100 respostas humanas é um ponto de
partida operacional, não uma garantia de tamanho suficiente. A amostra deve
cobrir alunos, tarefas e níveis de qualidade; sua adequação depende da
incerteza observada.

## 14. Roteiro para apresentação de 15 minutos

| Tempo | Conteúdo e demonstração |
|---|---|
| 0–3 min | Problema do TrailUp e finalidade da triagem discursiva |
| 3–6 min | classEx: enunciado, gabarito, resposta, origem da nota e distribuição |
| 6–10 min | Tokens, grafo, características, regressão; comparação com baselines e encoder |
| 10–13 min | Tabelas de resultados históricos e execução do exemplo atual |
| 13–15 min | Erro da resposta fora do tema, limites da escala e pendências de validação |

Na demonstração, mostrar `preparar()`, `avaliar()`, `percentual_estimado` e
`triar()`. Explicar o caminho de uma resposta até a saída. Não é necessário
ler os dez coeficientes nem explicar toda a biblioteca scikit-learn.

## 15. Perguntas esperadas

**“Isso corrige uma resposta em porcentagem?”** Produz um score na escala
0–100 a partir de uma nota prevista. Ainda não mede uma fração validada de
conteúdo correto.

**“Onde está o aprendizado de máquina?”** Nos pesos ajustados da regressão e
nos experimentos de boosting. A construção do grafo e a transformação de
escala são procedimentos determinísticos.

**“Por que não usar só similaridade?”** Na pesquisa registrada, cobertura do
gabarito acrescentou sinal em relação ao baseline. Similaridade elevada,
entretanto, também pode ocorrer numa afirmação contraditória.

**“O modelo entende semântica?”** O experimento com encoder usa representação
semântica aprendida. O módulo leve implementa coocorrência e cobertura literal;
nenhum dos dois foi demonstrado como verificador confiável da verdade de um
argumento.

**“Qual resultado podemos apresentar?”** 0,532 de Spearman é o resultado
histórico relatado para grafo mais encoder, sujeito às ressalvas de validação.
O cabeçalho do módulo leve registra 0,501 em sua auditoria específica. Os
139 testes e os valores do exemplo foram verificados nesta revisão. São
evidências diferentes e devem ser identificadas dessa forma.

**“A limitação foi corrigida?”** A apresentação da saída foi ajustada e o uso
foi restringido à triagem. A limitação de previsão não foi eliminada. Corrigi-la
exige novos experimentos e validação; não basta converter a nota para percentual.

## 16. Fontes internas para a apresentação

- [Código do avaliador](../trailup_core/pre_avaliacao.py).
- [Experimentos, resultados históricos e limitações](discursiva/AVALIADOR_DISCURSIVO.md).
- [Construção de grafos](discursiva/scripts/01_grafos.py).
- [Baselines e validação do regressor](discursiva/scripts/02_avaliar.py).
- [Experimento com encoder](discursiva/scripts/06_encoder.py).
- [Experimento de triagem](discursiva/scripts/07_operacao.py).
- [Diário retrospectivo](discursiva/DIARIO.md).
- [Relatório do knowledge tracing](m2_knowledge_tracing/RELATORIO_M2.md).
- [Script de treino M2](m2_knowledge_tracing/scripts/15_treina_v3.py).
- [Histórico geral](TRAJETORIA.md) e [auditoria](auditoria/AUDITORIA.md).
- [Testes do pacote](../tests/test_core.py).
