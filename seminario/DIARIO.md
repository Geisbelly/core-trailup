# Diário de decisões

Uma entrada por sessão de trabalho. O que foi tentado, e o que aconteceu —
inclusive quando não deu certo, que é a maior parte.

> **Nota para o grupo:** as datas abaixo são as das sessões reais de trabalho.
> Ajuste se alguma estiver errada. O que não pode mudar é a ordem, porque cada
> decisão só faz sentido depois da anterior.

---

## 28/08 — escolha do problema

Queríamos algo com dado real e desfecho verificável, não classificação de
sentimento em review de app. Ficamos com: **prever se um aluno vai parar de
estudar**, olhando só o comportamento das primeiras semanas.

Por que esse: o desfecho existe no dado sem ninguém precisar rotular. Se o
aluno tem atividade nos dias 30–59, ele voltou. Não há anotação humana, não há
julgamento nosso, não há vazamento de rótulo.

Primeira dúvida registrada: quantos dias de janela? Escolhemos 30 para medir e
30 para verificar, mas isso é arbitrário e não testamos alternativas. Fica como
limitação.

---

## 30/08 — um dataset de emoções que descartamos

Apareceu o **WESAD** — sinais fisiológicos (batimento, condutância da pele)
rotulados por estado emocional. A ideia era medir foco e interesse direto na
fonte, em vez de inferir por comportamento.

Descartado, por três motivos que valem registrar:

1. São **15 sujeitos** de laboratório. Não dá para falar de generalização.
2. O rótulo é estado emocional induzido em sessão controlada, não engajamento
   com estudo ao longo de semanas. É outro fenômeno.
3. Nenhum aluno nosso vai usar pulseira. Um modelo que depende de sensor que o
   produto não tem não é um modelo, é um exercício.

Lição: **a disponibilidade do sinal em produção é critério de seleção de
feature, não detalhe de implementação.**

---

## 02/09 — primeira rodada, e dois defeitos nossos

Painel montado só com o **EdNet KT3** (autoestudo de inglês, coreano, adulto).
Quatro eixos: frequência, volume, profundidade, persistência.

AUC alta de cara. Boa demais. Fomos olhar e achamos dois erros **nossos**:

- **`persistência` era circular.** Ela era dividida pelo próprio volume, então
  correlacionava 0,94 com ele por álgebra, não por fenômeno. Estávamos medindo
  a mesma coisa duas vezes e chamando de dois eixos.
- **O desfecho estava errado.** "Voltou depois da metade do próprio período"
  dava positivo para **100% dos alunos** — por construção, todo mundo tem
  atividade depois da metade do próprio período de atividade. Não havia o que
  prever.

Refeito com janela **fixa** (dias 0–29 medem, dias 30–59 verificam), igual para
todo mundo, e sem a feature circular.

Lição que ficou: **quando a métrica vem boa demais logo de cara, o erro está no
desenho, não na sorte.**

---

## 05/09 — decisão: uma base só não prova nada

Com o EdNet arrumado, o modelo dava AUC ~0,80. A pergunta que travou a sessão:
isso é sobre **engajamento** ou sobre **o EdNet**?

Decisão do grupo: **trazer uma segunda base de realidade deliberadamente
oposta.** Entrou o **OULAD** — universitário britânico, EAD, com turma,
professor, matrícula e abandono formal registrado. O oposto de autoestudo
voluntário coreano.

Critério que estabelecemos aqui e usamos até o fim:

> Um eixo só entra no modelo se **as duas bases concordarem com ele**.

Não é sobre ter mais dados. É sobre ter dados que possam **discordar**.

---

## 06/09 — o volume morre, e a recência aparece

Rodando cada eixo separado nas duas bases:

O **volume** (eventos por sessão) era o achado mais interessante que tínhamos —
sinal negativo forte no EdNet, do tipo "maratona é sinal de abandono".
Sobrevivia até a controles por quintil. Na segunda base **virou ruído**.

Foi removido. Dói remover o resultado mais bonito, mas um efeito que existe numa
realidade e some na outra é **propriedade daquela base**, não achado sobre
engajamento.

No lugar entrou um eixo que não estava no plano: **recência** — dias ativos nos
**últimos 10** dias da janela, não nos 30. Bate frequência nas duas bases.

Leitura: *"apareceu ultimamente"* vale mais que *"aparece bastante"*.
Consequência de produto: recalcular perto da decisão, não uma vez por mês.

---

## 09/09 — hipótese nossa, e ela estava errada

Achávamos que converter os eixos para **percentil** resolveria a diferença entre
as bases — mesma escala, mesmo significado.

Não resolveu. AUC igual, e a calibração **piorou**. O motivo, que só entendemos
depois: normalizar a *feature* não corrige diferença na *prevalência do
desfecho*. As duas bases não diferem só na escala do sinal; diferem em quantos
alunos voltam (13,8% contra 91,5% de retenção nos nossos cortes).

Registrado porque foi uma tarde inteira gasta numa hipótese razoável e errada.

---

## 10/09 — juntar as bases, e a armadilha

Decisão do grupo: em vez de um modelo por base, treinar **um modelo só nas duas
juntas**. E, antes de treinar, uma instrução explícita que virou o teste mais
importante do trabalho:

> **normalizar os datasets e juntar num só de treino, antes de treinar o
> modelo, para evitar essa adivinhação.**

Testamos a adivinhação de propósito: *dá para saber de qual base veio a linha,
olhando só os eixos?*

Dá — e muito bem. Com os valores absolutos, **AUC 0,92** para identificar a
base. Ou seja: um modelo treinado no conjunto cru aprenderia a **reconhecer a
plataforma** e aplicar a taxa de retenção dela. A métrica subiria e ele não
estaria medindo engajamento nenhum.

Padronizando **dentro de cada coorte** antes de juntar, a identificação cai para
o acaso. Aí sim o modelo único é legítimo.

Esse teste não estava em tutorial nenhum. Saiu da desconfiança de que juntar
bases diferentes é bom demais para ser de graça.

**Addendum de 13/09, e ele nos contraria em parte.** Ao medir o *custo* da
versão crua neste painel, o estrago em AUC foi **pequeno** — uma a três
milésimas. Esperávamos muito mais. O que de fato aparece é nos **pesos**: no
conjunto cru eles ficam cerca de **15× maiores** (9,49 contra 0,61), porque a
única forma de uma logística separar duas bases de escalas diferentes é esticar
o coeficiente até a distância entre elas virar decisão.

Então a defesa da padronização mudou de formato: não é "senão a métrica
despenca" — não despencou aqui. É que **o modelo cru chega ao resultado por um
caminho que não é o declarado**, e um peso de 9,5 numa feature de duas não é
interpretável por ninguém. Padronizar custa uma linha e remove as duas
objeções.

Registrado assim porque a versão anterior desta entrada afirmava um custo
grande em AUC, com base no que tínhamos visto num painel anterior. Neste, não
se confirma.

---

## 11/09 — a divisão sortuda

Tínhamos anotado a AUC do modelo a partir de **uma** divisão treino/teste.
Refizemos com **12 divisões** diferentes e comparamos.

O número que tínhamos estava quase **dois desvios acima da média** das 12. Não
era um modelo melhor: era uma divisão favorável, e nós a havíamos copiado para
todos os lugares onde o resultado aparecia.

Regra adotada daí em diante: **todo número que vai para o relatório vem de pelo
menos 12 partições, citado como média ± desvio.** Uma divisão só serve para
*comparar* duas opções lado a lado, nunca para declarar o nível de nada.

---

## 12/09 — a base onde o modelo não funciona

Testamos numa terceira base (**ARES**, 200 universitários, leitura obrigatória
em segunda língua, 6 semanas). AUC entre 0,49 e 0,57 em todos os eixos — ou
seja, nada.

Não é defeito do modelo. Naquela base **87,7% voltam** porque a participação é
**obrigatória**. Quando quase todo mundo volta, não há variação para prever.

Virou escopo declarado, não nota de rodapé: **este modelo pressupõe continuação
voluntária.**

---

## 13/09 — três defeitos no nosso próprio pipeline

Ao reescrever tudo neste repositório, do zero e reprodutível, achamos três
erros que estavam escondidos no código anterior:

1. **O teste de replicação do `volume` estava comparando a coluna com ela
   mesma.** O OULAD registra o *dia*, não a *hora* — sem hora não dá para
   cortar sessão, então `volume` não é calculável ali. A primeira versão do
   script preenchia com `profundidade` para "não ficar vazio". O teste passava
   sem medir nada. Agora fica `NaN`, e o eixo é reprovado pelo motivo certo:
   **metade das realidades-alvo não consegue nem calcular.**

2. **O teste de vazamento não podia detectar nada.** Comparávamos "cru contra
   padronizado" treinando numa base e testando na outra. Dava o mesmo número
   nos dois casos, e a razão é aritmética: padronizar dentro da coorte é uma
   transformação **monótona** dentro de cada coorte, e AUC só olha ordem. O
   teste era vazio por construção. O efeito aparece onde o dano acontece — no
   treino **conjunto**, onde os coeficientes se acomodam à distância entre as
   bases. Refeito assim.

3. **Acurácia não serve para este problema, e agora temos o número.** No OULAD,
   91,5% dos alunos voltam. "Nunca alertar ninguém" acerta 91,5% das vezes.
   Qualquer modelo que reporte acurácia aqui está escondendo que não faz nada.
   Passamos a usar **ponto de operação** — alertar os X% de maior risco, com o
   limiar saindo dos dados daquela coorte.

---

## 14/09 — um segundo modelo entra, e ele quase foi descartado por erro nosso

Decisão: o trabalho leva **dois modelos**, não um. O segundo avalia resposta
discursiva — compara o grafo de coocorrência da resposta com o do gabarito.
O que une os dois é a mesma pergunta de produto: **onde gastar uma chamada
cara**. Um decide em quem vale intervir; o outro, qual resposta precisa mesmo
de correção por LLM.

Antes de aceitar o modelo, a linha de base que importa em texto: **contar
palavras**. Resposta longa costuma ganhar nota maior, e isso derruba a maioria
dos trabalhos de avaliação automática.

Primeira medição: módulo **0,429**, `cobertura` sozinha **0,463**. Ou seja, o
modelo de dez features era *pior que uma de suas próprias features*. Chegamos a
escrever a conclusão: as nove features extras não pagam.

**Estava errado, e o erro era de uso da API.** Chamávamos `preparar(gabarito)`
sem passar o enunciado e sem as stopwords derivadas do corpus. Do jeito
documentado, o número é **0,495** — e a conclusão inverte: o módulo ganha da
melhor feature sozinha por +0,033, IC95 [+0,005; +0,061], positivo em 99% das
reamostras por aluno.

A diferença de **0,066** é maior que o ganho de todas as nove features extras
somadas. Sem o enunciado, o que o aluno repetiu da pergunta conta como conceito
coberto; sem as stopwords, palavra funcional vira nó do grafo. Os dois erros
inflam a nota de quem escreveu muito.

Lição, e ela é sobre biblioteca e não sobre modelo: **uma API que deixa omitir o
parâmetro importante em silêncio produz resultado plausível e errado.** O
defeito foi documentado no módulo, com os dois números lado a lado.

## 14/09 — a primeira ideia para o modelo 2, e por que ela morreu

Antes de chegar na forma final, tentamos o caminho que parecia óbvio:
**alimentar o modelo com um corpus do domínio** para que ele "soubesse mais" de
macroeconomia. O gabarito é curto — mediana de 466 caracteres — e parecia pobre
demais para servir sozinho de referência.

Enriquecemos a referência com as respostas bem avaliadas de outros alunos da
mesma tarefa. Sem raspar nada da internet, e com o cuidado de o aluno avaliado
nunca entrar na própria referência:

| referência | Spearman |
|---|---|
| só o gabarito | **0,486** |
| + as 5 melhores respostas | 0,497 |
| + as 15 melhores | 0,490 |
| + as 40 melhores | 0,428 |
| + todas as do treino | **0,396** |

**Quanto mais ele sabia, menos ele sabia.** As primeiras cinco respostas até dão
um ganho pequeno, e depois disso cada bloco de texto novo piora — até 0,396,
abaixo de onde tinha começado.

Passamos um tempo achando que era erro de ajuste. Não era. O modo de falhar é
**estrutural**: quanto mais texto entra na referência, mais **genérica** ela
fica. Com um corpus grande, qualquer resposta cobre alguma parte dele —
inclusive a resposta ruim. A cobertura para de separar quem sabe de quem não
sabe, porque tudo passa a estar "coberto".

Foi isso que definiu a forma final: **comparar a resposta com o gabarito, e só
com ele.** Não é simplificação por preguiça — é a versão que sobreviveu. O
gabarito funciona por ser **preciso**, não por ser rico. A referência tem de ser
aquilo que a resposta deveria dizer, e nada além.

Lição que vale além deste modelo: **mais dado nem sempre é mais sinal.** Aqui,
mais dado na referência foi literalmente menos sinal, de forma ordenada e
reproduzível.

## 14/09 — deixando claro: o modelo não chama LLM

Uma pergunta que apareceu e que vale registrar, porque a banca vai fazer: esse
modelo chama LLM?

**Não. Ele não chama nada.** Conferimos nos imports do módulo: `re`,
`collections`, `dataclasses` e `math` — tudo biblioteca padrão. Zero ocorrências
de `requests`, `openai`, `urllib` ou socket. É aritmética sobre conjuntos de
palavras, e é por isso que ele custa 167 microssegundos e roda dentro do
navegador na página do protótipo.

A LLM aparece em dois lugares que não são a execução: ela produziu o **rótulo**
contra o qual os pesos foram ajustados, e ela é **o que o modelo decide se vale
chamar**. Ou seja, ele aprendeu a concordar com o GPT-4 sem precisar perguntar
nada a ele.

Escrito no README porque a distinção é fácil de errar, e porque a parte
"aprendeu a concordar com o GPT-4" é a limitação mais séria do trabalho.

## 14/09 — o teto não é 1,00

O corpus tem **três avaliações independentes do mesmo LLM** para cada resposta.
Isso permite medir uma coisa que quase nenhum trabalho mede: **o teto**.

Comparando as execuções entre si: 0,871, 0,883, 0,897. Média **0,883**. Nenhum
modelo treinado contra esse rótulo pode passar disso, porque o próprio rótulo
não é estável acima disso.

Muda completamente como se lê o 0,495: não é "metade do caminho até a
perfeição", é **56% do máximo alcançável**. E deixa claro o que o modelo é —
ele não avalia resposta, ele ordena e tria.

Medimos também onde ele paga de verdade:

- **triagem**: as 5 de menor previsão têm nota real média 1,80 contra 3,50 do
  geral. Ordenar exige muito menos que pontuar.
- **pré-filtro**: cortando em 10%/90%, 20% das respostas deixam de ir para a
  LLM, errando 14% na ponta baixa e 10% na alta.
- **custo**: 167 microssegundos por resposta, sem rede e sem GPU, contra 1 a 3
  segundos de uma chamada de LLM.

## 14/09 — o protótipo em JavaScript podia mentir

O modelo 2 foi reimplementado em JavaScript para rodar no navegador sem
servidor. Duas implementações da mesma coisa **divergem em silêncio** —
tokenização, ordenação de aresta, arredondamento.

Escrevemos o `09_conferir_port.py`: roda Python e JavaScript sobre as mesmas
200 respostas e falha se discordarem. Batem até a 15ª casa decimal (1,8e-15,
que é erro de ponto flutuante e nada mais).

Não é zelo excessivo. Sem esse teste, a página poderia mostrar um número que o
repositório não produz, e ninguém perceberia.

## 14/09 — o que fica, e o que não fica

**Modelo 1 — fica:** dois eixos (recência e frequência), padronizados por
coorte, um modelo logístico único, calibração obrigatória por coorte, e um
protótipo web que roda o modelo inteiro no navegador — porque duas variáveis e
uma logística cabem num JSON.

**Modelo 2 — fica:** dez features sobre grafo de coocorrência, usadas para
**ordenar** e para dizer quais conceitos faltaram. Não para dar nota. Spearman
0,495 contra um teto de 0,883, estável em 12 partições.

**Modelo 2 — não fica:** dar nota, e qualquer leitura do valor absoluto. Os
pesos foram ajustados em alemão; até haver ~100 respostas em português com nota
humana, só a ordem se sustenta.

**Não fica:** volume, profundidade, percentil como conserto de escala, e a
pretensão de que isso mede aprendizado. Mede **permanência**. Nenhum eixo
prediz ganho de acerto entre os dois meses, e está medido.

O ganho do modelo sobre a linha de base — o melhor eixo sozinho — é **pequeno**.
Está no README com esse nome. A parte que vale deste trabalho não é a AUC; é
saber quais dos quatro eixos iniciais merecem confiança, e por quê.
