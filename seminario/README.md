# Dois modelos para gastar LLM só onde vale

Dois problemas reais de uma plataforma de estudo, cada um com dado real,
linha de base e protótipo funcionando:

| | pergunta | métrica | protótipo |
|---|---|---|---|
| **1. engajamento** | este aluno vai parar de estudar? | AUC 0,873 / 0,885 contra linha de base 0,858 / 0,879 | [abrir](https://geisbelly.github.io/core-trailup/seminario/) |
| **2. resposta discursiva** | vale abrir a LLM para esta resposta? | Spearman 0,495 ± 0,021, teto medido 0,883 | [abrir](https://geisbelly.github.io/core-trailup/seminario/discursivo.html) |

O que une os dois: **decidir onde gastar uma chamada cara**. Um decide em quem
vale intervir; o outro, qual resposta precisa mesmo de correção por LLM.

**Grupo:** Geisbelly · Victória · Maria Antonia

**Diário de decisões:** [`DIARIO.md`](DIARIO.md) — é onde está o caminho.

---

## 1. Qual o problema, e por que ele importa

### Modelo 1 — engajamento

Plataformas de estudo perdem a maior parte dos alunos em silêncio. Ninguém
cancela, ninguém avisa — a pessoa simplesmente para de aparecer. Quando o
abandono é percebido, já passou o momento em que dava para fazer algo.

A pergunta operacional é: **dá para identificar, com quatro semanas de
comportamento, quem não vai voltar?** Se der, uma intervenção barata (mensagem,
revisão mais leve, mudança de ritmo) pode ser dirigida a quem precisa, em vez
de disparada para todo mundo.

O que torna o problema interessante — e é por isso que ele foi escolhido — é
que **a resposta provavelmente não é a mesma em plataformas diferentes**. Num
curso obrigatório, quase todo mundo volta. Num app de autoestudo, quase ninguém.
Um modelo que só foi visto numa realidade não tem como saber qual dos dois casos
ele aprendeu.

### Modelo 2 — resposta discursiva

Corrigir texto aberto é caro. Mandar toda resposta para uma LLM custa dinheiro
e segundos; não mandar nenhuma significa não corrigir. A pergunta é se dá para
**decidir sozinho nas pontas óbvias** — a resposta claramente vazia e a
claramente completa — e mandar para a LLM só o meio, que é onde a decisão é
difícil.

Também importa o caso em que a LLM já está trabalhando: enquanto ela não
responde, o aluno pode receber um retorno provisório dizendo **quais conceitos
do gabarito faltaram**.

---

## 2. De onde vieram os dados, e quantos são

Duas bases públicas, escolhidas por serem **o mais diferentes possível**:

| base | o que é | alunos no painel | não voltam |
|---|---|---|---|
| **EdNet KT3** | autoestudo de inglês (TOEIC), adulto, coreano. Sem turma, sem professor, sem matrícula — continuar é 100% voluntário | 26.547 | 86,2% |
| **OULAD** | universitário britânico, EAD, com turma, professor, matrícula e abandono formalmente registrado | 25.434 | 8,5% |

**51.981 alunos no total.** A diferença entre 86,2% e 8,5% não é defeito de
amostragem: são fenômenos diferentes, e é exatamente por isso que as duas estão
aqui. Um eixo que funcione nas duas está descrevendo engajamento; um que
funcione só numa está descrevendo aquela plataforma.

- EdNet KT3 — https://github.com/riiid/ednet — licença **CC BY-NC 4.0**
- OULAD — https://analyse.kmi.open.ac.uk/open_dataset — licença **CC BY 4.0**

Os dados **não estão no repositório** (o EdNet sozinho tem 4,2 GB). Os links
acima e o [`dados/README.md`](dados/README.md) dizem como baixar; os scripts
avisam com instrução exata se não encontrarem os arquivos.

**Para o modelo 2**, o corpus **classEx**: 1.167 respostas discursivas de
macroeconomia, de 249 alunos em 8 tarefas, em **alemão**. Cada resposta tem
três avaliações independentes do GPT-4 — e são essas três que tornam o trabalho
possível, porque permitem medir **o teto**: o quanto o próprio LLM concorda
consigo mesmo. Esse arquivo cabe no repositório (3,3 MB) e está em
`dados/classex.parquet`, então o modelo 2 reproduz sem baixar nada.

O rótulo é nota de **LLM, não humana**. É a limitação mais séria do modelo 2, e
está no item 7.

---

## 3. O que foi feito com eles antes de modelar

**Janela fixa, igual para todos.** Os eixos são medidos nos **dias 0–29** de
cada aluno, e o desfecho é verificado nos **dias 30–59**. O "dia 0" é o primeiro
evento *daquele* aluno, não uma data de calendário — cada um entra quando entra,
e o que se compara é o mesmo trecho de vida.

**Nada da segunda janela entra no cálculo dos eixos.** É a regra que impede
vazamento, e ela vale para as duas bases.

**Sessões, no EdNet.** Eventos separados por mais de 30 minutos contam como
sessões diferentes. O OULAD registra o *dia*, não a *hora* — lá isso não é
calculável, e o eixo que dependia disso ficou `NaN` em vez de ser preenchido
(ver item 6).

**Corte de entrada.** Aluno com menos de 30 eventos na janela não entra: não há
comportamento suficiente para medir nada.

**O desfecho não é rotulado por ninguém.** "Voltou" é ter qualquer atividade nos
dias 30–59. Sai direto do dado, sem julgamento nosso e sem anotação humana.

---

## 4. Que modelos foram testados

### Modelo 1 — engajamento

Antes de qualquer modelo, **cada eixo foi medido sozinho, nas duas bases**, e só
entrou quem passou nas duas:

| eixo | EdNet | OULAD | entra? |
|---|---|---|---|
| recência (dias ativos nos últimos 10) | 0,794 | 0,879 | **sim** |
| frequência (dias ativos nos 30) | 0,858 | 0,837 | **sim** |
| profundidade (eventos por dia ativo) | 0,665 | 0,516 | não — fraco demais numa |
| volume (eventos por sessão) | 0,593 | *não medível* | não — ver item 6 |

Depois, três desenhos:

1. **Um modelo por base.** Funciona, mas não responde à pergunta: dois modelos
   que discordam não dizem qual está certo.
2. **Um modelo nas duas bases juntas, com os valores crus.** Testado, e
   descartado pelo motivo do item 6.
3. **Um modelo nas duas bases, padronizadas dentro de cada coorte** — regressão
   logística de duas variáveis. **É este.**

A escolha por logística de duas variáveis não é falta de ambição: é que o modelo
inteiro precisa caber num JSON e rodar no navegador, sem servidor. E, como o
item 5 mostra, o ganho de qualquer coisa mais sofisticada seria medido contra
uma linha de base que já é alta.

### Modelo 2 — resposta discursiva

Compara o **grafo de coocorrência** da resposta com o do gabarito: quais
conceitos aparecem, quais ligações entre conceitos, o que falta. Dez features,
todas operações de conjunto sobre tokens — sem rede neural, sem embedding, sem
chamada de rede.

Foram testadas e descartadas: enriquecer a referência com corpus do domínio
(**piora**), camada semântica por LSA (**não acrescenta nada**), boosting sobre
as mesmas features (pior que a logística), e um encoder multilíngue pré-treinado
(+0,057 ao custo de ~900 MB de dependência).

---

## 5. Qual o resultado, com a métrica adequada e a comparação com a linha de base

**A métrica é AUC**, não acurácia. AUC é a chance de ordenar certo um par de
alunos escolhido ao acaso — um que voltou e um que não. 0,50 é cara ou coroa.

**Por que acurácia não serve aqui, com número:** no OULAD 91,5% dos alunos
voltam. Um modelo que **nunca alerta ninguém** acerta **91,5%**. Qualquer
trabalho que reporte acurácia nesta base está escondendo que não faz nada.

**Linha de base:** o melhor eixo **sozinho**, sem modelo nenhum. É contra isso
que o modelo tem de ganhar — ganhar de 0,50 não significa nada.

| base | linha de base | modelo (12 partições) | ganho |
|---|---|---|---|
| EdNet | 0,858 | **0,873 ± 0,004** | +0,015 |
| OULAD | 0,879 | **0,885 ± 0,006** | +0,006 |

**O ganho é pequeno, e está declarado como pequeno.** A frequência sozinha já
faz quase todo o trabalho. O que o modelo acrescenta é combinar os dois eixos de
forma estável — o peso da recência é **3,2×** o da frequência, e essa proporção
se mantém nas 12 partições.

Todos os números são **média ± desvio de 12 partições independentes por aluno**.
Não de uma divisão só — ver a entrada de 11/09 do diário, onde a divisão sortuda
nos custou um número errado repetido em todo lugar.

**Como se usa na prática:** ponto de operação, não limiar fixo. Alertando os 10%
de maior risco de cada coorte, o limiar é 0,541 no EdNet e 0,708 no OULAD.
Exportar um limiar de uma base para a outra troca completamente quem é alertado
— por isso a calibração por coorte é requisito, não enfeite.

### Modelo 2 — e aqui a linha de base é o comprimento do texto

Em avaliação automática de texto, **contar palavras** derruba a maioria dos
trabalhos: resposta longa costuma ganhar nota maior. Se contar palavras chega
perto, o resto não está fazendo nada.

E o **teto não é 1,00**. Como cada resposta tem três avaliações independentes do
mesmo LLM, dá para medir quanto ele concorda consigo mesmo: **0,883**. Nenhum
modelo treinado contra esse rótulo pode passar disso.

| critério | Spearman | do teto |
|---|---|---|
| contar palavras da resposta | +0,283 | 32% |
| contar caracteres | +0,303 | 34% |
| sobreposição crua de tokens com o gabarito | +0,409 | 46% |
| cobertura do gabarito (1 feature) | +0,463 | 52% |
| **o módulo inteiro (10 features)** | **+0,495** | **56%** |
| *teto: duas execuções do próprio LLM* | *+0,883* | *100%* |

Ganha, mas **por pouco** — e a distância até o teto é maior que a distância até
a linha de base. O ganho sobre a melhor feature sozinha é **+0,033**, IC95
[+0,005; +0,061], positivo em 99% das reamostras por aluno. Estável:
**0,495 ± 0,021** em 12 partições.

**Onde ele de fato paga.** Como triagem, as 5 respostas de menor previsão têm
nota real média **1,80 contra 3,50 do geral**. Ordenar exige muito menos que
pontuar — é por isso que um Spearman de 0,50 serve para triagem e não serve para
dar nota.

Como pré-filtro, cortando em 10%/90%, **20% das respostas deixam de ir para a
LLM**, errando 14% na ponta baixa e 10% na alta. E custa **167 microssegundos
por resposta**, contra 1 a 3 segundos de uma chamada de LLM.

---

## 6. O que não funcionou

**O volume, que era o achado mais bonito.** "Eventos por sessão" tinha sinal
forte no EdNet. Ao levar para o OULAD, descobrimos algo pior que um resultado
fraco: **não dá para calcular lá**, porque a base registra o dia e não a hora.
Pior ainda, a primeira versão do nosso script preenchia esse buraco com outra
coluna, e o teste de replicação passava **comparando a coluna com ela mesma**.
Não dá para defender num modelo um eixo que metade das realidades-alvo não
consegue nem medir.

**Percentil como conserto de escala.** Achávamos que converter os eixos para
percentil resolveria a diferença entre as bases. Não resolveu: AUC igual,
calibração pior. Normalizar a *feature* não corrige diferença na *prevalência do
desfecho*.

**Nosso teste de vazamento, na primeira versão, não podia detectar nada.**
Comparávamos cru contra padronizado treinando numa base e testando na outra —
e dava o mesmo número sempre, porque padronizar dentro da coorte é uma
transformação monótona e AUC só olha ordem. O teste era vazio por construção.

**E a correção que nos contrariou:** refeito do jeito certo, o custo de juntar
as bases cruas em **AUC** é de uma a três milésimas — muito menor do que
afirmávamos. O estrago real aparece nos **pesos**, que ficam ~15× maiores
(9,49 contra 0,61), porque a única forma de a logística separar duas bases de
escalas diferentes é esticar o coeficiente. Mantivemos a padronização por isso,
não pelo motivo que tínhamos escrito antes.

**Uma terceira base onde o modelo não funciona.** Numa coorte de leitura
**obrigatória**, a AUC fica entre 0,49 e 0,57. Não é defeito: com 87,7%
voltando por obrigação, não há variação a prever.

**E no modelo 2, um erro nosso que quase virou conclusão publicada.** A primeira
medição chamava `preparar(gabarito)` sem o enunciado e sem as stopwords do
corpus — e deu Spearman **0,429**, *abaixo* da `cobertura` sozinha (0,463).
Íamos concluir que as dez features não valiam nada. Chamando a API do jeito
documentado, o número é **0,495** e a conclusão inverte. A diferença de **0,066**
é maior que o ganho de todas as nove features extras.

Sem o enunciado, o que o aluno repetiu da pergunta conta como conceito coberto;
sem as stopwords, palavra funcional vira nó do grafo. Os dois erros inflam a
nota de quem escreveu muito. O defeito real é da **API**, que permite omitir os
dois em silêncio — está documentado no módulo agora.

---

## 7. Uma limitação honesta

**Isto mede permanência, não aprendizado.** Nenhum dos eixos prediz ganho de
acerto entre os dois meses. Um aluno que aparece todo dia e não avança é
classificado como engajado, porque ele *está* engajado pela definição que
usamos — e essa definição pode não ser a que interessa a quem ensina.

Duas limitações menores, também reais: a janela de 30 dias foi escolhida por
conveniência e não testamos alternativas; e as duas bases são estrangeiras, de
adultos, em contextos que não são o de uma escola brasileira. **A ordem
transfere entre plataformas; o nível não transfere de jeito nenhum** — qualquer
uso novo exige recalibrar com dado próprio antes de acreditar no número.

**No modelo 2: o alvo do treino é uma nota de LLM, não humana.** Ele aprende a
concordar com o GPT-4 — e o GPT-4 pode estar errado de forma sistemática, sem
que nada nesta medição consiga perceber. Não há nota humana no corpus para
comparar.

E os pesos foram ajustados em texto **alemão**, de uma única disciplina. As
features são agnósticas de idioma — são operações de conjunto sobre tokens — mas
os pesos não são. Até haver ~100 respostas em português com nota humana, o uso
defensável é só **ordenar**. O protótipo mostra um exemplo em português
justamente para deixar isso visível: ele ordena de forma plausível, e o valor
absoluto não tem validação nenhuma nesse idioma.

---

## Como rodar

```bash
git clone https://github.com/Geisbelly/core-trailup.git
cd core-trailup/seminario
pip install -r requirements.txt

# MODELO 2 roda agora, sem baixar nada (o corpus está no repositório):
cd scripts
python3 07_discursivo.py            # contra as linhas de base
python3 08_discursivo_operacao.py   # pré-filtro, triagem, custo
python3 09_conferir_port.py         # o JavaScript == o Python? (precisa de Node)

# MODELO 1 exige baixar as duas bases — ver dados/README.md:
python3 scripts/01_preparar.py    # monta o painel (~6 min)
python3 scripts/02_eixos.py       # cada eixo, nas duas bases
python3 scripts/03_vazamento.py   # o teste de adivinhação da base
python3 scripts/04_treinar.py     # modelo, 12 partições
python3 scripts/05_calibrar.py    # ponto de operação e calibração
python3 scripts/06_exportar.py    # gera docs/modelo.json do protótipo
```

Cada script imprime o que mede e o que aquilo quer dizer. Rodar do 01 ao 09
reproduz todos os números deste README.

O `09_conferir_port.py` merece nota: o protótipo reimplementa o modelo 2 em
JavaScript para rodar sem servidor, e **duas implementações da mesma coisa
divergem em silêncio**. Ele roda as duas sobre 200 respostas e falha se
discordarem. Hoje batem até a 15ª casa decimal.

Os scripts rodam de dentro de `seminario/scripts/`. Para ver o protótipo
localmente, a partir da raiz do repositório:

```bash
python3 -m http.server -d docs/seminario
```

---

## Onde isto vive

Este seminário é uma pasta dentro do repositório [`core-trailup`](../), que
reúne os modelos do projeto TrailUp. Tudo o que o enunciado pede está aqui
dentro — README, diário, scripts, dados e `requirements.txt`.

**Uma dependência do repositório-mãe, e ela é de propósito.** Os scripts do
modelo 2 importam [`trailup_core.pre_avaliacao`](../trailup_core/pre_avaliacao.py),
que é o módulo em produção. Poderíamos ter copiado o código para cá, mas aí o
que este trabalho mede não seria o que roda de verdade — e a primeira medição
(item 6) mostra exatamente por que isso importa: o erro estava em *como
chamamos* o módulo, e uma cópia teria escondido isso.

Por isso, clone o repositório inteiro. O modelo 2 roda em seguida, sem baixar
mais nada.

## Estrutura

```
README.md            este arquivo
DIARIO.md            o registro do processo — é o documento principal
requirements.txt     pandas, numpy, scikit-learn
scripts/             as nove etapas, na ordem (01-06 modelo 1, 07-09 modelo 2)
dados/README.md      como baixar as bases do modelo 1 (não cabem aqui)
dados/classex.parquet  o corpus do modelo 2 (3,3 MB, cabe)
saida/               o que os scripts produzem (fora do versionamento)

../docs/seminario/   o protótipo web, publicado no GitHub Pages
```
