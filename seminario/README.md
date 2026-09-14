# Quem vai parar de estudar?

Prever, a partir do comportamento das primeiras 4 semanas, se um aluno ainda
vai estar estudando no mês seguinte — e testar a mesma pergunta em **duas
realidades opostas** de ensino a distância.

**Grupo:** Geisbelly · Victória · Maria Antonia

**Protótipo web:** https://geisbelly.github.io/core-trailup/seminario/
**Diário de decisões:** [`DIARIO.md`](DIARIO.md) — é onde está o caminho.

---

## 1. Qual o problema, e por que ele importa

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

---

## Como rodar

```bash
pip install -r requirements.txt
# baixe as duas bases conforme dados/README.md, depois:
python3 scripts/01_preparar.py    # monta o painel (~6 min)
python3 scripts/02_eixos.py       # cada eixo, nas duas bases
python3 scripts/03_vazamento.py   # o teste de adivinhação da base
python3 scripts/04_treinar.py     # modelo, 12 partições
python3 scripts/05_calibrar.py    # ponto de operação e calibração
python3 scripts/06_exportar.py    # gera docs/modelo.json do protótipo
```

Cada script imprime o que mede e o que aquilo quer dizer. Rodar do 01 ao 06
reproduz todos os números deste README.

Os scripts rodam de dentro de `seminario/scripts/`. Para ver o protótipo
localmente, a partir da raiz do repositório:

```bash
python3 -m http.server -d docs/seminario
```

---

## Onde isto vive

Este seminário é uma pasta dentro do repositório [`core-trailup`](../), que
reúne os modelos do projeto TrailUp. O trabalho do seminário é
**auto-contido**: tudo o que o enunciado pede está aqui dentro, e os scripts
não dependem de nada do resto do repositório.

## Estrutura

```
README.md            este arquivo
DIARIO.md            o registro do processo — é o documento principal
requirements.txt     pandas, numpy, scikit-learn
scripts/             as seis etapas, na ordem
dados/README.md      como baixar as bases (elas não cabem aqui)
saida/               o que os scripts produzem (fora do versionamento)

../docs/seminario/   o protótipo web, publicado no GitHub Pages
```
