# trailup-core

Medidas locais do estado do aluno, para decidir **quando vale a pena chamar a LLM**.

O TrailUp hoje dispara de 8 a 14 chamadas ao Gemini por lote de telemetria de 60 segundos, com base em regras escritas à mão que têm nome de modelo. Este pacote substitui a parte dessas decisões que **dá para medir localmente** — e diz explicitamente onde não dá.

**Sem dependência externa.** Python puro, `dataclasses` e `math`. Roda dentro de uma API que hiberna no free tier.

```bash
python3 exemplos/exemplo.py     # os módulos trabalhando juntos
python3 -m pytest tests/ -q     # 141 testes de invariante

python3 docs/auditoria/scripts/70_consistencia.py   # números divergentes entre arquivos
python3 docs/auditoria/scripts/79_cobertura.py      # símbolo público sem teste
python3 docs/auditoria/scripts/80_fuzz.py           # NaN/infinito vazando
python3 docs/auditoria/scripts/82_monotonia.py      # direção quebrada no domínio
```

---

## O que cada módulo responde

| módulo | pergunta | medida | custo |
|---|---|---|---|
| [`dificuldade`](trailup_core/dificuldade.py) | quão difícil é esta questão? | cobertura 88–91%; `prever_turma()` para a pergunta do professor | 30 alunos |
| [`ritmo`](trailup_core/ritmo.py) | é de reconhecimento ou de elaboração? | 97,3% de acerto | **5 alunos** |
| [`tempo`](trailup_core/tempo.py) | quanto deve demorar? | R² 0,562 ± 0,010 | 5 alunos |
| [`dominio`](trailup_core/dominio.py) | o aluno acerta a próxima? | AUC 0,727 / ECE 0,008 (regra atual: 0,587) | histórico do aluno |
| [`chute`](trailup_core/chute.py) | não sabe, ou não tentou? | traço com +0,922 de confiabilidade | 50 respostas na questão |
| [`revisao`](trailup_core/revisao.py) | quando trazer de volta? | AUC 0,669 / ECE 0,015 em 752 mil reencontros | nenhum |
| [`discriminacao`](trailup_core/discriminacao.py) | esta questão está quebrada? | 22,5% ± 4,0 de precisão com `confirmar()`, contra 13,8% de uma medida só | 40 respostas |
| [`gate`](trailup_core/gate.py) | vale abrir a LLM agora? | AUC 0,732 (regra atual: 0,687) | 5 respostas no tópico |
| [`engajamento`](trailup_core/engajamento.py) | o aluno vai continuar? | AUC 0,846 / 0,868 — **um modelo, duas bases** | 30 eventos |
| [`pre_avaliacao`](trailup_core/pre_avaliacao.py) | esta resposta aberta está boa? | Spearman 0,501 (teto ≈0,88) | nenhum |
| [`sql/evasao.sql`](sql/evasao.sql) | quem está em risco de abandonar? | AUC 0,767 ± 0,012 entre coortes; ECE 0,006; lift 3,4–4,1× no top 10% | semanas de uso |

Cada arquivo traz **no cabeçalho** os números que o sustentam, o experimento que os produziu, e o que já se provou que **não** adianta tentar.

Relatórios completos em [`docs/`](docs/) — cada um termina com um **apêndice de correções**. O caminho inteiro, de hipótese a estado atual, está em [`docs/TRAJETORIA.md`](docs/TRAJETORIA.md). A fila do que ainda não foi construído está em [`docs/CANDIDATOS.md`](docs/CANDIDATOS.md).

---

Todo número dos cabeçalhos foi **reconferido chamando o código** — ver [`docs/auditoria/`](docs/auditoria/AUDITORIA.md).

## As três regras de desenho

**Categoria no que é discreto, intervalo no que é contínuo.** E quem decidiu não fui eu: três métodos que descobrem a quantidade de grupos rodaram sobre 10.526 questões, e o único estável encontrou **dois** — separados pelo tempo, não pelo acerto. O ritmo vira rótulo porque o dado separa (d de Cohen **4,18** na fronteira medida, que por Otsu cai em 39,8 s). A dificuldade vira intervalo porque é contínua, e cravar um corte ali inventaria uma fronteira que não existe.

**Persistir a ação, não o diagnóstico.** `precisa_reforco()` devolve um booleano acionável; `evasao.sql` devolve faixa, não score. Dos 9 `IAMentalStateKind` que o app define, ele age sobre 2 — guardar os outros 7 é coletar dado sensível de menor para nada.

> ⚠️ **Quem já usava `dominio`:** a recalibração de 2026-09-13 mudou a escala do `p`. `precisa_reforco(limiar=0,45)` passa a disparar em **12,2%** das decisões, contra 2,3% antes — cinco vezes mais chamadas de reforço. Para manter o volume antigo use `limiar=0,28`; para manter o significado, dimensione a operação para 12%. Detalhes em [`docs/auditoria/`](docs/auditoria/AUDITORIA.md).

**Calibrar com dado próprio antes de usar.** Todas as constantes vêm de corpora estrangeiros. O que transfere é a **forma** do achado — quais sinais importam, em que direção, quanta evidência cada um exige. Os valores, não. Veja [Calibração](#calibração).

---

## O fio que liga tudo

Seis experimentos independentes chegaram à mesma conclusão:

| experimento | o que ganhou |
|---|---|
| domínio (M2) | a questão — 0,706 sozinha, contra 0,641 de todo o histórico do aluno |
| tempo esperado | a questão (R² 0,56; o aluno não prediz o próprio tempo) |
| dificuldade | a questão (o aluno não forma grupos) |
| avaliador discursivo | cobertura do gabarito, não a relação entre conceitos |
| engajamento | presença recente (traço acumulado), não estado |
| estado do aluno (M1) | **nada** — AUC 0,505, com intervalo contendo 0,50 |

**O sinal está no conteúdo e no acumulado. Não está no estado momentâneo do aluno.**

Daí o desenho: tudo aqui mede propriedade de questão ou traço acumulado. Nada infere como o aluno está se sentindo agora — o que, além de não funcionar, é o que a [issue #195](https://github.com/BitStudioLabs/trail-up/issues/195) pede para parar de fazer.

---

## Calibração

**O ponto mais importante do pacote, e o mais fácil de ignorar.**

Os cortes do EdNet aplicados ao OULAD, na **mesma unidade** (fração de dias ativos):

| | baixo | médio | alto |
|---|---|---|---|
| EdNet (onde os cortes foram medidos) | 42% | 24% | 33% |
| **OULAD (mesma régua)** | **6%** | **17%** | **77%** |

A régua não transfere. Por isso `engajamento.calibrar()` existe, e por isso `dificuldade.derivar_prior()` não é parâmetro de ajuste — é propriedade do seu banco de questões.

### Um modelo só, sem deixar ele adivinhar a plataforma

Juntar as duas bases cruas **não funciona**: com os valores absolutos, prever *de qual base a linha veio* dá **AUC 0,994**. O modelo aprende a plataforma e aplica a taxa dela. Treinado assim, o EdNet piora de 0,849 para 0,788.

Padronizando **dentro de cada coorte** antes de juntar, a identificabilidade cai para **0,508** — indistinguível de cara ou coroa. E aí um modelo único empata com os específicos:

| base de teste | modelo único (média de 12 partições) |
|---|---|
| EdNet | **0,846** ± 0,006 |
| OULAD | **0,868** ± 0,004 |

**A ordem transfere; o nível não.** Exportando o modelo de uma base para a outra sem recalibrar, o AUC se mantém (0,832–0,869) mas o ECE vai a **0,36–0,57**. O mesmo ordenador precisa de limiares de 0,440 e 0,582 para alertar os mesmos 10%.

```python
from trailup_core import engajamento

cal = engajamento.calibrar(coorte)          # >= 300 alunos com desfecho
engajamento.ordenar(2, 9, cal)              # score comum, sobre z da sua coorte
engajamento.risco(2, 9, cal, taxa_alerta=0.10)   # está entre os 10% piores?

e = engajamento.medir(dias_recentes=2, dias_ativos=9,
                      eventos=180, calibracao=cal)
e.recencia.faixa        # 'medio'
e.retencao_esperada     # None enquanto a calibração não for confiável
e.alertas()             # ['recencia'] quando há o que fazer
```

`trajetoria([dias_j1, dias_j2, dias_j3])` devolve `'caiu' | 'estavel' | 'subiu'` — **descrição, não predição**: ela não acrescenta ao score (+0,003), mas explica o risco. Só compare alunos de **mesmo total de atividade**; comparando por recência o sinal se inverte e o alerta marca os mais engajados.

Sem calibração, o módulo entrega **só a recência** — o único eixo cujos cortes são contagem, não escala — e `ordenar()` levanta erro em vez de devolver um número sem sentido.

### Acurácia não é a métrica aqui

No OULAD, com limiar 0,5, o modelo **alerta ninguém e acerta 92,9%** — a mesma acurácia de não ter modelo. Acurácia balanceada nesse ponto: 50,0%. No EdNet, onde 69,8% saem, "sempre alertar" já dá 69,8%.

Por isso `risco()` recebe **taxa de alerta**, não limiar: alertar os 10% de maior risco dá precisão de 95,1% no EdNet e 39,7% no OULAD, com a mesma régua de ordenação.

---

## Como isto foi verificado

> **Todo número de capa foi checado sob reamostragem.** Três vieram de partição favorável e foram corrigidos: o engajamento no EdNet (0,856 → **0,846 ± 0,006**), o acumulado linear do gate (0,746 → **0,732 ± 0,003**), o R² do tempo (0,578 → **0,562 ± 0,010**) e o 0,862 do OULAD (→ **0,868 ± 0,004**). Os demais confirmam dentro de 1,3 desvios.

O pacote passou por uma auditoria de **13 rodadas**: cada número dos cabeçalhos remedido chamando o próprio código, depois fuzzing com entrada hostil, monotonicidade ao longo do domínio inteiro, cobertura símbolo a símbolo da API e consistência entre documentos.

**97 verificações, 49 defeitos** — e nenhum deles apareceria olhando AUC. O padrão: **as saídas que só ordenam passaram sem defeito; as que têm unidade prometiam mais do que alguém tinha medido.**

Seis verificadores rodam **sem dataset** e falham se o contrato quebrar — ver [`docs/auditoria/`](docs/auditoria/AUDITORIA.md) e o [histórico completo](docs/TRAJETORIA.md).

## O que **não** está aqui, e por quê

| descartado | motivo |
|---|---|
| inferência de estado emocional | AUC 0,505 no único dataset com comportamento e estado declarado juntos |
| agrupamento de alunos em perfis | grupos não replicam (ARI 0,18); os traços individuais sim (0,70–0,98) |
| categorias de dificuldade por habilidade | dificuldade é unidimensional no corpus (HDBSCAN: 0 grupos) |
| roteador de intenção do chat | era circular — 88% viraram 38% ao remover o template da ferramenta |
| **volume por sessão** no engajamento | sinal forte no EdNet (AUC 0,400) que **some no OULAD** (0,522) |
| **empenho do aluno** | 3 operacionalizações, 3 nulos — nível 0,516, pós-erro 0,524 (placebo 0,527), trajetória 0,504 |
| avaliador discursivo como substituto da LLM | 0,532 contra teto de ≈0,88 — serve como triagem, nunca como nota |
| análise de frames de câmera | nenhum consumidor, nenhum dataset, e é biometria de menor |

---

## Instrumentação que falta no TrailUp

Duas lacunas pequenas no código destravam quatro módulos:

1. **Tempo por questão** não é coletado. Sem ele, `chute` e `tempo` não rodam, e `engajamento` perde o eixo de profundidade.
2. **Questões personalizadas não gravam em `questao_aluno`** (`QuestionActivity.tsx:1281`) — e são a maior parte do fluxo adaptativo. Sem isso, `dificuldade` e `discriminacao` nunca acumulam evidência, e o `dominio` fica preso em 0,64 em vez de 0,75.

---

## Licença dos dados de referência

As constantes deste pacote foram medidas em: **EdNet KT3** (CC BY-**NC**), **OULAD** (CC BY 4.0), **ARES/OSF 8y3zp** (CC BY 4.0), **classEx** (CC BY).

O EdNet é **não comercial**. Os arquivos aqui contêm *constantes medidas*, não pesos treinados — mas a decisão sobre uso comercial é de quem publica, e deve ser tomada antes do deploy, não depois.
