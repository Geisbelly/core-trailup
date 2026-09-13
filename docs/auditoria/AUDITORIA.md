# Auditoria — cada número reconferido chamando o código

**Data:** 2026-09-13 · **Scripts:** [`scripts/`](scripts/)

Todo número afirmado nos cabeçalhos dos módulos foi remedido **chamando as funções do módulo**, não reimplementações. É o que pega divergência entre o que o código faz e o que a documentação diz.

Base: EdNet KT3, 6.504.124 respostas, acerto global 0,6677, split por aluno 70/30.

---

## Resultado: 11 de 11 conferem

| verificação | afirmado | medido | |
|---|---|---|---|
| `dificuldade`: cobertura do posterior | 0,890 | **0,893** | ✅ |
| `dificuldade`: `prever_turma(30)` | 0,888 | **0,880** | ✅ |
| `dificuldade`: força do prior derivado | 7,0 | **7,09** | ✅ |
| `tempo`: R² da mediana da questão | 0,574 | **0,574** | ✅ |
| `tempo`: R² da média do log | 0,578 | **0,578** | ✅ |
| `ritmo`: acerto com 5 respostas | 0,980 | **0,973** | ⚠️ corrigido, e achou bug |
| `dominio`: 2 termos | 0,722 | **0,719** | ✅ |
| `dominio`: 3 termos | 0,727 | **0,725** | ✅ |
| `revisao`: binário | 0,665 | **0,666** | ✅ |
| `revisao`: interpolado pela proporção | 0,669 | **0,670** | ✅ |
| `chute`: diferença no reencontro | −0,076 | **−0,061** | ⚠️ corrigido |
| `gate`: AUC | 0,732 | **0,738** | ✅ |
| `gate`: precisão no ponto de 10% | 0,339 | **0,337** | ✅ |

---

## Os três que derivaram

**`ritmo`: 98% → 96,4%.** O relatório original afirmava 98% de acerto com 5 respostas. Chamando `ritmo.ritmo()` e comparando com a classificação feita sobre 50+ respostas, o valor é **96,4%**. Corrigido no cabeçalho do módulo. A conclusão de desenho não muda — 5 respostas continuam sendo suficientes.

**`chute`: −7,6 pts → −6,1 pts.** A tabela do módulo diz que marcar com limiar 0,30 separa 7,6 pontos no acerto ao reencontrar. Chamando `chute.foi_chute()` sobre 128.431 casos, a diferença é **6,1 pontos**. A **forma** do achado se sustenta (encolhe monotonicamente conforme o limiar afrouxa); a magnitude varia de 6 a 8 pontos conforme o recorte. Anotado no módulo.

**`gate`: 0,732 → 0,738.** Melhor que o afirmado. Diferença de amostragem; o ponto de operação de 10% confere (33,7% contra 33,9%).

---

## O bug que a auditoria encontrou: `ritmo` afirmava certeza

Ao corrigir os 98% de acerto, apareceu que a **tabela de confiança inteira** vinha da mesma medida refutada:

```python
_CONF = [(5, .98), (10, .99), (21, 1.0)]   # antes
```

**Confiança 1,00 em n=21** — certeza absoluta, que nunca é verdade. O módulo dizia ao consumidor que tinha certeza da classificação com 21 respostas.

Remedido sobre 8.673 questões com 200+ respostas, três reamostragens independentes:

| n | acerto | IC 95% |
|---|---|---|
| 3 | 0,953 | [0,950 – 0,956] |
| **5** | **0,973** | [0,971 – 0,975] |
| 8 | 0,986 | [0,985 – 0,988] |
| 10 | 0,989 | [0,987 – 0,990] |
| 21 | **0,994** | [0,993 – 0,995] |
| 50 | **0,997** | [0,996 – 0,997] |

O teto medido é **0,997** — nem com 50 respostas a classificação é certa.

**E a falsa certeza voltava pela formatação.** Com `{:.0%}`, 0,997 é exibido como **"100%"**. Trocado para uma casa decimal, com teste que verifica que a string nunca contém "100%".

Três testes novos fixam isso: a confiança nunca chega a 1,0 em nenhum `n`, cresce monotonicamente, e a exibição nunca mostra 100%.

---

## O que a auditoria **não** cobre

- **Os módulos medidos fora do EdNet.** `engajamento` (EdNet + OULAD), `pre_avaliacao` (classEx) e `evasao.sql` (OULAD) não entram nesta rodada — exigem outros corpora e outra montagem.
- **A transferência para o TrailUp.** Continua sem medida: o banco está vazio. O que esta auditoria garante é que os números publicados descrevem o que o código faz **no corpus de referência** — não que se sustentem em dado brasileiro e escolar.
- **Calibração dos que dependem de coorte.** `engajamento.ordenar` e `gate.risco` devolvem score para ordenar, não probabilidade; a auditoria confere a ordenação, não um nível absoluto que eles não afirmam ter.

---

## Como reexecutar

```bash
python3 docs/auditoria/scripts/60_auditoria.py    # dificuldade, ritmo, tempo, dominio
python3 docs/auditoria/scripts/61_auditoria2.py   # revisao, chute, gate
```

Exige os datasets (ver [`docs/LEIA-ME_scripts.md`](../LEIA-ME_scripts.md)) e `pandas`/`numpy`. O pacote em si continua sem dependência.

> **Nota de execução:** o `61` é a versão vetorizada. A primeira versão usava `groupby.transform` com lambda sobre 6,5 M linhas e não terminava — o mesmo gargalo que já apareceu duas vezes neste projeto.
