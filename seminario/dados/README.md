# Os dados

Não estão versionados: o EdNet sozinho tem 4,2 GB. Baixe os dois e descompacte
aqui, ou aponte a variável `DADOS` para onde você os colocou.

## EdNet KT3

- https://github.com/riiid/ednet
- Licença **CC BY-NC 4.0** — uso acadêmico, não comercial. Cite os autores.
- Baixe `KT3.zip` e descompacte em `dados/EdNet-KT3/KT3/`.
- Resultado esperado: ~297 mil arquivos, um por aluno (`u1.csv`, `u2.csv`, …),
  com colunas `timestamp, action_type, item_id, source, user_answer, platform`.

## OULAD (Open University Learning Analytics Dataset)

- https://analyse.kmi.open.ac.uk/open_dataset
- Licença **CC BY 4.0**.
- Descompacte os CSV em `dados/oulad/`.
- Os scripts usam apenas `studentVle.csv`.

## classEx — o corpus do modelo 2

Este **está** no repositório: `dados/classex.parquet`, 3,3 MB. O modelo 2
reproduz sem baixar nada.

- 1.167 respostas discursivas de macroeconomia, em alemão
- 249 alunos, 8 tarefas
- três avaliações independentes do GPT-4 por resposta — é o que permite medir
  o teto (o quanto o LLM concorda consigo mesmo)
- licença **CC BY**

Colunas usadas: `q` (enunciado), `gab` (gabarito), `resp` (resposta),
`Color (Pseudonym)` (id do aluno), `Task` (id da tarefa) e as três
`RunN_Content_AI Evaluation`.

## Rodar com outro caminho

```bash
DADOS=/onde/voce/baixou python3 scripts/01_preparar.py
```

## Rodar mais rápido

O `01_preparar.py` lê no máximo 60 mil alunos do EdNet. Para um teste rápido:

```bash
LIMITE_ALUNOS=5000 python3 scripts/01_preparar.py
```

Os números do README vêm do padrão (60 mil). Com menos alunos eles mudam um
pouco — é a mesma lição da entrada de 11/09 do diário.
