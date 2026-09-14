# Sobre os `scripts/` dentro de `docs/`

São os scripts que produziram cada número dos relatórios. Estão aqui para
auditoria — para conferir *como* uma medida foi obtida, não para rodar
direto.

**Exceção: os `*_sem_*.py`.** Esses são o pipeline do seminário, e *rodam* —
mas a partir de [`seminario/scripts/`](../seminario/README.md), que tem a
ordem, o `requirements.txt` e o diário de decisões. A cópia aqui é para cada
modelo guardar seu próprio código junto dos demais scripts de auditoria; ela
tem os caminhos normalizados como os outros e o cabeçalho diz de onde veio.

**Eles não rodam sem os datasets**, que não estão neste repositório por
licença e por tamanho (EdNet KT3 sozinho tem 10,6 GB). Os caminhos foram
substituídos por `/caminho/para/os/datasets` e `/caminho/para/os/intermediarios`;
aponte-os para onde você baixou cada base.

| dataset | licença | onde obter |
|---|---|---|
| EdNet KT3 | CC BY-**NC** | github.com/riiid/ednet |
| OULAD | CC BY 4.0 | analyse.kmi.open.ac.uk/open_dataset |
| ARES / OSF 8y3zp | CC BY 4.0 | osf.io/8y3zp |
| classEx | CC BY | Zenodo 19467052 |
| AI4EDU | CC BY | Zenodo 19232684 |

Dependências dos scripts (não do pacote): `pandas`, `numpy`, `scikit-learn`,
`scipy`, `pyarrow`. O pacote `trailup_core` em si não depende de nada.
