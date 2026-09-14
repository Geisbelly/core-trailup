"""Os scripts do seminario copiados para docs/ ainda batem com os originais?

O pipeline do seminario vive em `seminario/scripts/`. Cada modelo guarda uma
copia dos scripts que produziram seus numeros, em `docs/<modelo>/scripts/`,
com dois desvios DELIBERADOS: um cabecalho de proveniencia no fim do docstring
e os caminhos de dataset normalizados.

Duas copias do mesmo codigo divergem em silencio - alguem conserta uma e
esquece a outra, e a partir dai o relatorio descreve um codigo que nao existe.
Este verificador falha quando isso acontece.

Roda sem dataset.
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[3]
SEM = RAIZ / 'seminario' / 'scripts'

PARES = [
    ('01_preparar.py', 'docs/engajamento/scripts/40_sem_painel.py'),
    ('02_eixos.py', 'docs/engajamento/scripts/41_sem_eixos.py'),
    ('03_vazamento.py', 'docs/engajamento/scripts/42_sem_vazamento.py'),
    ('04_treinar.py', 'docs/engajamento/scripts/43_sem_treino.py'),
    ('05_calibrar.py', 'docs/engajamento/scripts/44_sem_calibracao.py'),
    ('06_exportar.py', 'docs/engajamento/scripts/45_sem_exporta.py'),
    ('07_discursivo.py', 'docs/discursiva/scripts/20_sem_linhas_de_base.py'),
    ('08_discursivo_operacao.py', 'docs/discursiva/scripts/21_sem_operacao.py'),
    ('09_conferir_port.py', 'docs/discursiva/scripts/22_sem_port_js.py'),
]

# Os unicos desvios permitidos entre original e copia.
MARCA = '-- PROVENIENCIA'
TROCAS = [("'/Users/user/Downloads/Inicio", "'/caminho/para/os/datasets")]


def corpo(texto, tirar_proveniencia):
    """Devolve o codigo depois do docstring de topo, mais o docstring sem o
    bloco de proveniencia - que e o unico texto que pode diferir."""
    if not texto.startswith('"""'):
        return None, texto
    fim = texto.index('"""', 3)
    doc = texto[3:fim]
    if tirar_proveniencia and MARCA in doc:
        doc = doc[:doc.index(MARCA)]
    return doc.rstrip(), texto[fim + 3:]


def main():
    problemas = []
    for origem, destino in PARES:
        a, b = SEM / origem, RAIZ / destino
        if not a.exists():
            problemas.append(f'{origem}: o original sumiu de seminario/scripts/')
            continue
        if not b.exists():
            problemas.append(f'{destino}: a copia sumiu')
            continue

        doc_a, cod_a = corpo(a.read_text(), False)
        doc_b, cod_b = corpo(b.read_text(), True)

        for de, para in TROCAS:
            cod_a = cod_a.replace(de, para)
            doc_a = (doc_a or '').replace(de, para)

        if doc_a != doc_b:
            problemas.append(f'{destino}: o docstring divergiu do original')
        if cod_a != cod_b:
            # aponta a primeira linha diferente, para nao mandar procurar
            la, lb = cod_a.splitlines(), cod_b.splitlines()
            n = next((i for i in range(max(len(la), len(lb)))
                      if (la[i:i+1] or [None]) != (lb[i:i+1] or [None])), 0)
            problemas.append(
                f'{destino}: o codigo divergiu do original, a partir da linha {n+1}\n'
                f'      original: {(la[n:n+1] or ["(fim do arquivo)"])[0][:70]}\n'
                f'      copia   : {(lb[n:n+1] or ["(fim do arquivo)"])[0][:70]}')
        if MARCA not in (b.read_text()[:b.read_text().find('"""', 3)]):
            problemas.append(f'{destino}: a copia perdeu o cabecalho de proveniencia')

    print(f'{len(PARES)} pares conferidos\n')
    for p in problemas:
        print(f'  DIVERGE  {p}')
    if problemas:
        print(f'\n{len(problemas)} divergencias. A copia em docs/ deixou de descrever')
        print('o codigo que roda. Sincronize a partir de seminario/scripts/.')
        return 1
    print('  0 divergencias - as copias descrevem o codigo que roda')
    return 0


if __name__ == '__main__':
    sys.exit(main())
