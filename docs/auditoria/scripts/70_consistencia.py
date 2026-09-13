"""Varre o repositorio atras de CONTRADICOES: o mesmo numero citado com
valores diferentes em arquivos diferentes.

Nao precisa de dataset nenhum - le so o repositorio. Rode depois de qualquer
mudanca de numero; foi assim que apareceram tres divergencias que nenhum teste
pegaria (o modulo dizia uma coisa e o relatorio outra).
"""
import os, re, sys, collections

CHAVES = {
    'ritmo, acerto com 5 respostas': r'ritmo[^\n]{0,60}?(9[0-9],\d)%|(9[0-9],\d)%[^\n]{0,40}?ritmo',
    'evasao, AUC':                   r'evas[aã]o[^\n]{0,80}?0,(7\d{2})',
    'dominio, AUC':                  r'dominio[^\n]{0,80}?AUC\s*0,(7\d{2})',
}
# valores que convivem de proposito (progressao historica, variantes do modelo)
PERMITIDOS = {
    'evasao, AUC': {'748'},        # valor anterior ao conserto do SQL, anotado como tal
}

def arquivos(raiz='.'):
    for base in ('trailup_core', 'docs', 'sql'):
        for root, _, fs in os.walk(os.path.join(raiz, base)):
            if 'scripts' in root:
                continue
            for f in fs:
                if f.endswith(('.py', '.md', '.sql')):
                    yield os.path.join(root, f)
    yield os.path.join(raiz, 'README.md')

def main(raiz='.'):
    problemas = 0
    for nome, pat in CHAVES.items():
        vals = collections.defaultdict(list)
        for a in arquivos(raiz):
            try:
                linhas = open(a, errors='ignore').readlines()
            except OSError:
                continue
            for i, l in enumerate(linhas, 1):
                for m in re.finditer(pat, l, re.I):
                    v = next(g for g in m.groups() if g)
                    vals[v].append(f'{a}:{i}')
        distintos = set(vals) - PERMITIDOS.get(nome, set())
        if len(distintos) > 1:
            problemas += 1
            print(f'DIVERGENCIA em "{nome}": {sorted(distintos)}')
            for v in sorted(distintos):
                print(f'    {v} <- {vals[v][0]}' + (f' (+{len(vals[v])-1})' if len(vals[v]) > 1 else ''))
        else:
            print(f'ok  {nome}: {sorted(vals) or "nao citado"}')
    print(f'\n{problemas} divergencias')
    return 1 if problemas else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '.'))
