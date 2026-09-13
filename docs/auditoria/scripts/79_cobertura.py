"""Mapa de cobertura: toda funcao publica e todo campo de dataclass do modulo,
e se existe (a) teste e (b) numero medido no cabecalho.

Roda SEM dataset. Serve para nao afirmar "esta tudo auditado" sem conferir -
que foi exatamente o erro cometido duas vezes durante esta auditoria.
"""
import ast, os, re, sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
PKG = os.path.join(RAIZ, 'trailup_core')
TESTES = open(os.path.join(RAIZ, 'tests', 'test_core.py'), errors='ignore').read()

def publicos(caminho):
    arv = ast.parse(open(caminho, errors='ignore').read())
    fn, campos = [], []
    for no in arv.body:
        if isinstance(no, ast.FunctionDef) and not no.name.startswith('_'):
            fn.append(no.name)
        if isinstance(no, ast.ClassDef):
            for c in no.body:
                if isinstance(c, ast.AnnAssign) and isinstance(c.target, ast.Name):
                    if not c.target.id.startswith('_'):
                        campos.append(f'{no.name}.{c.target.id}')
    return fn, campos

print(f'{"modulo":<16} {"simbolo":<28} {"teste":>7} {"numero medido":>15}')
faltando = []
for arq in sorted(os.listdir(PKG)):
    if not arq.endswith('.py') or arq == '__init__.py':
        continue
    mod = arq[:-3]
    corpo = open(os.path.join(PKG, arq), errors='ignore').read()
    cabecalho = corpo[:corpo.index('"""', 3) + 3] if corpo.startswith('"""') else ''
    fn, campos = publicos(os.path.join(PKG, arq))
    for nome in fn + campos:
        curto = nome.split('.')[-1]
        tem_teste = bool(re.search(rf'\b{re.escape(curto)}\b', TESTES))
        # numero medido: o simbolo aparece perto de um numero no arquivo
        ctx = '\n'.join(l for l in corpo.split('\n') if curto in l)
        tem_num = bool(re.search(r'\d+[,.]\d{2,}', ctx + cabecalho))
        print(f'{mod:<16} {nome:<28} {"sim" if tem_teste else "NAO":>7} {"sim" if tem_num else "NAO":>15}')
        if not tem_teste:
            faltando.append(f'{mod}.{nome}')
print(f'\n{len(faltando)} simbolos publicos sem teste' + (f': {faltando}' if faltando else ''))
sys.exit(1 if faltando else 0)
