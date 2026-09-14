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
    'dominio, AUC':                  r'dom[ií]nio[^\n]{0,80}?AUC\s*0,(7\d{2})',
    'dominio, ECE':                  r'dom[ií]nio[^\n]{0,60}?ECE\s*0,(0\d{3})',
    'revisao, AUC':                  r'revis[aã]o[^\n]{0,60}?AUC\s*0,(6\d{2})',
    'pre_avaliacao, Spearman':       r'pr[eé].?avalia[çc][aã]o[^\n]{0,70}?Spearman\s*0,(\d{3})',
    'gate, AUC da forma fechada':    r'fechada[^\n]{0,40}?0,(7\d{2})',
    # engajamento tem varios 0,8xx legitimos lado a lado (modelo unico, por
    # base, e a variante de janela 14 que foi rejeitada) - fora do escopo deste
    # verificador, que compara um numero por conceito.
    # A faixa 3,4-4,1x cobre as quatro coortes; o verificador ancora no teto,
    # que e o numero que alguem seria tentado a citar sozinho.
    'lift da evasao no top 10%':     r'lift\s*(?:de\s*)?(?:\d,\d[–-])?(\d,\d)×\s*no top 10%',
    'numero de testes':              r'(\d{2,3}) testes de invariante',
    # Nivel nominal 90%. Ancorado porque ja esteve em 88,8% com sigma medido
    # em vez de calibrado, e o intervalo mentia sobre o proprio nivel.
    # So o preditivo de turma. O 90,1% do `estimar` e outro conceito (efeito do
    # reencontro em n=100) e nao deve entrar na mesma comparacao.
    'prever_turma, cobertura':       r'(?:preditivo: cobre|turma de 30(?:[^\n]{0,30}?\|){2}\s*\*\*|numa turma de 30, contra|cobertura )(9\d,\d)%',
}
# valores que convivem de proposito (progressao historica, variantes do modelo)
PERMITIDOS = {
    'evasao, AUC': {'748', '783', '770', '759', '757'},  # 748 = pre-conserto; os demais
                                     # sao coortes individuais. O valor vigente e 767.
    'pre_avaliacao, Spearman': {'447', '451', '463', '429', '482', '532', '871', '897'},
    'dominio, AUC': {'703', '719', '721', '722', '725', '727', '753', '754', '755', '780'},
    'gate, AUC da forma fechada': {'725', '732', '738', '740', '746', '748', '779'},
    'revisao, AUC': {'661', '665', '666', '669', '670'},
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
