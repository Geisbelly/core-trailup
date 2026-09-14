"""ETAPA 9 - o JavaScript do prototipo da o MESMO numero que o Python?

O prototipo reimplementa o avaliador discursivo em JavaScript para rodar sem
servidor. Duas implementacoes da mesma coisa divergem em silencio - tokenizacao,
ordenacao de aresta, arredondamento. Este script roda as duas sobre as mesmas
respostas e compara.

Falha com codigo 1 se divergirem. E o teste que impede o prototipo de mostrar
um numero que o repositorio nao produz.

-- PROVENIENCIA ------------------------------------------------------------
Este script e a etapa 9 do pipeline do seminario, em `seminario/scripts/`,
que tem README e diario de decisoes. A copia aqui existe para o modelo
`pre_avaliacao` guardar, junto dos demais scripts de auditoria, o codigo que
produziu os numeros do relatorio.

Para RODAR o pipeline na ordem e com as instrucoes, use `seminario/scripts/`.
Aqui os caminhos foram normalizados como nos outros scripts de `docs/` -
ver docs/LEIA-ME_scripts.md.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from comum import SAIDA, exige

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trailup_core import pre_avaliacao as pa

RAIZ = Path(__file__).resolve().parents[2]
JS = RAIZ / 'docs' / 'seminario' / 'discursivo.js'
TOLERANCIA = 1e-9

PONTE = r'''
global.window = {};
const src = require('fs').readFileSync(process.argv[2], 'utf8')
  .replace('window.Discursivo', 'module.exports');
const mod = {exports: {}};
new Function('module', 'window', src)(mod, global.window);
const D = mod.exports;
const caso = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const stop = D.stopwords(caso.corpus);
const out = caso.itens.map(it => {
  const ref = D.preparar(it.gab, it.q, stop);
  return D.avaliar(it.resp, ref).nota;
});
console.log(JSON.stringify({stop: [...stop].sort(), notas: out}));
'''

if __name__ == '__main__':
    d = pd.read_parquet(exige(SAIDA.parent / 'dados' / 'classex.parquet', 'ver etapa 7'))
    amostra = d.sample(min(200, len(d)), random_state=7).reset_index(drop=True)
    corpus = (list(d.gab.astype(str).unique()) + list(d.q.astype(str).unique())
              + list(d.resp.astype(str)))

    stop = pa.stopwords(corpus)
    py = []
    for r, g, q in zip(amostra.resp, amostra.gab, amostra.q):
        py.append(pa.avaliar(str(r), pa.preparar(str(g), enunciado=str(q), stop=stop)).nota)

    caso = {'corpus': corpus,
            'itens': [{'resp': str(r), 'gab': str(g), 'q': str(q)}
                      for r, g, q in zip(amostra.resp, amostra.gab, amostra.q)]}
    SAIDA.mkdir(exist_ok=True)
    (SAIDA / 'caso.json').write_text(json.dumps(caso))
    (SAIDA / 'ponte.js').write_text(PONTE)

    try:
        saida = subprocess.run(['node', str(SAIDA / 'ponte.js'), str(JS), str(SAIDA / 'caso.json')],
                               capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit('node nao encontrado - instale Node.js para rodar esta conferencia')
    except subprocess.CalledProcessError as e:
        raise SystemExit(f'o JavaScript falhou:\n{e.stderr}')
    js = json.loads(saida)

    print(f'{len(amostra)} respostas conferidas\n')
    print(f'  stopwords  Python {len(stop):>3}  |  JavaScript {len(js["stop"]):>3}  '
          f'|  {"iguais" if sorted(stop) == js["stop"] else "DIFERENTES"}')
    dif = np.abs(np.array(py) - np.array(js['notas']))
    print(f'  notas      diferenca maxima {dif.max():.2e}  |  media {dif.mean():.2e}')
    piores = np.argsort(-dif)[:3]
    for i in piores:
        print(f'    Python {py[i]:.6f}  JavaScript {js["notas"][i]:.6f}  '
              f'diferenca {dif[i]:.2e}')

    if sorted(stop) != js['stop'] or dif.max() > TOLERANCIA:
        raise SystemExit('\nDIVERGEM. O prototipo mostraria um numero que o '
                         'repositorio nao produz. Conserte antes de publicar.')
    print(f'\n  CONFEREM dentro de {TOLERANCIA:.0e}. O prototipo mostra o que o Python calcula.')
