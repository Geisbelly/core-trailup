/* O avaliador discursivo, portado do Python para JavaScript.
   As dez features sao operacoes de conjunto sobre tokens - nao ha rede neural,
   nao ha embedding, nao ha chamada de rede. Por isso o modelo inteiro cabe
   nesta pagina e roda enquanto voce digita.

   Os pesos e as constantes sao os mesmos de trailup_core/pre_avaliacao.py. */

const JANELA = 4;
const SEP = '~|~';
const PESOS = {
  no_peso: +1.986397, no_cobertura: +0.368581, cob_conceitos_centrais: +0.882276,
  divagacao: +0.132323, razao_tam: +0.250126, aresta_cobertura: +0.658591,
  aresta_peso: -2.120108, log_len: +0.812077, n_nos: -0.013081, n_arestas: -0.002815,
};
const INTERCEPTO = -1.364109;

const tokens = t => (String(t).toLowerCase().match(/[\wÀ-ÿ]{3,}/gu) || []);

function grafo(texto, stop) {
  const ws = tokens(texto).filter(w => !stop.has(w));
  const nos = new Map(), ar = new Map();
  ws.forEach(w => nos.set(w, (nos.get(w) || 0) + 1));
  for (let i = 0; i < ws.length; i++)
    for (let j = i + 1; j < Math.min(i + JANELA, ws.length); j++)
      if (ws[i] !== ws[j]) {
        const k = [ws[i], ws[j]].sort().join(SEP);
        ar.set(k, (ar.get(k) || 0) + 1);
      }
  return { nos, ar };
}

/* Stopwords por frequencia de documento. Com menos de 10 textos devolve vazio:
   melhor nenhuma stopword do que transformar o vocabulario inteiro em stopword,
   que e o que acontece com dois textos curtos. */
function stopwords(textos, dfMax = 0.40) {
  if (textos.length < 10) return new Set();
  const df = new Map();
  textos.forEach(t => new Set(tokens(t)).forEach(w => df.set(w, (df.get(w) || 0) + 1)));
  const s = new Set();
  df.forEach((n, w) => { if (n / textos.length > dfMax) s.add(w); });
  return s;
}

function preparar(gabarito, enunciado, stop) {
  const { nos, ar } = grafo(gabarito, stop);
  const grau = new Map();
  ar.forEach((w, k) => k.split(SEP).forEach(x => grau.set(x, (grau.get(x) || 0) + w)));
  const centrais = [...grau.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(x => x[0]);
  const tokensEnunciado = new Set(enunciado ? tokens(enunciado).filter(w => !stop.has(w)) : []);
  return { nos, arestas: ar, centrais, tokensEnunciado, stop };
}

function avaliar(resposta, ref) {
  const { nos: nr, ar } = grafo(resposta, ref.stop);
  const sr = new Set(nr.keys()), sg = new Set(ref.nos.keys());
  const er = new Set(ar.keys()), eg = new Set(ref.arestas.keys());
  const ni = [...sr].filter(w => sg.has(w));
  const ei = [...er].filter(e => eg.has(e));
  const soma = m => [...m.values()].reduce((a, b) => a + b, 0);
  const totN = Math.max(soma(ref.nos), 1), totA = Math.max(soma(ref.arestas), 1);

  const f = {
    no_peso: ni.reduce((a, w) => a + ref.nos.get(w), 0) / totN,
    no_cobertura: ni.length / Math.max(sg.size, 1),
    cob_conceitos_centrais: ref.centrais.length
      ? ref.centrais.filter(w => sr.has(w)).length / ref.centrais.length : 0,
    divagacao: [...sr].filter(w => !sg.has(w) && !ref.tokensEnunciado.has(w)).length
      / Math.max(sr.size, 1),
    razao_tam: sr.size / Math.max(sg.size, 1),
    aresta_cobertura: ei.length / Math.max(eg.size, 1),
    aresta_peso: ei.reduce((a, e) => a + Math.min(ar.get(e), ref.arestas.get(e)), 0) / totA,
    log_len: Math.log1p(String(resposta).length),
    n_nos: sr.size,
    n_arestas: er.size,
  };
  let nota = INTERCEPTO;
  for (const k in PESOS) nota += PESOS[k] * f[k];
  nota = Math.max(1, Math.min(5, nota));
  return {
    nota, features: f,
    cobertura: f.no_cobertura,
    faltando: ref.centrais.filter(w => !sr.has(w)),
    faixa: nota < 3 ? 'baixo' : nota < 4 ? 'medio' : 'alto',
  };
}

window.Discursivo = { stopwords, preparar, avaliar };
