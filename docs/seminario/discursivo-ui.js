/* Interface do demo do avaliador discursivo. O modelo esta em discursivo.js. */

const $ = id => document.getElementById(id);
const pct = x => (x * 100).toFixed(0) + '%';

/* Um exemplo em portugues, para o demo nao depender do corpus alemao.
   ATENCAO: os PESOS foram ajustados em alemao, de macroeconomia. Aqui eles
   ordenam de forma plausivel, mas o valor absoluto nao tem validacao em
   portugues - e exatamente a limitacao que o modulo declara. */
const ENUNCIADO = 'Explique o que determina a demanda agregada de uma economia.';
const GABARITO = 'A demanda agregada e a soma do consumo das familias, do ' +
  'investimento das empresas, dos gastos do governo e das exportacoes liquidas. ' +
  'O consumo depende da renda disponivel das familias. O investimento depende ' +
  'da taxa de juros e da expectativa de lucro das empresas. Os gastos do ' +
  'governo sao decididos pela politica fiscal. As exportacoes liquidas dependem ' +
  'do cambio e da renda externa.';

const EXEMPLOS = {
  boa: 'A demanda agregada e a soma do consumo das familias, do investimento ' +
    'das empresas, dos gastos do governo e das exportacoes liquidas. O consumo ' +
    'varia com a renda disponivel e o investimento responde a taxa de juros e a ' +
    'expectativa de lucro. O cambio afeta as exportacoes liquidas.',
  parcial: 'A demanda agregada depende do consumo das familias e do ' +
    'investimento das empresas. Quando a renda sobe, o consumo aumenta.',
  vazia: 'Esse e um tema muito importante da economia e que precisa ser bem ' +
    'estudado, porque envolve varios aspectos relevantes que os economistas ' +
    'discutem bastante ao longo do tempo em diferentes situacoes.',
  copia: 'O que determina a demanda agregada de uma economia sao os fatores ' +
    'que determinam a demanda agregada de uma economia.',
};

/* O corpus serve para derivar as stopwords por frequencia de documento.
   Com menos de 10 textos a funcao devolve conjunto vazio - de proposito. */
const CORPUS = [ENUNCIADO, GABARITO, ...Object.values(EXEMPLOS),
  'A politica fiscal e o conjunto de decisoes do governo sobre gastos e tributos.',
  'A taxa de juros e o principal instrumento da politica monetaria.',
  'A renda disponivel e a renda das familias depois dos impostos.',
  'O cambio e o preco de uma moeda estrangeira em moeda nacional.',
  'O produto interno bruto mede o valor dos bens e servicos finais.',
  'A inflacao e o aumento generalizado e continuo dos precos.'];

let stop = null, ref = null;

/* Os conceitos vem do gabarito, que e um campo editavel. Montamos os chips com
   textContent em vez de innerHTML: com innerHTML, escrever uma tag no gabarito
   faria a pagina executa-la. */
function chips(destino, palavras) {
  destino.replaceChildren(...palavras.map(w => {
    const s = document.createElement('span');
    s.className = 'chip';
    s.textContent = w;
    return s;
  }));
}

function recalcular() {
  const enunciado = $('enunciado').value;
  const gabarito = $('gabarito').value;
  const resposta = $('resposta').value;

  stop = Discursivo.stopwords([...CORPUS, gabarito, enunciado].filter(Boolean));
  ref = Discursivo.preparar(gabarito, enunciado, stop);

  if (!resposta.trim()) {
    $('nota').textContent = '—';
    $('faixa').textContent = '—';
    $('faixa').className = 'num';
    $('faixaTxt').textContent = 'escreva uma resposta acima';
    $('cobertura').textContent = '—';
    $('barraCob').style.width = '0';
    $('nFalta').textContent = '—';
    chips($('falta'), []);
    return;
  }

  const a = Discursivo.avaliar(resposta, ref);
  $('nota').textContent = a.nota.toFixed(1);
  $('faixa').textContent = a.faixa;
  $('faixa').className = 'num ' + (a.faixa === 'baixo' ? 'alerta'
    : a.faixa === 'alto' ? 'ok' : '');
  $('faixaTxt').textContent = a.faixa === 'baixo' ? 'topo da fila de revisão'
    : a.faixa === 'alto' ? 'provavelmente dispensa revisão' : 'mandar para a LLM';
  $('cobertura').textContent = pct(a.cobertura);
  $('barraCob').style.width = Math.min(100, a.cobertura * 100) + '%';
  $('nFalta').textContent = a.faltando.length;
  chips($('falta'), a.faltando.length ? a.faltando : ['nenhum dos centrais']);
}

$('enunciado').value = ENUNCIADO;
$('gabarito').value = GABARITO;
$('resposta').value = EXEMPLOS.parcial;

['enunciado', 'gabarito', 'resposta'].forEach(id =>
  $(id).addEventListener('input', recalcular));

document.querySelectorAll('.exemplos button').forEach(b =>
  b.addEventListener('click', () => {
    $('resposta').value = EXEMPLOS[b.dataset.ex];
    recalcular();
  }));

recalcular();
