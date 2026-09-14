/* O modelo inteiro: uma regressao logistica de duas variaveis.
   Nao ha servidor nem biblioteca - os pesos vem de modelo.json, produzido por
   scripts/06_exportar.py. O calculo abaixo e o MESMO que o Python faz. */

let M = null;

const $ = id => document.getElementById(id);
const pct = x => (x * 100).toFixed(1) + '%';

function z(valor, coorte, feature) {
  const c = M.coortes[coorte];
  return (valor - c.media[feature]) / (c.desvio[feature] || 1);
}

function risco(diasRecentes, diasAtivos, coorte) {
  const c = M.coortes[coorte];
  const recencia = diasRecentes / M.janela_recente;
  const frequencia = diasAtivos / M.janela_dias;
  const x = [z(recencia, coorte, 'recencia'), z(frequencia, coorte, 'frequencia')];
  let s = M.intercepto;
  for (let i = 0; i < x.length; i++) s += M.pesos[i] * x[i];
  // Recalibracao por coorte. Sem ela o numero sai com o nivel da media das duas
  // bases, que nao e o nivel de nenhuma das duas: no EdNet 86% nao voltam, no
  // OULAD 8,5%. A ordem e a mesma; so o nivel muda.
  const sc = c.recal_a + c.recal_b * s;
  return 1 - 1 / (1 + Math.exp(-sc));   // 1 - P(volta) = risco de nao voltar
}

function posicao(r, coorte) {
  const c = M.coortes[coorte];
  if (r >= c.limiar10) return { rot: 'alerta', cls: 'alerta',
    txt: 'entre os 10% de maior risco desta coorte' };
  if (r >= c.limiar25) return { rot: 'atenção', cls: '',
    txt: 'entre os 25% de maior risco desta coorte' };
  return { rot: 'estável', cls: 'ok', txt: 'fora das faixas de alerta' };
}

function desenhar() {
  const dr = +$('recentes').value, da = +$('ativos').value;
  const coorte = $('coorte').value;
  // Nao da para estar ativo em mais dias recentes do que dias ativos no total.
  if (dr > da) { $('ativos').value = dr; }
  const diasAtivos = Math.max(da, dr);
  $('vr').textContent = dr + ' de 10'; $('va').textContent = diasAtivos + ' de 30';

  const r = risco(dr, diasAtivos, coorte);
  const p = posicao(r, coorte);
  const c = M.coortes[coorte];

  $('risco').textContent = pct(r);
  // A cor fica na FAIXA, nao no risco. 53% de risco e alto em termos absolutos
  // e normal no EdNet, onde 86% nao voltam - pintar o numero de vermelho ali
  // seria afirmar um alarme que a coorte nao sustenta.
  $('risco').className = 'num';
  $('faixa').textContent = p.rot;
  $('faixa').className = 'num ' + p.cls;
  $('faixaTxt').textContent = p.txt;
  $('barra').style.width = Math.min(100, r * 100) + '%';

  const obs = c.referencia[dr];
  $('obs').textContent = obs === null ? '—' : pct(1 - obs);
  $('obsN').textContent = obs === null ? 'ninguém nesta faixa' :
    c.referencia_n[dr].toLocaleString('pt-BR') + ' alunos observados';
  $('retencao').textContent = pct(1 - c.retencao);
}

function tabelaReferencia() {
  const nomes = Object.keys(M.coortes);
  let h = '<table><thead><tr><th>dias ativos nos últimos 10</th>' +
    nomes.map(n => `<th>${n}</th>`).join('') + '</tr></thead><tbody>';
  for (let d = 0; d <= 10; d++) {
    h += `<tr><td>${d}</td>` + nomes.map(n => {
      const v = M.coortes[n].referencia[d];
      return `<td>${v === null ? '—' : pct(1 - v)}</td>`;
    }).join('') + '</tr>';
  }
  $('ref').innerHTML = h + '</tbody></table>';
}

function tabelaDesempenho() {
  let h = '<table><thead><tr><th>base</th><th>alunos</th><th>não voltam</th>' +
    '<th>linha de base</th><th>modelo (AUC)</th></tr></thead><tbody>';
  for (const [n, c] of Object.entries(M.coortes)) {
    const [m, sd] = M.auc[n];
    h += `<tr><td>${n}</td><td>${c.n.toLocaleString('pt-BR')}</td>` +
      `<td>${pct(1 - c.retencao)}</td><td>${M.linha_de_base[n].toFixed(3)}</td>` +
      `<td><strong>${m.toFixed(3)}</strong> ± ${sd.toFixed(3)}</td></tr>`;
  }
  $('desempenho').innerHTML = h + '</tbody></table>';
}

fetch('modelo.json').then(r => r.json()).then(m => {
  M = m;
  const sel = $('coorte');
  sel.innerHTML = Object.keys(M.coortes).map(n =>
    `<option value="${n}">${n}</option>`).join('');
  ['recentes', 'ativos', 'coorte'].forEach(id =>
    $(id).addEventListener('input', desenhar));
  tabelaReferencia(); tabelaDesempenho(); desenhar();
}).catch(e => {
  $('erro').textContent = 'Não consegui carregar modelo.json: ' + e.message;
  $('erro').hidden = false;
});
