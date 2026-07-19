/*
 * srs.js — Repetição espaçada (algoritmo SM-2, estilo Anki).
 *
 * `atualizarSM2(card, nota, agora)` é uma função PURA: recebe o estado atual do
 * card e a nota do usuário, devolve um NOVO card com o agendamento atualizado.
 * Não toca em IndexedDB nem no DOM — por isso é fácil de testar isoladamente.
 *
 * Estado SRS de um card:
 *   ease     — fator de facilidade (>= 1.3; começa em 2.5)
 *   interval — intervalo atual em dias
 *   reps     — nº de acertos consecutivos (zera ao errar)
 *   lapses   — nº de vezes que o usuário errou o card já "aprendido"
 *   due      — timestamp (ms) de quando o card volta a vencer
 *
 * Notas aceitas: 'again' (errei) | 'hard' (difícil) | 'good' (bom) | 'easy' (fácil)
 */
(function (root) {
  var DAY = 86400000; // ms em um dia
  var MIN_EASE = 1.3;

  function clampEase(e) {
    return Math.max(MIN_EASE, e);
  }

  function atualizarSM2(card, nota, agora) {
    agora = agora || Date.now();
    var ease = typeof card.ease === 'number' ? card.ease : 2.5;
    var interval = typeof card.interval === 'number' ? card.interval : 0;
    var reps = typeof card.reps === 'number' ? card.reps : 0;
    var lapses = typeof card.lapses === 'number' ? card.lapses : 0;

    var next = {};
    for (var k in card) { if (Object.prototype.hasOwnProperty.call(card, k)) next[k] = card[k]; }

    if (nota === 'again') {
      // Erro: volta pra fila de reaprendizado (~10 min) e reduz a facilidade.
      next.reps = 0;
      next.lapses = lapses + 1;
      next.ease = clampEase(ease - 0.2);
      next.interval = 0;
      next.due = agora + 10 * 60 * 1000; // 10 minutos
      return next;
    }

    // Acertos: ajusta a facilidade conforme a dificuldade relatada.
    if (nota === 'hard') ease = clampEase(ease - 0.15);
    else if (nota === 'easy') ease = ease + 0.15; // sem teto rígido
    // 'good' não mexe na facilidade.

    var newInterval;
    if (reps === 0) {
      // Primeiro acerto: "gradua" o card.
      newInterval = (nota === 'easy') ? 4 : 1; // 4 dias no fácil, 1 dia caso contrário
    } else {
      if (nota === 'hard') newInterval = Math.round(interval * 1.2);
      else if (nota === 'easy') newInterval = Math.round(interval * ease * 1.3);
      else newInterval = Math.round(interval * ease); // 'good'
    }
    if (newInterval < 1) newInterval = 1;

    next.ease = ease;
    next.reps = reps + 1;
    next.lapses = lapses;
    next.interval = newInterval;
    next.due = agora + newInterval * DAY;
    return next;
  }

  /* Rótulo curto do próximo intervalo, para os botões de nota. */
  function rotuloIntervalo(card, nota, agora) {
    agora = agora || Date.now();
    var r = atualizarSM2(card, nota, agora);
    var ms = r.due - agora;
    if (ms < 60 * 60 * 1000) return Math.max(1, Math.round(ms / 60000)) + ' min';
    if (ms < DAY) return Math.round(ms / (60 * 60 * 1000)) + ' h';
    var d = Math.round(ms / DAY);
    if (d < 30) return d + (d === 1 ? ' dia' : ' d');
    if (d < 365) return Math.round(d / 30) + ' m';
    return (Math.round(d / 36.5) / 10) + ' a';
  }

  var api = { atualizarSM2: atualizarSM2, rotuloIntervalo: rotuloIntervalo };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; // Node (testes)
  root.SRS = api; // navegador
})(typeof window !== 'undefined' ? window : this);
