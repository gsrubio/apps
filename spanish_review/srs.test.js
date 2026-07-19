/* Testes do SM-2 — rodar com: node spanish_review/srs.test.js */
var SRS = require('./srs.js');
var DAY = 86400000;
var agora = 1000000000000;
var falhas = 0;

function ok(cond, msg) {
  if (cond) { console.log('  ✓ ' + msg); }
  else { console.log('  ✗ ' + msg); falhas++; }
}
function novo() { return { ease: 2.5, interval: 0, reps: 0, lapses: 0, due: agora }; }
function dias(card) { return Math.round((card.due - agora) / DAY); }

console.log('SM-2:');

// Primeiro acerto "good" gradua para 1 dia.
var c = SRS.atualizarSM2(novo(), 'good', agora);
ok(c.reps === 1 && dias(c) === 1, 'novo + good -> 1 dia, reps=1');

// Primeiro acerto "easy" gradua para 4 dias e aumenta ease.
c = SRS.atualizarSM2(novo(), 'easy', agora);
ok(dias(c) === 4 && c.ease > 2.5, 'novo + easy -> 4 dias, ease sobe');

// "again" reseta reps, some lapse, volta em ~10 min, baixa ease.
c = SRS.atualizarSM2({ ease: 2.5, interval: 10, reps: 3, lapses: 0, due: agora }, 'again', agora);
ok(c.reps === 0 && c.lapses === 1 && c.ease < 2.5 && (c.due - agora) < DAY, 'again -> reset, lapse+1, <1 dia');

// "good" num card maduro multiplica pelo ease.
c = SRS.atualizarSM2({ ease: 2.5, interval: 10, reps: 3, lapses: 0, due: agora }, 'good', agora);
ok(dias(c) === 25, 'maduro + good -> interval*ease (10*2.5=25)');

// "hard" cresce devagar e baixa o ease.
c = SRS.atualizarSM2({ ease: 2.5, interval: 10, reps: 3, lapses: 0, due: agora }, 'hard', agora);
ok(dias(c) === 12 && c.ease < 2.5, 'maduro + hard -> interval*1.2, ease baixa');

// "easy" maduro cresce mais (ease*1.3) e sobe o ease.
c = SRS.atualizarSM2({ ease: 2.5, interval: 10, reps: 3, lapses: 0, due: agora }, 'easy', agora);
ok(dias(c) > 25 && c.ease > 2.5, 'maduro + easy -> maior que good, ease sobe');

// Ease nunca fica abaixo de 1.3.
c = { ease: 1.35, interval: 5, reps: 2, lapses: 0, due: agora };
for (var i = 0; i < 5; i++) c = SRS.atualizarSM2(c, 'hard', agora);
ok(c.ease >= 1.3, 'ease respeita o piso de 1.3');

// Não muta o card original.
var orig = novo();
SRS.atualizarSM2(orig, 'good', agora);
ok(orig.reps === 0, 'função é pura (não muta o card original)');

console.log(falhas === 0 ? '\nTodos os testes passaram ✓' : ('\n' + falhas + ' teste(s) falharam ✗'));
process.exit(falhas === 0 ? 0 : 1);
