/* app.js — app estático de revisão espaçada de espanhol.
   Sem backend: dados em IndexedDB; SM-2 vem de srs.js (window.SRS). */
(function () {
  'use strict';

  var DAY = 86400000;

  /* ---------------- IndexedDB ---------------- */
  var db = null;
  function openDB() {
    return new Promise(function (resolve, reject) {
      var req = indexedDB.open('espanhol-review', 1);
      req.onupgradeneeded = function (e) {
        var d = e.target.result;
        if (!d.objectStoreNames.contains('cards')) d.createObjectStore('cards', { keyPath: 'id' });
        if (!d.objectStoreNames.contains('meta')) d.createObjectStore('meta', { keyPath: 'k' });
      };
      req.onsuccess = function () { db = req.result; resolve(db); };
      req.onerror = function () { reject(req.error); };
    });
  }
  function tx(store, mode) { return db.transaction(store, mode).objectStore(store); }
  function reqP(r) { return new Promise(function (res, rej) { r.onsuccess = function () { res(r.result); }; r.onerror = function () { rej(r.error); }; }); }
  function allCards() { return reqP(tx('cards', 'readonly').getAll()); }
  function putCard(c) { return reqP(tx('cards', 'readwrite').put(c)); }
  function delCardDB(id) { return reqP(tx('cards', 'readwrite').delete(id)); }
  function clearCards() { return reqP(tx('cards', 'readwrite').clear()); }
  function getMeta(k) { return reqP(tx('meta', 'readonly').get(k)).then(function (r) { return r ? r.v : null; }); }
  function setMeta(k, v) { return reqP(tx('meta', 'readwrite').put({ k: k, v: v })); }

  /* ---------------- estado ---------------- */
  var cards = [];
  var settings = {
    variant: 'es-MX', newPerDay: 10, audioOnReveal: true, dir: 'ambos',
    streak: 0, lastStudy: null, doneToday: 0, doneDate: null,
    driveUrl: '', driveToken: '', driveLast: null
  };
  var uid = function () {
    return (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
      : 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  };
  function todayStr() { return new Date().toISOString().slice(0, 10); }
  function norm(s) { return (s || '').trim().toLowerCase(); }
  function esc(s) { return (s == null ? '' : String(s)).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  /* chave para dedupe: vocab pela palavra ES, correção pela forma certa */
  function noteKey(c) { return c.tipo === 'vocab' ? 'vocab|' + norm(c.es) : 'correcao|' + norm(c.ans); }

  /* ---------------- navegação ---------------- */
  var titles = {
    hoje: ['Hoje', 'revisão espaçada'], revisar: ['Revisar', ''],
    biblioteca: ['Biblioteca', 'todos os cards'], importar: ['Importar', 'colar do Claude'], ajustes: ['Ajustes', 'preferências']
  };
  window.go = function (name) {
    document.querySelectorAll('.screen').forEach(function (s) { s.classList.remove('active'); });
    document.getElementById('s-' + name).classList.add('active');
    document.querySelectorAll('.tab').forEach(function (t) { t.classList.remove('active'); });
    document.getElementById('t-' + name).classList.add('active');
    document.getElementById('bar-title').textContent = titles[name][0];
    document.getElementById('bar-sub').textContent = titles[name][1];
    document.querySelector('.screenwrap').scrollTop = 0;
    if (name === 'hoje') renderHoje();
    if (name === 'revisar') startReview();
    if (name === 'biblioteca') renderLib();
  };

  /* ---------------- HOJE ---------------- */
  function isNew(c) { return c.reps === 0 && c.lapses === 0; }
  function dueList(now) { now = now || Date.now(); return cards.filter(function (c) { return c.due <= now; }); }
  function buildQueue(now) {
    now = now || Date.now();
    var due = dueList(now);
    var news = due.filter(isNew);
    var others = due.filter(function (c) { return !isNew(c); }).sort(function (a, b) { return a.due - b.due; });
    var restanteNovos = Math.max(0, settings.newPerDay - (settings.doneDate === todayStr() ? settings.newToday || 0 : 0));
    return others.concat(news.slice(0, restanteNovos));
  }
  function dueLabel(c, now) {
    now = now || Date.now();
    if (c.due <= now) return isNew(c) ? 'novo' : 'vence hoje';
    var d = Math.round((c.due - now) / DAY);
    return d <= 0 ? 'hoje' : 'em ' + d + ' d';
  }
  function renderHoje() {
    var now = Date.now();
    var q = buildQueue(now);
    var news = dueList(now).filter(isNew).length;
    var revs = dueList(now).filter(function (c) { return !isNew(c); }).length;
    document.getElementById('st-new').textContent = news;
    document.getElementById('st-due').textContent = revs;
    document.getElementById('st-total').textContent = cards.length;
    document.getElementById('ring-due').textContent = q.length;
    var done = settings.doneDate === todayStr() ? settings.doneToday : 0;
    var frac = (done + q.length) > 0 ? done / (done + q.length) : 0;
    document.getElementById('ring-arc').setAttribute('stroke-dashoffset', String(264 * (1 - frac)));
    var stk = settings.streak || 0, stkEl = document.getElementById('streak');
    if (stk > 0) { stkEl.textContent = '🔥 ' + stk + (stk === 1 ? ' dia' : ' dias'); stkEl.style.display = ''; }
    else stkEl.style.display = 'none';
    var greet = document.getElementById('hoje-greet'), msg = document.getElementById('hoje-msg'), btn = document.getElementById('hoje-review');
    if (q.length > 0) {
      greet.textContent = '¡Hola de nuevo!';
      msg.textContent = q.length + (q.length === 1 ? ' card vence agora.' : ' cards vencem agora.');
      btn.textContent = '▶ Revisar ' + q.length + (q.length === 1 ? ' card' : ' cards'); btn.disabled = false;
    } else {
      greet.textContent = cards.length ? '¡Todo al día!' : '¡Empecemos!';
      msg.textContent = cards.length ? 'Nada vencendo agora. Volte mais tarde.' : 'Importe cards do Claude para começar.';
      btn.textContent = cards.length ? 'Nada para revisar' : 'Importar cards';
      btn.disabled = cards.length > 0;
      if (!cards.length) btn.disabled = false, btn.onclick = function () { go('importar'); };
      else btn.onclick = function () { go('revisar'); };
    }
    // recentes (um por "note", pra não repetir as duas direções da mesma palavra)
    var seen = {}, recent = [];
    cards.slice().sort(function (a, b) { return (b.created_at || 0) - (a.created_at || 0); }).forEach(function (c) {
      var k = c.note_id || c.id; if (seen[k]) return; seen[k] = 1; recent.push(c);
    });
    recent = recent.slice(0, 3);
    document.getElementById('hoje-recent').innerHTML = recent.length ? recent.map(function (c) {
      var head = c.tipo === 'vocab' ? c.es : c.ans;
      var sub = c.tipo === 'vocab' ? c.pt : short(c.explicacao_pt);
      return '<div class="libitem"><span class="chip ' + c.tipo + '">' + (c.tipo === 'vocab' ? 'Voc' : 'Cor') + '</span>' +
        '<div><div class="es">' + esc(head) + '</div><div class="pt">' + esc(sub) + '</div></div>' +
        '<span class="due">' + dueLabel(c, now) + '</span></div>';
    }).join('') : '<p class="muted" style="font-size:13px">Nenhum card ainda.</p>';
  }
  function short(s, n) { s = s || ''; n = n || 42; return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  /* ---------------- REVISAR ---------------- */
  var queue = [], qi = 0, cur = null, revealed = false;
  function startReview() {
    queue = buildQueue(); qi = 0;
    var live = document.getElementById('rev-live'), done = document.getElementById('rev-done'), empty = document.getElementById('rev-empty');
    if (!queue.length) { live.classList.add('hidden'); done.classList.add('hidden'); empty.classList.remove('hidden'); return; }
    empty.classList.add('hidden'); done.classList.add('hidden'); live.classList.remove('hidden');
    renderCard();
  }
  function speakIco() { return '<svg viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9H4z"></path><path d="M16 8a5 5 0 010 8" fill="none" stroke="currentColor" stroke-width="1.8"></path></svg>'; }
  function renderCard() {
    cur = queue[qi]; revealed = false;
    var el = document.getElementById('card');
    el.className = 'card ' + cur.tipo;
    var front;
    if (cur.tipo === 'vocab') {
      if (cur.dir === 'pt-es') {
        front = '<div class="type">Vocabulário <span class="dirtag">PT → ES</span></div>' +
          '<div class="front"><div class="headword">' + esc(cur.pt) + '</div><div class="prompt-lbl">Como se diz em espanhol?</div></div>';
      } else {
        front = '<div class="type">Vocabulário <span class="dirtag">ES → PT</span></div>' +
          '<div class="front"><div class="headword">' + esc(cur.es) + '</div>' +
          '<button class="audiobtn" onclick="sayCur(\'es\')">' + speakIco() + ' Ouvir</button></div>';
      }
    } else {
      front = '<div class="type">Correção</div><div class="front"><div class="prompt-lbl">Você disse:</div>' +
        '<div class="said"><span class="strike">' + esc(cur.said) + '</span></div></div>';
    }
    el.innerHTML = front + '<div class="reveal hidden" id="reveal"></div>';
    document.getElementById('showbtn').classList.remove('hidden');
    document.getElementById('grades').classList.add('hidden');
    document.getElementById('rev-count').textContent = 'Card ' + (qi + 1) + ' de ' + queue.length;
    document.getElementById('rev-left').textContent = 'restam ' + (queue.length - qi);
    document.getElementById('prog').style.width = (qi / queue.length * 100) + '%';
  }
  window.revealCard = function () {
    var c = cur, r = document.getElementById('reveal'), body;
    if (c.tipo === 'vocab') {
      var ans = c.dir === 'pt-es' ? c.es : c.pt;
      body = '<div class="ans">' + esc(ans) + '</div>' +
        (c.contexto_es ? '<div class="row"><span class="k">Exemplo</span><span class="ex">' + esc(c.contexto_es) + '</span></div>' : '') +
        (c.explicacao_pt ? '<div class="row"><span class="k">Nota</span><span>' + esc(c.explicacao_pt) + '</span></div>' : '');
    } else {
      body = '<div class="ans">' + esc(c.ans) + '</div>' +
        (c.explicacao_pt ? '<div class="row"><span class="k">Por quê</span><span>' + esc(c.explicacao_pt) + '</span></div>' : '') +
        (c.contexto_es ? '<div class="row"><span class="k">Exemplo</span><span class="ex">' + esc(c.contexto_es) + '</span></div>' : '');
    }
    var esText = c.tipo === 'vocab' ? (c.dir === 'pt-es' ? c.es : c.contexto_es || c.es) : (c.contexto_es || c.ans);
    body += '<button class="audiobtn" style="margin-top:6px" onclick="sayText(' + JSON.stringify(esText).replace(/"/g, '&quot;') + ')">' + speakIco() + ' Ouvir</button>';
    r.innerHTML = body; r.classList.remove('hidden');
    document.getElementById('showbtn').classList.add('hidden');
    var g = document.getElementById('grades');
    g.querySelectorAll('small[data-g]').forEach(function (sm) {
      sm.textContent = SRS.rotuloIntervalo(c, sm.getAttribute('data-g'));
    });
    g.classList.remove('hidden');
    revealed = true;
    if (settings.audioOnReveal) sayText(c.tipo === 'vocab' ? (c.dir === 'pt-es' ? c.es : c.es) : c.ans);
  };
  window.grade = function (nota) {
    if (!revealed) return;
    var card = cur; // captura antes de renderCard() reatribuir `cur`
    var updated = SRS.atualizarSM2(card, nota);
    var i = cards.findIndex(function (x) { return x.id === card.id; });
    if (i >= 0) cards[i] = updated; // atualiza memória de forma síncrona
    putCard(updated); // persiste (assíncrono, sem depender de `cur`)
    bumpDone();
    qi++;
    if (qi >= queue.length) {
      document.getElementById('prog').style.width = '100%';
      document.getElementById('rev-live').classList.add('hidden');
      document.getElementById('rev-done-msg').textContent = 'Você revisou ' + queue.length + (queue.length === 1 ? ' card.' : ' cards.');
      document.getElementById('rev-done').classList.remove('hidden');
      return;
    }
    renderCard();
  };
  function bumpDone() {
    var t = todayStr();
    if (settings.doneDate !== t) { settings.doneDate = t; settings.doneToday = 0; settings.newToday = 0; }
    settings.doneToday++;
    // streak
    if (settings.lastStudy !== t) {
      var y = new Date(Date.now() - DAY).toISOString().slice(0, 10);
      settings.streak = (settings.lastStudy === y) ? (settings.streak || 0) + 1 : 1;
      settings.lastStudy = t;
    }
    saveSettings();
  }

  /* ---------------- BIBLIOTECA ---------------- */
  var libFilter = 'todos';
  window.setFilter = function (btn) {
    document.querySelectorAll('.fchip').forEach(function (b) { b.classList.remove('on'); });
    btn.classList.add('on'); libFilter = btn.getAttribute('data-f'); renderLib();
  };
  window.renderLib = function () {
    var now = Date.now();
    var q = norm(document.getElementById('libsearch').value);
    var rows = cards.filter(function (c) {
      if (libFilter === 'vocab' && c.tipo !== 'vocab') return false;
      if (libFilter === 'correcao' && c.tipo !== 'correcao') return false;
      if (libFilter === 'due' && c.due > now) return false;
      if (q) { var hay = norm((c.es || c.said) + ' ' + (c.pt || c.ans) + ' ' + (c.explicacao_pt || '')); if (hay.indexOf(q) < 0) return false; }
      return true;
    }).sort(function (a, b) { return a.due - b.due; });
    document.getElementById('libhdr').textContent = rows.length + ' de ' + cards.length + ' cards';
    document.getElementById('liblist').innerHTML = rows.length ? rows.map(function (c) {
      var head = c.tipo === 'vocab' ? c.es : c.ans;
      var sub = c.tipo === 'vocab' ? c.pt : short(c.explicacao_pt, 40);
      var stcls = c.due <= now ? (isNew(c) ? 'new' : 'due') : 'ok';
      var stlab = c.due <= now ? (isNew(c) ? 'novo' : 'vence hoje') : ('revisão ' + dueLabel(c, now));
      var dir = c.tipo === 'vocab' ? '<span class="dirtag">' + (c.dir === 'pt-es' ? 'PT→ES' : 'ES→PT') + '</span>' : '';
      return '<div class="lib2"><span class="chip ' + c.tipo + '">' + (c.tipo === 'vocab' ? 'Voc' : 'Cor') + '</span>' +
        '<div class="body"><div class="es">' + esc(head) + dir + '</div><div class="pt">' + esc(sub) + '</div>' +
        '<div class="meta"><span class="dot ' + stcls + '"></span><span class="st">' + stlab + '</span></div></div>' +
        '<button class="kebab" onclick="cardMenu(\'' + c.id + '\')" aria-label="Opções">⋯</button></div>';
    }).join('') : '<p class="muted" style="text-align:center;padding:24px 0">Nenhum card encontrado.</p>';
  };
  window.cardMenu = function (id) {
    var c = cards.find(function (x) { return x.id === id; }); if (!c) return;
    var campo = c.tipo === 'vocab' ? 'tradução' : 'forma certa';
    var atual = c.tipo === 'vocab' ? c.pt : c.ans;
    var v = prompt('Editar ' + campo + ' (apague tudo e confirme para excluir o card):', atual);
    if (v === null) return;
    if (v.trim() === '') {
      if (!confirm('Excluir este card?')) return;
      delCardDB(id).then(function () { cards = cards.filter(function (x) { return x.id !== id; }); renderLib(); toast('Card excluído'); });
      return;
    }
    if (c.tipo === 'vocab') c.pt = v.trim(); else c.ans = v.trim();
    putCard(c).then(function () { renderLib(); toast('Card atualizado'); });
  };

  /* ---------------- IMPORTAR ---------------- */
  var parsed = [];
  function parseBlock(txt) {
    // aceita bloco com ``` cercas; ignora vazias e comentários (#)
    txt = txt.replace(/```[a-z]*\n?/gi, '');
    var out = [];
    txt.split('\n').forEach(function (line) {
      var l = line.replace(/\s+$/, '');
      if (!l.trim() || l.trim().charAt(0) === '#') return;
      var p = l.split('\t');
      if (p.length < 3) { // tolera separador por " | " se não houver tabs
        if (l.indexOf('|') >= 0) p = l.split('|').map(function (x) { return x.trim(); });
      }
      if (p.length < 3) return;
      var tipo = norm(p[0]).indexOf('cor') === 0 ? 'correcao' : 'vocab';
      out.push({ tipo: tipo, a: (p[1] || '').trim(), b: (p[2] || '').trim(), ctx: (p[3] || '').trim(), exp: (p[4] || '').trim() });
    });
    return out;
  }
  window.analyze = function () {
    parsed = parseBlock(document.getElementById('paste').value);
    var existing = {}; cards.forEach(function (c) { existing[noteKey(c)] = true; });
    var voc = 0, cor = 0, novos = 0, novosCards = 0, html = '';
    parsed.forEach(function (r) {
      var key = r.tipo === 'vocab' ? 'vocab|' + norm(r.a) : 'correcao|' + norm(r.b);
      r.dup = !!existing[key];
      if (!r.dup) { existing[key] = true; novos++; novosCards += (r.tipo === 'vocab' && settings.dir === 'ambos') ? 2 : 1; }
      if (r.tipo === 'vocab') voc++; else cor++;
      html += '<div class="pv-row' + (r.dup ? ' dup' : '') + '"><span class="chip ' + r.tipo + '">' + (r.tipo === 'vocab' ? 'Voc' : 'Cor') + '</span>' +
        '<span class="es">' + esc(r.a) + '</span><span class="arrow">→</span><span>' + esc(r.b) + '</span>' +
        (r.dup ? '<span class="dupflag">já existe</span>' : '') + '</div>';
    });
    if (!parsed.length) { toast('Nada reconhecido no bloco'); document.getElementById('preview').classList.add('hidden'); return; }
    document.getElementById('pv-voc').textContent = voc + ' vocab';
    document.getElementById('pv-cor').textContent = cor + (cor === 1 ? ' correção' : ' correções');
    document.getElementById('pv-list').innerHTML = html;
    var b = document.getElementById('pv-import');
    b.textContent = novosCards ? ('Importar ' + novosCards + (novosCards === 1 ? ' card' : ' cards')) : 'Nada novo para importar';
    b.disabled = !novosCards;
    document.getElementById('preview').classList.remove('hidden');
  };
  window.doImport = function () {
    var now = Date.now(), created = 0, writes = [];
    parsed.forEach(function (r) {
      if (r.dup) return;
      var note = uid();
      var base = { note_id: note, contexto_es: r.ctx, explicacao_pt: r.exp, ease: 2.5, interval: 0, reps: 0, lapses: 0, due: now, created_at: now };
      if (r.tipo === 'correcao') {
        var cc = Object.assign({ id: uid(), tipo: 'correcao', dir: null, said: r.a, ans: r.b }, base);
        cards.push(cc); writes.push(putCard(cc)); created++;
      } else {
        var dirs = settings.dir === 'ambos' ? ['es-pt', 'pt-es'] : [settings.dir];
        dirs.forEach(function (d) {
          var vc = Object.assign({ id: uid(), tipo: 'vocab', dir: d, es: r.a, pt: r.b }, base, { note_id: note });
          cards.push(vc); writes.push(putCard(vc)); created++;
        });
      }
    });
    Promise.all(writes).then(function () {
      toast(created + (created === 1 ? ' card importado ✓' : ' cards importados ✓'));
      document.getElementById('paste').value = '';
      document.getElementById('preview').classList.add('hidden');
      parsed = [];
      go('hoje');
    });
  };

  /* ---------------- AJUSTES ---------------- */
  function applySettingsUI() {
    document.getElementById('variant-lbl').textContent = flag(settings.variant) + ' ' + settings.variant;
    setSeg('seg-variant', 'v', settings.variant);
    setSeg('seg-audio', 'a', settings.audioOnReveal ? '1' : '0');
    setSeg('seg-dir', 'd', settings.dir);
    document.getElementById('newn').textContent = settings.newPerDay;
    document.getElementById('drive-url').value = settings.driveUrl || '';
    document.getElementById('drive-token').value = settings.driveToken || '';
    document.getElementById('drive-last').textContent = settings.driveLast ? new Date(settings.driveLast).toLocaleString('pt-BR') : 'nunca';
    document.getElementById('storage-n').textContent = cards.length;
  }
  function flag(v) { return v === 'es-ES' ? '🇪🇸' : v === 'es-AR' ? '🇦🇷' : '🇲🇽'; }
  function setSeg(id, attr, val) {
    var seg = document.getElementById(id); if (!seg) return;
    seg.querySelectorAll('button').forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-' + attr) === val); });
  }
  function wireSeg(id, attr, cb) {
    document.getElementById(id).addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      cb(b.getAttribute('data-' + attr)); setSeg(id, attr, b.getAttribute('data-' + attr));
    });
  }
  window.stepNew = function (d) { settings.newPerDay = Math.max(0, Math.min(60, settings.newPerDay + d * 5)); document.getElementById('newn').textContent = settings.newPerDay; saveSettings(); };
  function saveSettings() { setMeta('settings', settings); }

  /* ---------------- áudio (TTS grátis do navegador) ---------------- */
  window.sayText = function (text) {
    try {
      if (!('speechSynthesis' in window) || !text) return;
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text); u.lang = settings.variant || 'es-MX'; u.rate = 0.95;
      var vs = window.speechSynthesis.getVoices();
      var v = vs.find(function (x) { return x.lang && x.lang.replace('_', '-') === u.lang; }) || vs.find(function (x) { return /^es/i.test(x.lang); });
      if (v) u.voice = v;
      window.speechSynthesis.speak(u);
    } catch (e) {}
  };
  window.sayCur = function (which) { sayText(which === 'es' ? cur.es : cur.pt); };

  /* ---------------- backup: JSON / Anki / Drive ---------------- */
  function backupObject() { return { schema: 1, exported_at: new Date().toISOString(), settings: settings, cards: cards }; }
  function download(name, text, type) {
    var blob = new Blob([text], { type: type || 'application/json' });
    var url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; document.body.appendChild(a); a.click();
    setTimeout(function () { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
  }
  window.exportJSON = function () { download('espanhol-backup-' + todayStr() + '.json', JSON.stringify(backupObject(), null, 2)); toast('Backup JSON gerado'); };
  window.exportAnki = function () {
    var q = function (s) { s = (s == null ? '' : String(s)); return '"' + s.replace(/"/g, '""') + '"'; };
    var rows = cards.map(function (c) {
      var front, back, tags;
      if (c.tipo === 'vocab') {
        front = c.dir === 'pt-es' ? c.pt : c.es;
        back = (c.dir === 'pt-es' ? c.es : c.pt) + (c.contexto_es ? '<br>' + c.contexto_es : '') + (c.explicacao_pt ? '<br><i>' + c.explicacao_pt + '</i>' : '');
        tags = 'espanhol vocab ' + (c.dir || '');
      } else {
        front = 'Corrija: ' + c.said; back = c.ans + (c.explicacao_pt ? '<br>' + c.explicacao_pt : '') + (c.contexto_es ? '<br><i>' + c.contexto_es + '</i>' : '');
        tags = 'espanhol correcao';
      }
      return [q(front), q(back), q(tags)].join(',');
    });
    download('espanhol-anki-' + todayStr() + '.csv', rows.join('\n'), 'text/csv');
    toast('CSV do Anki gerado');
  };
  window.restoreJSON = function (ev) {
    var f = ev.target.files && ev.target.files[0]; if (!f) return;
    var rd = new FileReader();
    rd.onload = function () {
      try {
        var data = JSON.parse(rd.result);
        if (!data.cards || !Array.isArray(data.cards)) throw new Error('formato');
        if (!confirm('Restaurar backup? Isso substitui os ' + cards.length + ' cards atuais por ' + data.cards.length + '.')) return;
        clearCards().then(function () {
          return Promise.all(data.cards.map(function (c) { return putCard(c); }));
        }).then(function () {
          cards = data.cards;
          if (data.settings) { settings = Object.assign(settings, data.settings); saveSettings(); }
          applySettingsUI(); toast('Backup restaurado ✓'); go('hoje');
        });
      } catch (e) { toast('Arquivo inválido'); }
    };
    rd.readAsText(f);
    ev.target.value = '';
  };

  function driveCfg() {
    settings.driveUrl = document.getElementById('drive-url').value.trim();
    settings.driveToken = document.getElementById('drive-token').value.trim();
    saveSettings();
    return settings.driveUrl;
  }
  window.driveExport = function () {
    var url = driveCfg(); if (!url) { toast('Configure a URL do endpoint'); return; }
    toast('Enviando ao Drive…');
    fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'text/plain;charset=utf-8' },
      body: JSON.stringify({ token: settings.driveToken, action: 'save', data: backupObject() })
    }).then(function (r) { return r.json(); }).then(function (res) {
      if (res && res.ok) { settings.driveLast = Date.now(); saveSettings(); applySettingsUI(); toast('Exportado pro Drive ✓'); }
      else toast('Drive recusou: ' + (res && res.error || 'erro'));
    }).catch(function () { toast('Falha ao contatar o Drive'); });
  };
  window.driveImport = function () {
    var url = driveCfg(); if (!url) { toast('Configure a URL do endpoint'); return; }
    toast('Baixando do Drive…');
    fetch(url + (url.indexOf('?') < 0 ? '?' : '&') + 'action=load&token=' + encodeURIComponent(settings.driveToken))
      .then(function (r) { return r.json(); }).then(function (res) {
        if (!res || !res.ok || !res.data || !Array.isArray(res.data.cards)) { toast('Drive: nada para importar'); return; }
        var byId = {}; cards.forEach(function (c) { byId[c.id] = c; });
        var writes = [], add = 0, upd = 0;
        res.data.cards.forEach(function (c) { if (byId[c.id]) upd++; else add++; byId[c.id] = c; writes.push(putCard(c)); });
        Promise.all(writes).then(function () {
          cards = Object.keys(byId).map(function (k) { return byId[k]; });
          settings.driveLast = Date.now(); saveSettings(); applySettingsUI();
          toast('Do Drive: ' + add + ' novos, ' + upd + ' atualizados');
        });
      }).catch(function () { toast('Falha ao contatar o Drive'); });
  };

  /* ---------------- tema + toast ---------------- */
  window.toggleTheme = function () {
    var dark = document.documentElement.getAttribute('data-theme') === 'dark' ||
      (!document.documentElement.getAttribute('data-theme') && matchMedia('(prefers-color-scheme:dark)').matches);
    document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
    try { localStorage.setItem('theme', dark ? 'light' : 'dark'); } catch (e) {}
  };
  var toastT;
  window.toast = function (m) {
    var el = document.getElementById('toast'); el.textContent = m; el.classList.add('show');
    clearTimeout(toastT); toastT = setTimeout(function () { el.classList.remove('show'); }, 1900);
  };

  /* ---------------- init ---------------- */
  function init() {
    try { var t = localStorage.getItem('theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) {}
    openDB().then(function () {
      return Promise.all([allCards(), getMeta('settings')]);
    }).then(function (res) {
      cards = res[0] || [];
      if (res[1]) settings = Object.assign(settings, res[1]);
      wireSeg('seg-variant', 'v', function (v) { settings.variant = v; document.getElementById('variant-lbl').textContent = flag(v) + ' ' + v; saveSettings(); });
      wireSeg('seg-audio', 'a', function (a) { settings.audioOnReveal = a === '1'; saveSettings(); });
      wireSeg('seg-dir', 'd', function (d) { settings.dir = d; saveSettings(); });
      document.getElementById('drive-url').addEventListener('change', driveCfg);
      document.getElementById('drive-token').addEventListener('change', driveCfg);
      applySettingsUI();
      renderHoje();
      if ('speechSynthesis' in window) { window.speechSynthesis.getVoices(); window.speechSynthesis.onvoiceschanged = function () {}; }
      if ('serviceWorker' in navigator) { navigator.serviceWorker.register('sw.js').catch(function () {}); }
    }).catch(function (e) {
      toast('Erro ao abrir o banco local');
      console.error(e);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
