/* build-artifact.js — gera artifact.html (arquivo único) a partir de
   index.html + styles.css + srs.js + app.js, para publicar como um
   Claude Artifact. Rode: node spanish_review/build-artifact.js */
var fs = require('fs'), path = require('path');
var dir = __dirname;
function read(f) { return fs.readFileSync(path.join(dir, f), 'utf8'); }

var html = read('index.html');
var css = read('styles.css');
var srs = read('srs.js');
var app = read('app.js');

// corpo entre <body> e </body>, sem as tags <script src=...>
var body = html.slice(html.indexOf('<body>') + 6, html.indexOf('</body>'));
body = body.split('\n').filter(function (l) { return l.indexOf('<script src=') < 0; }).join('\n').trim();

var out =
  '<title>Revisão de Espanhol</title>\n' +
  '<style>\n' + css.trim() + '\n</style>\n\n' +
  body + '\n\n' +
  '<script>\n' + srs.trim() + '\n</script>\n' +
  '<script>\n' + app.trim() + '\n</script>\n';

fs.writeFileSync(path.join(dir, 'artifact.html'), out);
console.log('artifact.html gerado (' + out.length + ' bytes)');
