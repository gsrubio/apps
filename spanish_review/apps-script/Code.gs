/**
 * Endpoint do Google Drive para o app "Revisão de Espanhol" (Drive em 1 clique).
 *
 * O que faz: guarda 1 arquivo JSON de backup DENTRO de uma pasta dedicada no
 * SEU Drive (não na raiz) e o devolve quando pedido.
 * O app chama:
 *   - GET  ?action=load&token=SEU_TOKEN         -> devolve o backup ({ok:true,data:{...}})
 *   - POST {token, action:"save", data:{...}}    -> grava o backup ({ok:true,saved:true})
 *
 * COMO PUBLICAR (uma vez):
 *   1. Acesse https://script.google.com -> Novo projeto.
 *   2. Apague o conteúdo e cole este arquivo inteiro.
 *   3. Troque o TOKEN abaixo por uma senha sua (a mesma que você porá no app).
 *   4. Implantar > Nova implantação > tipo "App da Web".
 *        - Executar como: Eu
 *        - Quem tem acesso: Qualquer pessoa
 *   5. Autorize quando pedir. Copie a URL que termina em /exec.
 *   6. No app, em Ajustes > Google Drive, cole a URL e o mesmo TOKEN.
 *
 * A PASTA: o script cria (ou reaproveita) uma pasta chamada FOLDER_NAME na sua
 * conta na primeira vez que você exporta. Todos os backups ficam ali dentro.
 *
 * Observação: a URL + token funcionam como uma senha do arquivo. Troque o token
 * quando quiser (e reimplante) para revogar acesso.
 */

var TOKEN = 'troque-este-token';                 // <<< defina o MESMO valor no app
var FOLDER_NAME = 'Revisão de Espanhol';         // pasta dedicada no seu Drive
var FILENAME = 'espanhol-review-backup.json';    // nome do arquivo dentro da pasta

function doGet(e) {
  var p = (e && e.parameter) || {};
  if (p.token !== TOKEN) return json({ ok: false, error: 'token' });
  if (p.action === 'load') {
    var folder = findFolder();                   // não cria pasta só pra ler
    var f = folder ? findFileIn(folder) : null;
    if (!f) return json({ ok: true, data: null });
    try { return json({ ok: true, data: JSON.parse(f.getBlob().getDataAsString()) }); }
    catch (err) { return json({ ok: false, error: 'arquivo corrompido' }); }
  }
  return json({ ok: true, pong: true });
}

function doPost(e) {
  var body;
  try { body = JSON.parse(e.postData.contents); }
  catch (err) { return json({ ok: false, error: 'json' }); }
  if (!body || body.token !== TOKEN) return json({ ok: false, error: 'token' });
  if (body.action === 'save') {
    var content = JSON.stringify(body.data || {});
    var folder = getOrCreateFolder();            // cria a pasta se ainda não existir
    // sobrescreve: manda a versão antiga (dentro da pasta) pra lixeira e cria a nova
    var it = folder.getFilesByName(FILENAME);
    while (it.hasNext()) it.next().setTrashed(true);
    folder.createFile(FILENAME, content, 'application/json');
    return json({ ok: true, saved: true, folder: folder.getName() });
  }
  return json({ ok: false, error: 'action' });
}

function findFolder() {
  var it = DriveApp.getFoldersByName(FOLDER_NAME);
  return it.hasNext() ? it.next() : null;
}

function getOrCreateFolder() {
  return findFolder() || DriveApp.createFolder(FOLDER_NAME);
}

function findFileIn(folder) {
  var it = folder.getFilesByName(FILENAME);
  return it.hasNext() ? it.next() : null;
}

function json(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
