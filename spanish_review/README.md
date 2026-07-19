# Revisão de Espanhol 🗣️

App **estático** (sem backend, sem chave de API) de **repetição espaçada** para treinar espanhol.
A conversa e o feedback por **voz acontecem no Claude**; aqui você **importa** as correções/palavras e **revisa** (algoritmo SM-2, estilo Anki). Os dados ficam no próprio dispositivo (IndexedDB).

## Como usar (fluxo)

1. **Treine por voz no Claude.** Crie um Projeto no claude.ai com as instruções de `claude-project-instructions.md` e converse por voz (cenários reais, correções em português).
2. **Peça `exporta`.** O Claude gera um bloco com as correções e o vocabulário da sessão.
3. **Importe no app.** Aba **Importar** → cole o bloco → **Analisar** → **Importar** (com dedupe automático).
4. **Revise.** Aba **Revisar**: ouça o espanhol (botão de áudio), revele a resposta e dê a nota (Errei / Difícil / Bom / Fácil) — o app reagenda cada card.
5. **Backup.** Aba **Ajustes**: Google Drive em 1 clique (ver abaixo), export **Anki (CSV)** e **JSON**.

## Rodar localmente

Precisa ser servido por HTTP (o IndexedDB não funciona abrindo o arquivo direto):

```sh
cd spanish_review
python3 -m http.server 8000
```

Abra `http://localhost:8000`. Para acessar do **celular na mesma rede**, use `http://IP-DO-PC:8000`.

## Publicar (GitHub Pages)

Sirva a pasta `spanish_review/` como site estático (Settings → Pages) e instale como PWA pelo navegador do celular ("Adicionar à tela inicial").

## Google Drive em 1 clique (opcional)

Backup/sync entre aparelhos **sem login OAuth e sem servidor**, via um endpoint na sua própria conta:

1. Abra `apps-script/Code.gs`, siga as instruções no topo (publicar como "App da Web").
2. Copie a URL `…/exec` e o token.
3. No app, **Ajustes → Google Drive**: cole a URL e o token.
4. Use **↑ Exportar** (salva o baralho + agendamento no Drive) e **↓ Importar** (traz o estado atual em outro aparelho).

## Arquivos

| Arquivo | Papel |
|---|---|
| `index.html` / `styles.css` | telas e estilo (5 abas: Hoje, Revisar, Biblioteca, Importar, Ajustes) |
| `srs.js` | algoritmo SM-2 (função pura `atualizarSM2`) — testes em `srs.test.js` (`node srs.test.js`) |
| `app.js` | IndexedDB, revisão, importação (parser + dedupe), biblioteca, backup JSON/CSV/Drive |
| `manifest.webmanifest` / `sw.js` / `icon.svg` | PWA (instalável, offline) |
| `claude-project-instructions.md` | instruções para o Projeto do Claude + formato de exportação |
| `apps-script/Code.gs` | endpoint do Google Drive (Apps Script) |

## Formato do bloco de importação

Uma linha por card, campos separados por **TAB**: `tipo ⇥ frente ⇥ verso ⇥ exemplo_es ⇥ explicacao_pt`
(`tipo` = `vocab` ou `correcao`). Detalhes e exemplos em `claude-project-instructions.md`.

> Este app é independente do app de visão computacional (`../computer_vision/`) e não usa Streamlit.
