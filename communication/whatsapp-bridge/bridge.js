'use strict';

const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  isJidBroadcast,
  downloadMediaMessage,
} = require('@whiskeysockets/baileys');
const express = require('express');
const qrcode = require('qrcode-terminal');
const pino = require('pino');
const { Boom } = require('@hapi/boom');

// ── Config ───────────────────────────────────────────────────────────────────
const WEBHOOK_URL = process.env.NAOMI_WHATSAPP_WEBHOOK_URL || 'http://127.0.0.1:18803/webhook';
const BRIDGE_PORT = Number(process.env.NAOMI_WHATSAPP_BRIDGE_PORT || 18804);
const AUTH_DIR = process.env.NAOMI_WHATSAPP_AUTH_DIR || './auth';

function maskIdentifier(value) {
  const text = String(value || '');
  if (text.length <= 4) return '***';
  return '***' + text.slice(-4);
}

// ── Logger (silent — only our console.log shows) ────────────────────────────
const logger = pino({ level: 'silent' });

// ── Express API server ───────────────────────────────────────────────────────
const app = express();
app.use(express.json({ limit: '20mb' }));

let sock = null;

// POST /send  {to, text}
app.post('/send', async (req, res) => {
  const { to, text } = req.body;
  if (!to || !text) return res.status(400).json({ error: 'to and text required' });
  if (!sock) return res.status(503).json({ error: 'WhatsApp not connected' });
  try {
    await sock.sendMessage(to, { text });
    const preview = String(text).slice(0, 60);
    console.log('[SEND] to=' + maskIdentifier(to) + ' text="' + preview + '"');
    res.json({ ok: true });
  } catch (err) {
    console.error('[SEND] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// POST /send-image  {to, image_base64, caption}
app.post('/send-image', async (req, res) => {
  const { to, image_base64, caption } = req.body;
  if (!to || !image_base64) return res.status(400).json({ error: 'to and image_base64 required' });
  if (!sock) return res.status(503).json({ error: 'WhatsApp not connected' });
  try {
    const buffer = Buffer.from(image_base64, 'base64');
    await sock.sendMessage(to, { image: buffer, caption: caption || '' });
    const cap = String(caption || '').slice(0, 40);
    console.log('[SEND-IMG] to=' + maskIdentifier(to) + ' caption="' + cap + '"');
    res.json({ ok: true });
  } catch (err) {
    console.error('[SEND-IMG] error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /status

// POST /typing  {to, state}  — state: "composing" or "paused"
app.post('/typing', async (req, res) => {
  const { to, state } = req.body;
  if (!sock || !to) return res.status(400).json({ error: 'missing params' });
  try {
    const jid = to.includes('@') ? to : to + '@s.whatsapp.net';
    await sock.sendPresenceUpdate(state || 'composing', jid);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/status', (req, res) => {
  res.json({ connected: sock !== null, bridge: 'naomi-whatsapp-bridge' });
});

// ── Forward incoming message to NAOMI webhook ────────────────────────────────
async function forwardToWebhook(payload) {
  try {
    const resp = await fetch(WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    const body = await resp.text();
    console.log('[WEBHOOK] status=' + resp.status + ' body=' + body.slice(0, 80));
  } catch (err) {
    console.error('[WEBHOOK] forward error:', err.message);
  }
}

// ── Main connect function ────────────────────────────────────────────────────
async function connectToWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();
  console.log('[BRIDGE] Baileys version: ' + version.join('.'));

  sock = makeWASocket({
    version,
    logger,
    auth: state,
    printQRInTerminal: false,
    browser: ['NAOMI Agent', 'Chrome', '120.0.0'],
    syncFullHistory: false,
    markOnlineOnConnect: false,
  });

  // Credentials update
  sock.ev.on('creds.update', saveCreds);

  // Connection state
  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      console.log('\n[BRIDGE] Scan this QR code with WhatsApp:');
      qrcode.generate(qr, { small: true });
      console.log('');
    }

    if (connection === 'open') {
      console.log('[BRIDGE] WhatsApp connected!');
    }

    if (connection === 'close') {
      const reason = new Boom(lastDisconnect?.error)?.output?.statusCode;
      console.log('[BRIDGE] Connection closed, reason: ' + reason);

      if (reason === DisconnectReason.loggedOut) {
        console.log('[BRIDGE] Logged out — delete ./auth and restart to re-link.');
        process.exit(1);
      } else {
        console.log('[BRIDGE] Reconnecting in 5s...');
        setTimeout(connectToWhatsApp, 5000);
      }
    }
  });

  // Incoming messages
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    for (const msg of messages) {
      if (!msg.message) continue;
      if (msg.key.fromMe) continue;
      if (isJidBroadcast(msg.key.remoteJid)) continue;

      const from      = msg.key.remoteJid;
      const timestamp = msg.messageTimestamp;
      const inner     = msg.message;

      // Extract text
      const text = inner.conversation
        || inner.extendedTextMessage?.text
        || inner.imageMessage?.caption
        || '';

      // Check for image media
      let hasMedia    = false;
      let mediaBuffer = null;

      if (inner.imageMessage) {
        hasMedia = true;
        try {
          const buf = await downloadMediaMessage(
            msg,
            'buffer',
            {},
            { logger, reuploadRequest: sock.updateMediaMessage }
          );
          mediaBuffer = buf.toString('base64');
        } catch (e) {
          console.error('[MEDIA] download failed:', e.message);
        }
      }

      const preview = String(text).slice(0, 60);
      console.log('[RECV] from=' + maskIdentifier(from) + ' text="' + preview + '" hasMedia=' + hasMedia);

      await forwardToWebhook({ from, text, timestamp, hasMedia, mediaBuffer });
    }
  });
}

// ── Start ────────────────────────────────────────────────────────────────────
app.listen(BRIDGE_PORT, '127.0.0.1', () => {
  console.log('[BRIDGE] HTTP API listening on http://127.0.0.1:' + BRIDGE_PORT);
  console.log('[BRIDGE] Webhook target: ' + WEBHOOK_URL);
});

connectToWhatsApp().catch(err => {
  console.error('[BRIDGE] Fatal error:', err);
  process.exit(1);
});
