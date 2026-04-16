"""
NAOMI Agent - Web Dashboard
Real-time control panel with WebSocket for live updates.
Token-based authentication for all API endpoints.
"""
import json
import time
import asyncio
import logging
import secrets
import os
from datetime import datetime
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("naomi.dashboard")

# Store connected WebSocket clients
connected_clients = set()

# Dashboard auth token — auto-generated on first run, stored in data/
TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dashboard_token.txt")


def _get_or_create_token() -> str:
    """Get existing token or generate a new one."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            return f.read().strip()
    token = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        f.write(token)
    logger.info(f"Dashboard token generated: {token[:8]}...")
    return token


DASHBOARD_TOKEN = _get_or_create_token()


def create_dashboard(agent) -> FastAPI:
    app = FastAPI(title="NAOMI Agent Dashboard")

    async def verify_token(request: Request):
        """Verify dashboard token from header or query param."""
        token = request.headers.get("X-Dashboard-Token") or request.query_params.get("token")
        if token != DASHBOARD_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing dashboard token")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        # Require token as query param to access dashboard
        token = request.query_params.get("token")
        if token != DASHBOARD_TOKEN:
            return HTMLResponse(
                '<html><body style="background:#0a0a0f;color:#e0e0e0;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh">'
                '<div style="text-align:center"><h1>NAOMI Dashboard</h1>'
                '<p>Access: <code>http://host:18802/?token=YOUR_TOKEN</code></p>'
                '<p style="color:#888">Token is in <code>data/dashboard_token.txt</code></p></div></body></html>',
                status_code=401,
            )
        html = DASHBOARD_HTML.replace("%%TOKEN%%", DASHBOARD_TOKEN)
        return HTMLResponse(html)

    @app.get("/api/status", dependencies=[Depends(verify_token)])
    async def get_status():
        return {
            "name": "NAOMI",
            "version": agent.config.get("agent", {}).get("version", "0.1.0"),
            "state": agent.heartbeat.state,
            "uptime": time.time() - agent.start_time,
            "beat_count": agent.heartbeat.beat_count,
            "memory_stats": {
                "short_term": len(agent.memory.recall_short(limit=100)),
                "long_term": len(agent.memory.recall_long(limit=100)),
                "skills": len(agent.memory.recall_skill() or []),
            },
            "tools": agent.tool_manager.list_tools(),
            "compaction": agent.compaction.get_status() if hasattr(agent, 'compaction') else {},
            "memory_agent": agent.memory_agent.get_status() if hasattr(agent, 'memory_agent') else {},
            "timestamp": time.time(),
        }

    @app.get("/api/tasks", dependencies=[Depends(verify_token)])
    async def get_tasks():
        return {"tasks": agent.memory.get_recent_tasks(50)}

    @app.get("/api/memory", dependencies=[Depends(verify_token)])
    async def get_memory():
        return {
            "short_term": agent.memory.recall_short(limit=20),
            "long_term": agent.memory.recall_long(limit=20),
            "conversations": agent.memory.get_conversations(limit=30),
        }

    @app.get("/api/skills", dependencies=[Depends(verify_token)])
    async def get_skills():
        skills = agent.memory.recall_skill()
        return {"skills": skills if isinstance(skills, list) else []}

    @app.get("/api/conversations", dependencies=[Depends(verify_token)])
    async def get_conversations():
        return {"conversations": agent.memory.get_conversations(50)}

    @app.post("/api/command", dependencies=[Depends(verify_token)])
    async def post_command(request: Request):
        data = await request.json()
        command = data.get("command", "")
        if not command:
            return JSONResponse({"error": "No command"}, status_code=400)
        await agent.submit_command(command)
        return {"status": "queued", "command": command}

    @app.get("/api/model", dependencies=[Depends(verify_token)])
    async def get_model():
        return {
            "current": agent.brain.get_model(),
            "available": agent.brain.list_models(),
        }

    @app.post("/api/model", dependencies=[Depends(verify_token)])
    async def set_model(request: Request):
        data = await request.json()
        model_name = data.get("model", "")
        if not model_name:
            return JSONResponse({"error": "No model specified"}, status_code=400)
        result = agent.brain.set_model(model_name)
        return result

    @app.get("/api/usage", dependencies=[Depends(verify_token)])
    async def get_usage():
        return {
            "brain": agent.brain.get_usage(),
            "scheduler": agent.scheduler.get_status() if hasattr(agent, 'scheduler') else {},
            "skills": agent.skills.get_status() if hasattr(agent, 'skills') else {},
        }

    @app.post("/api/council", dependencies=[Depends(verify_token)])
    async def council_debate(request: Request):
        data = await request.json()
        topic = data.get("topic", "")
        if not topic:
            return JSONResponse({"error": "No topic"}, status_code=400)
        result = agent.council.debate(topic)
        return result

    @app.get("/api/metrics", dependencies=[Depends(verify_token)])
    async def get_metrics():
        """Return aggregated observability metrics for the last 24 hours."""
        hours = 24
        return agent.memory.get_metrics_summary(hours)

    @app.post("/api/evolve", dependencies=[Depends(verify_token)])
    async def trigger_evolution(request: Request):
        result = agent.evolution.evolution_cycle()
        return result

    # ----- TTS API -----

    @app.post("/api/tts", dependencies=[Depends(verify_token)])
    async def synthesize_tts(request: Request):
        """Synthesize text to speech. Returns base64-encoded OGG audio."""
        data = await request.json()
        text = data.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        if len(text) > 2000:
            text = text[:2000]

        from core.tts import text_to_speech
        try:
            audio_path = await text_to_speech(text)
        except Exception as exc:
            logger.error("TTS synthesis error: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=500)

        if not audio_path or not os.path.exists(audio_path):
            return JSONResponse({"error": "TTS produced no output"}, status_code=500)

        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        size = os.path.getsize(audio_path)
        try:
            os.unlink(audio_path)
        except OSError:
            pass

        return {"audio": audio_b64, "format": "ogg", "size": size}

    @app.get("/api/tts/config", dependencies=[Depends(verify_token)])
    async def get_tts_config():
        """Return current TTS configuration."""
        from core.tts import _load_tts_config
        cfg = _load_tts_config()
        return {
            "backend": cfg["backend"],
            "kokoro": cfg["kokoro"],
            "qwen3": cfg["qwen3"],
            "edge": cfg["edge"],
        }

    @app.post("/api/tts/config", dependencies=[Depends(verify_token)])
    async def update_tts_config(request: Request):
        """Update TTS backend or voice settings at runtime (non-persistent)."""
        data = await request.json()
        from core.tts import _load_tts_config

        cfg = _load_tts_config()
        changed = []

        if "backend" in data and data["backend"] in ("kokoro", "qwen3", "edge", "auto"):
            cfg["backend"] = data["backend"]
            changed.append(f"backend={data['backend']}")

        for section in ("kokoro", "qwen3", "edge"):
            if section in data and isinstance(data[section], dict):
                cfg[section].update(data[section])
                changed.append(section)

        return {"status": "ok", "changed": changed, "config": cfg}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        # Require token as query param for WebSocket
        token = websocket.query_params.get("token")
        if token != DASHBOARD_TOKEN:
            await websocket.close(code=4001, reason="Invalid token")
            return
        await websocket.accept()
        connected_clients.add(websocket)
        try:
            while True:
                # Send status updates every 5 seconds
                status = {
                    "type": "status",
                    "state": agent.heartbeat.state,
                    "beat_count": agent.heartbeat.beat_count,
                    "uptime": int(time.time() - agent.start_time),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
                await websocket.send_json(status)

                # Check for new log entries
                recent = agent.memory.recall_short(limit=3)
                if recent:
                    await websocket.send_json({
                        "type": "log",
                        "entries": [{"content": r["content"], "category": r["category"]} for r in recent]
                    })

                await asyncio.sleep(5)
        except WebSocketDisconnect:
            connected_clients.discard(websocket)

    return app


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NAOMI Agent Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --bg: #0a0a0f; --surface: #12121a; --surface2: #1a1a2e;
  --accent: #00d4aa; --accent2: #7c3aed; --text: #e0e0e0;
  --text2: #888; --danger: #ef4444; --success: #22c55e;
  --border: #2a2a3e;
}
body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
.header {
  background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
  border-bottom: 1px solid var(--border); padding: 16px 24px;
  display: flex; align-items: center; justify-content: space-between;
}
.logo { font-size: 24px; font-weight: 700; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.status-badge { padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.status-active { background: rgba(0,212,170,0.15); color: var(--accent); border: 1px solid rgba(0,212,170,0.3); }
.status-idle { background: rgba(124,58,237,0.15); color: var(--accent2); border: 1px solid rgba(124,58,237,0.3); }
.status-error { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }

.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; padding: 16px 24px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.card h3 { font-size: 14px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.metric { font-size: 36px; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 13px; color: var(--text2); margin-top: 4px; }

.main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 0 24px 24px; }
.full-width { grid-column: 1 / -1; }

.command-input {
  display: flex; gap: 12px; padding: 16px 24px;
}
.command-input input {
  flex: 1; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 16px; color: var(--text); font-size: 15px;
  outline: none; transition: border-color 0.2s;
}
.command-input input:focus { border-color: var(--accent); }
.command-input button {
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border: none; border-radius: 8px; padding: 12px 24px; color: white;
  font-weight: 600; cursor: pointer; font-size: 14px; transition: opacity 0.2s;
}
.command-input button:hover { opacity: 0.9; }

.log-container { max-height: 400px; overflow-y: auto; }
.log-entry { padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px; font-family: 'Cascadia Code', monospace; }
.log-entry .time { color: var(--text2); margin-right: 8px; }
.log-entry .cat { color: var(--accent); margin-right: 8px; font-weight: 600; }
.log-entry.error .cat { color: var(--danger); }

.task-item { padding: 10px 12px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.task-status { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.task-status.completed { background: rgba(34,197,94,0.15); color: var(--success); }
.task-status.running { background: rgba(0,212,170,0.15); color: var(--accent); }
.task-status.failed { background: rgba(239,68,68,0.15); color: var(--danger); }
.task-status.pending { background: rgba(136,136,136,0.15); color: var(--text2); }

.btn-group { display: flex; gap: 8px; margin-top: 12px; }
.btn { padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border); background: var(--surface2); color: var(--text); cursor: pointer; font-size: 13px; transition: all 0.2s; }
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn-danger:hover { border-color: var(--danger); color: var(--danger); }

/* TTS Panel */
.tts-panel { padding: 16px 24px; }
.tts-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
.tts-card h3 { font-size: 14px; color: var(--text2); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
.tts-row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
.tts-field { display: flex; flex-direction: column; gap: 4px; }
.tts-field label { font-size: 11px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
.tts-field select, .tts-field input[type="text"] {
  background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 12px; color: var(--text); font-size: 13px; outline: none;
}
.tts-field select:focus, .tts-field input[type="text"]:focus { border-color: var(--accent); }
.tts-textarea {
  flex: 1; min-width: 300px; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; color: var(--text); font-size: 14px;
  resize: vertical; min-height: 60px; font-family: inherit; outline: none;
}
.tts-textarea:focus { border-color: var(--accent); }
.tts-btn {
  background: linear-gradient(135deg, #f59e0b, #ef4444); border: none; border-radius: 8px;
  padding: 10px 20px; color: white; font-weight: 600; cursor: pointer; font-size: 13px;
  white-space: nowrap; transition: opacity 0.2s;
}
.tts-btn:hover { opacity: 0.9; }
.tts-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.tts-audio { margin-top: 12px; width: 100%; }
.tts-audio audio { width: 100%; height: 36px; border-radius: 8px; }
.tts-status { font-size: 12px; color: var(--text2); margin-top: 8px; }

.conversation { padding: 10px 12px; border-bottom: 1px solid var(--border); }
.conversation .role { font-weight: 600; font-size: 12px; text-transform: uppercase; margin-bottom: 4px; }
.conversation .role.user { color: var(--accent2); }
.conversation .role.naomi { color: var(--accent); }
.conversation .role.system { color: var(--text2); }
.conversation .msg { font-size: 13px; line-height: 1.5; }

@media (max-width: 768px) {
  .grid { grid-template-columns: 1fr; }
  .main-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="header">
  <div class="logo">NAOMI Agent</div>
  <div>
    <span id="statusBadge" class="status-badge status-active">Initializing...</span>
    <span style="color: var(--text2); margin-left: 12px; font-size: 13px;" id="uptimeDisplay">Uptime: 0s</span>
  </div>
</div>

<div class="command-input">
  <input type="text" id="commandInput" placeholder="Tell NAOMI what to do... (e.g., 'Search for Python web scraping tutorials')" onkeydown="if(event.key==='Enter')sendCommand()">
  <button onclick="sendCommand()">Execute</button>
  <button onclick="triggerCouncil()" style="background: var(--accent2);">Council</button>
  <button onclick="triggerEvolve()" style="background: linear-gradient(135deg, #ef4444, #f97316);">Evolve</button>
</div>

<div class="tts-panel">
  <div class="tts-card">
    <h3>Voice Lab</h3>
    <div class="tts-row">
      <div class="tts-field">
        <label>Backend</label>
        <select id="ttsBackend" onchange="onTtsBackendChange()">
          <option value="kokoro">Kokoro (MLX)</option>
          <option value="qwen3">Qwen3-TTS (MLX)</option>
          <option value="edge">Edge TTS (Cloud)</option>
          <option value="auto">Auto</option>
        </select>
      </div>
      <div class="tts-field" id="ttsVoiceField">
        <label>Voice</label>
        <select id="ttsVoice">
          <optgroup label="Chinese">
            <option value="zf_xiaobei">Xiaobei (Female)</option>
            <option value="zm_yunxi">Yunxi (Male)</option>
          </optgroup>
          <optgroup label="English">
            <option value="af_heart">Heart (Female)</option>
            <option value="af_bella">Bella (Female)</option>
            <option value="am_adam">Adam (Male)</option>
          </optgroup>
          <optgroup label="Japanese">
            <option value="jf_alpha">Alpha (Female)</option>
            <option value="jm_kumo">Kumo (Male)</option>
          </optgroup>
        </select>
      </div>
      <div class="tts-field" id="ttsSpeakerField" style="display:none">
        <label>Speaker</label>
        <select id="ttsSpeaker">
          <option value="Vivian">Vivian (ZH Female)</option>
          <option value="Serena">Serena (ZH Female)</option>
          <option value="Ryan">Ryan (EN Male)</option>
          <option value="Aiden">Aiden (EN Male)</option>
        </select>
      </div>
      <div class="tts-field" id="ttsEmotionField" style="display:none">
        <label>Emotion</label>
        <input type="text" id="ttsEmotion" placeholder="e.g. happy, whisper..." style="width:140px">
      </div>
      <div class="tts-field" id="ttsEdgeVoiceField" style="display:none">
        <label>Edge Voice</label>
        <select id="ttsEdgeVoice">
          <option value="zh-TW-HsiaoChenNeural">HsiaoChen (TW Female)</option>
          <option value="zh-TW-YunJheNeural">YunJhe (TW Male)</option>
          <option value="zh-CN-XiaoxiaoNeural">Xiaoxiao (CN Female)</option>
          <option value="en-US-AriaNeural">Aria (EN Female)</option>
          <option value="ja-JP-NanamiNeural">Nanami (JP Female)</option>
        </select>
      </div>
    </div>
    <div class="tts-row" style="margin-top:12px">
      <textarea class="tts-textarea" id="ttsText" placeholder="Enter text to synthesize..." rows="2">NAOMI 隨時為你服務。</textarea>
      <button class="tts-btn" id="ttsBtn" onclick="synthesizeTTS()">Synthesize</button>
    </div>
    <div class="tts-audio" id="ttsAudioContainer" style="display:none">
      <audio id="ttsAudio" controls></audio>
    </div>
    <div class="tts-status" id="ttsStatus"></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3>Heartbeat</h3>
    <div class="metric" id="beatCount">0</div>
    <div class="metric-label">beats</div>
  </div>
  <div class="card">
    <h3>Memory</h3>
    <div class="metric" id="memoryCount">0</div>
    <div class="metric-label">memories stored</div>
  </div>
  <div class="card">
    <h3>Skills</h3>
    <div class="metric" id="skillCount">0</div>
    <div class="metric-label">skills learned</div>
  </div>
</div>

<div class="grid" id="metricsGrid" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
  <div class="card">
    <h3>Total Calls (24h)</h3>
    <div class="metric" id="metricsTotalCalls">-</div>
    <div class="metric-label">API calls</div>
  </div>
  <div class="card">
    <h3>Success Rate</h3>
    <div class="metric" id="metricsSuccessRate">-</div>
    <div class="metric-label">percent</div>
  </div>
  <div class="card">
    <h3>Avg Latency</h3>
    <div class="metric" id="metricsAvgLatency">-</div>
    <div class="metric-label">ms</div>
  </div>
  <div class="card">
    <h3>Calls by Backend</h3>
    <div id="metricsBackendBars" style="margin-top:8px;"></div>
  </div>
</div>

<div class="main-grid">
  <div class="card">
    <h3>Live Log</h3>
    <div class="log-container" id="logContainer"></div>
  </div>
  <div class="card">
    <h3>Recent Tasks</h3>
    <div id="taskList" style="max-height: 400px; overflow-y: auto;"></div>
  </div>
  <div class="card full-width">
    <h3>Conversation</h3>
    <div id="conversationList" style="max-height: 300px; overflow-y: auto;"></div>
  </div>
</div>

<script>
const WS_URL = `ws://${location.host}/ws`;
const API_URL = `http://${location.host}/api`;
const TOKEN = '%%TOKEN%%';
const AUTH_HEADERS = {'Content-Type': 'application/json', 'X-Dashboard-Token': TOKEN};
function esc(s){if(!s)return'';return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
let ws;

function connectWebSocket() {
  ws = new WebSocket(WS_URL + '?token=' + TOKEN);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'status') updateStatus(data);
    if (data.type === 'log') updateLog(data.entries);
  };
  ws.onclose = () => setTimeout(connectWebSocket, 3000);
  ws.onerror = () => ws.close();
}

function updateStatus(data) {
  const badge = document.getElementById('statusBadge');
  badge.textContent = data.state;
  badge.className = 'status-badge status-' + (data.state === 'error' ? 'error' : data.state === 'idle' ? 'idle' : 'active');
  document.getElementById('beatCount').textContent = data.beat_count;
  const h = Math.floor(data.uptime/3600);
  const m = Math.floor((data.uptime%3600)/60);
  const s = data.uptime%60;
  document.getElementById('uptimeDisplay').textContent = `Uptime: ${h}h ${m}m ${s}s`;
}

function updateLog(entries) {
  const container = document.getElementById('logContainer');
  entries.forEach(e => {
    const div = document.createElement('div');
    div.className = 'log-entry' + (e.category === 'error' ? ' error' : '');
    div.innerHTML = `<span class="time">${new Date().toLocaleTimeString()}</span><span class="cat">[${esc(e.category)}]</span>${esc(e.content)}`;
    container.prepend(div);
  });
  while (container.children.length > 100) container.removeChild(container.lastChild);
}

async function sendCommand() {
  const input = document.getElementById('commandInput');
  const cmd = input.value.trim();
  if (!cmd) return;
  input.value = '';

  // Add to conversation immediately
  addConversation('user', cmd);

  const resp = await fetch(API_URL + '/command', {
    method: 'POST', headers: AUTH_HEADERS,
    body: JSON.stringify({command: cmd})
  });
  const data = await resp.json();
  addConversation('system', `Command queued: ${data.status}`);
}

async function triggerCouncil() {
  const topic = prompt('Enter topic for NAOMI Council to debate:');
  if (!topic) return;
  addConversation('user', `[Council] ${topic}`);
  const resp = await fetch(API_URL + '/council', {
    method: 'POST', headers: AUTH_HEADERS,
    body: JSON.stringify({topic})
  });
  const data = await resp.json();
  addConversation('naomi', `Council result: ${data.consensus || JSON.stringify(data).substring(0, 500)}`);
}

async function triggerEvolve() {
  if (!confirm('Trigger NAOMI self-evolution cycle?')) return;
  addConversation('system', 'Evolution cycle triggered...');
  const resp = await fetch(API_URL + '/evolve', {method: 'POST', headers: AUTH_HEADERS});
  const data = await resp.json();
  addConversation('naomi', `Evolution complete: ${JSON.stringify(data).substring(0, 500)}`);
}

function addConversation(role, msg) {
  const list = document.getElementById('conversationList');
  const div = document.createElement('div');
  div.className = 'conversation';
  div.innerHTML = `<div class="role ${esc(role)}">${esc(role)}</div><div class="msg">${esc(msg)}</div>`;
  list.prepend(div);
}

async function refreshData() {
  try {
    const [status, tasks, memory] = await Promise.all([
      fetch(API_URL + '/status', {headers: AUTH_HEADERS}).then(r => r.json()),
      fetch(API_URL + '/tasks', {headers: AUTH_HEADERS}).then(r => r.json()),
      fetch(API_URL + '/memory', {headers: AUTH_HEADERS}).then(r => r.json()),
    ]);

    document.getElementById('memoryCount').textContent =
      (memory.short_term?.length || 0) + (memory.long_term?.length || 0);
    document.getElementById('skillCount').textContent = status.memory_stats?.skills || 0;

    // Update tasks
    const taskList = document.getElementById('taskList');
    taskList.innerHTML = (tasks.tasks || []).map(t =>
      `<div class="task-item"><span style="font-size:13px">${esc(t.task?.substring(0,60)) || '?'}</span><span class="task-status ${esc(t.status)}">${esc(t.status)}</span></div>`
    ).join('');

    // Update conversations
    if (memory.conversations?.length) {
      const convList = document.getElementById('conversationList');
      if (convList.children.length === 0) {
        memory.conversations.forEach(c => addConversation(c.role, c.content?.substring(0, 300)));
      }
    }
  } catch(e) { console.log('Refresh error:', e); }
}

// ----- TTS Functions -----
function onTtsBackendChange() {
  const b = document.getElementById('ttsBackend').value;
  document.getElementById('ttsVoiceField').style.display = b === 'kokoro' ? '' : 'none';
  document.getElementById('ttsSpeakerField').style.display = b === 'qwen3' ? '' : 'none';
  document.getElementById('ttsEmotionField').style.display = b === 'qwen3' ? '' : 'none';
  document.getElementById('ttsEdgeVoiceField').style.display = b === 'edge' ? '' : 'none';
  // Push config change to server
  const payload = {backend: b};
  if (b === 'kokoro') payload.kokoro = {voice: document.getElementById('ttsVoice').value};
  if (b === 'qwen3') payload.qwen3 = {speaker: document.getElementById('ttsSpeaker').value};
  if (b === 'edge') payload.edge = {voice: document.getElementById('ttsEdgeVoice').value};
  fetch(API_URL + '/tts/config', {method:'POST', headers: AUTH_HEADERS, body: JSON.stringify(payload)}).catch(e => console.warn('TTS config update failed:', e));
}

async function synthesizeTTS() {
  const text = document.getElementById('ttsText').value.trim();
  if (!text) return;
  const btn = document.getElementById('ttsBtn');
  const status = document.getElementById('ttsStatus');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  status.textContent = 'Synthesizing...';
  const t0 = performance.now();

  // Push latest voice selection before synthesizing
  const b = document.getElementById('ttsBackend').value;
  const cfgPayload = {backend: b};
  if (b === 'kokoro') cfgPayload.kokoro = {voice: document.getElementById('ttsVoice').value};
  if (b === 'qwen3') {
    cfgPayload.qwen3 = {speaker: document.getElementById('ttsSpeaker').value};
    const emo = document.getElementById('ttsEmotion').value.trim();
    if (emo) cfgPayload.qwen3.instruct = emo;
  }
  if (b === 'edge') cfgPayload.edge = {voice: document.getElementById('ttsEdgeVoice').value};
  await fetch(API_URL + '/tts/config', {method:'POST', headers: AUTH_HEADERS, body: JSON.stringify(cfgPayload)});

  try {
    const resp = await fetch(API_URL + '/tts', {
      method: 'POST', headers: AUTH_HEADERS,
      body: JSON.stringify({text})
    });
    const data = await resp.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
    } else {
      const audioEl = document.getElementById('ttsAudio');
      audioEl.src = 'data:audio/ogg;base64,' + data.audio;
      document.getElementById('ttsAudioContainer').style.display = '';
      audioEl.play().catch(() => { status.textContent += ' (Press play to listen)'; });
      status.textContent = `Done in ${elapsed}s | ${(data.size/1024).toFixed(1)} KB | backend: ${b}`;
    }
  } catch(e) {
    status.textContent = 'Request failed: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Synthesize';
}

// Load current TTS config on page load
async function loadTtsConfig() {
  try {
    const resp = await fetch(API_URL + '/tts/config', {headers: AUTH_HEADERS});
    const cfg = await resp.json();
    document.getElementById('ttsBackend').value = cfg.backend || 'kokoro';
    if (cfg.kokoro?.voice) document.getElementById('ttsVoice').value = cfg.kokoro.voice;
    if (cfg.qwen3?.speaker) document.getElementById('ttsSpeaker').value = cfg.qwen3.speaker;
    if (cfg.edge?.voice) document.getElementById('ttsEdgeVoice').value = cfg.edge.voice;
    onTtsBackendChange();
  } catch(e) { console.log('TTS config load error:', e); }
}

async function refreshMetrics() {
  try {
    const resp = await fetch(API_URL + '/metrics', {headers: AUTH_HEADERS});
    const m = await resp.json();
    document.getElementById('metricsTotalCalls').textContent = m.total_calls || 0;
    document.getElementById('metricsSuccessRate').textContent = (m.success_rate || 0) + '%';
    document.getElementById('metricsAvgLatency').textContent = m.avg_latency_ms || 0;
    // Render backend bar chart with inline CSS
    const bars = document.getElementById('metricsBackendBars');
    const backends = m.by_backend || {};
    const names = Object.keys(backends);
    if (names.length === 0) {
      bars.innerHTML = '<div style="color:var(--text2);font-size:12px">No data yet</div>';
      return;
    }
    const maxCalls = Math.max(...names.map(n => backends[n].calls || 0), 1);
    const colors = ['var(--accent)', 'var(--accent2)', '#f59e0b', '#ef4444', '#22c55e', '#3b82f6', '#ec4899'];
    bars.innerHTML = names.map((name, i) => {
      const b = backends[name];
      const pct = Math.round((b.calls / maxCalls) * 100);
      const c = colors[i % colors.length];
      return '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:12px">' +
        '<span style="width:55px;color:var(--text2)">' + esc(name) + '</span>' +
        '<div style="flex:1;background:var(--surface2);border-radius:4px;height:14px;overflow:hidden">' +
        '<div style="width:' + pct + '%;height:100%;background:' + c + ';border-radius:4px"></div></div>' +
        '<span style="width:30px;text-align:right;color:var(--text)">' + b.calls + '</span></div>';
    }).join('');
  } catch(e) { console.log('Metrics refresh error:', e); }
}

connectWebSocket();
refreshData();
refreshMetrics();
loadTtsConfig();
setInterval(refreshData, 10000);
setInterval(refreshMetrics, 15000);
</script>
</body>
</html>"""
