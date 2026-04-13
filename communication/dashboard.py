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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
    async def index():
        # Dashboard HTML doesn't need auth (token is embedded in JS)
        html = DASHBOARD_HTML.replace("%%TOKEN%%", DASHBOARD_TOKEN)
        return html

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

    @app.post("/api/council", dependencies=[Depends(verify_token)])
    async def council_debate(request: Request):
        data = await request.json()
        topic = data.get("topic", "")
        if not topic:
            return JSONResponse({"error": "No topic"}, status_code=400)
        result = agent.council.debate(topic)
        return result

    @app.post("/api/evolve", dependencies=[Depends(verify_token)])
    async def trigger_evolution(request: Request):
        result = agent.evolution.evolution_cycle()
        return result

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
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
let ws;

function connectWebSocket() {
  ws = new WebSocket(WS_URL);
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
    div.innerHTML = `<span class="time">${new Date().toLocaleTimeString()}</span><span class="cat">[${e.category}]</span>${e.content}`;
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
  div.innerHTML = `<div class="role ${role}">${role}</div><div class="msg">${msg}</div>`;
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
      `<div class="task-item"><span style="font-size:13px">${t.task?.substring(0,60) || '?'}</span><span class="task-status ${t.status}">${t.status}</span></div>`
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

connectWebSocket();
refreshData();
setInterval(refreshData, 10000);
</script>
</body>
</html>"""
