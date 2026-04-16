# NAOMI Agent

**Neural Autonomous Multi-purpose Intelligent Operator**

NAOMI is a local autonomous agent for chat, coding, system actions, memory, tool use, browser/desktop control, and proactive background work.

Current version: **0.5.1**

## Architecture

```text
Brain          Claude CLI Opus alias, Codex GPT-5.4, MiniMax, GLM, Ollama
Memory         SQLite short-term, long-term, skills, persona, metrics
Heartbeat      Sense -> classify -> plan -> act -> verify -> remember
Planner        Plan-execute-reflect loop for multi-step work
Actions        Shell, Python, files, web, Git, packages, desktop control
Channels       Telegram, WhatsApp, local dashboard
Security       Audit log, secret scan, sensitive-command detection
```

## Model Routing

Claude CLI is configured with `brain.primary.cli_model: "opus"`.

Claude Code must be **2.1.111 or newer** for the `opus` alias to resolve to Opus 4.7. Check with:

```bash
claude --version
claude -p --model opus "Reply only: OK"
```

Default routing:

```text
Chat:  Codex GPT-5.4 -> MiniMax M2.7 -> Ollama -> GLM -> Claude CLI
Code:  Claude CLI -> Codex GPT-5.4 -> MiniMax M2.7 -> GLM 5.1 -> Ollama OmniCoder
```

## Setup

```bash
cd ~/Projects/naomi-agent
pip3 install -r requirements.txt
cp .env.example .env
python3 run.py
```

Fill `.env` with local secrets and private identifiers:

```text
TELEGRAM_BOT_TOKEN=
TELEGRAM_MASTER_ID=
WHATSAPP_MASTER_NUMBER=
MINIMAX_API_KEY=
GLM_API_KEY=
```

Do not commit `.env`, `data/`, logs, local databases, or generated backup files.

Dashboard:

```text
http://127.0.0.1:18802
```

The dashboard token is generated locally in `data/dashboard_token.txt`.

## Recent Fixes

- Claude CLI calls now pass the configured model alias, with a safe fallback to CLI default if the local CLI rejects the alias.
- Long model calls and tool executions run off the async event loop, so Telegram, WhatsApp, heartbeat, and planner stay responsive.
- Python action execution uses unique temporary files instead of a shared `/tmp/naomi_exec.py`.
- Security scan skips virtualenvs, caches, binaries, logs, and generated data to avoid false critical reports.
- Logs redact common token/API-key patterns and suppress noisy third-party HTTP request logging.
- Tracked config no longer stores private Telegram IDs or WhatsApp numbers.

## Verification

Useful checks before pushing:

```bash
python3 -m compileall -q core actions communication naomi.py run.py
git diff --check
python3 - <<'PY'
from core.security import run_security_scan
print(run_security_scan('.'))
PY
```

## License

MIT
