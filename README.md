# NAOMI Agent

**Neural Autonomous Multi-purpose Intelligent Operator**

A digital life form that thinks, acts, and evolves autonomously.

## Architecture

```
Brain (Dual-Brain)     — Claude CLI + MiniMax fallback
Memory (SQLite)        — Short-term, long-term, skills, persona
Heartbeat              — Never-stopping life cycle
Evolution              — Multi-agent council + self-modification
Actions                — Shell, files, web, git, packages
Senses                 — System monitor, file watcher, web scanner
Dashboard              — Web-based real-time control panel
```

## Features

- **Autonomous Execution**: Give a command, NAOMI breaks it down and executes
- **Self-Evolution**: Multi-agent council debates improvements, modifies own code
- **Tool Auto-Install**: Missing a tool? NAOMI finds and installs it
- **Proactive Mode**: When idle, suggests improvements and new projects
- **Persistent Memory**: Remembers everything across sessions
- **Web Dashboard**: Real-time monitoring and control

## Quick Start

```bash
cd ~/Projects/naomi-agent
pip3 install -r requirements.txt
python3 naomi.py
```

Dashboard: `http://localhost:18802`

## Inspired By

- Hermes Agent: Tool system, skill management
- OpenHands: Autonomous execution loop
- CrewAI: Multi-agent collaboration
- Claude Agent SDK: MCP tool integration

## License

MIT
