"""
NAOMI Agent - Knowledge Importer
Import knowledge from OpenClaw and other sources into NAOMI's memory.
"""
import os
import json
import logging
from typing import Dict, List

logger = logging.getLogger("naomi.knowledge")


def import_openclaw_knowledge(memory, openclaw_dir: str = "/Users/yokowai/.openclaw"):
    """Import knowledge from OpenClaw's webui and config."""
    imported = 0

    # 1. Import tool/page inventory from webui
    webui_dir = os.path.join(openclaw_dir, "webui")
    if os.path.exists(webui_dir):
        tools = []
        for f in os.listdir(webui_dir):
            if f.endswith('.html') and f != 'index.html':
                name = f.replace('.html', '').replace('-', ' ').title()
                tools.append(name)

        if tools:
            memory.remember_long(
                "OpenClaw Tools Inventory",
                f"Available OpenClaw webui tools: {', '.join(tools)}",
                category="knowledge", importance=7,
                tags=["openclaw", "tools"]
            )
            imported += 1
            logger.info(f"Imported {len(tools)} OpenClaw tools")

    # 2. Import API server knowledge
    api_server = os.path.join(webui_dir, "api_server.py")
    if os.path.exists(api_server):
        with open(api_server, 'r') as f:
            first_100 = f.readlines()[:100]
        imports_and_routes = [l.strip() for l in first_100
                             if 'import' in l or '@app.' in l or 'def ' in l]
        memory.remember_long(
            "OpenClaw API Server Structure",
            f"API server at {api_server}\nKey components:\n" + "\n".join(imports_and_routes[:30]),
            category="knowledge", importance=6,
            tags=["openclaw", "api"]
        )
        imported += 1

    # 3. Import Hermes config knowledge
    hermes_config = os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")
    if os.path.exists(hermes_config):
        with open(hermes_config, 'r') as f:
            config_text = f.read()[:1000]
        memory.remember_long(
            "Hermes Agent Config",
            f"Hermes configuration:\n{config_text}",
            category="knowledge", importance=6,
            tags=["hermes", "config"]
        )
        imported += 1

    # 4. Import NAOMI's SOUL.md personality from Hermes
    soul_md = os.path.join(os.path.expanduser("~"), ".hermes", "SOUL.md")
    if os.path.exists(soul_md):
        with open(soul_md, 'r') as f:
            soul = f.read()[:2000]
        memory.remember_long(
            "NAOMI SOUL (from Hermes)",
            soul,
            category="persona", importance=9,
            tags=["soul", "personality"]
        )
        imported += 1

    # 5. Import SSH config (available machines)
    ssh_config = os.path.join(os.path.expanduser("~"), ".ssh", "config")
    if os.path.exists(ssh_config):
        with open(ssh_config, 'r') as f:
            hosts = [l.strip() for l in f if l.strip().startswith('Host ')]
        memory.remember_long(
            "Available SSH Machines",
            f"SSH hosts:\n" + "\n".join(hosts),
            category="infrastructure", importance=7,
            tags=["ssh", "machines"]
        )
        imported += 1

    # 6. Import Hermes skills inventory
    hermes_skills_dir = os.path.join(os.path.expanduser("~"), ".hermes", "hermes-agent", "skills")
    if os.path.exists(hermes_skills_dir):
        skills = [d for d in os.listdir(hermes_skills_dir)
                  if os.path.isdir(os.path.join(hermes_skills_dir, d))]
        memory.remember_long(
            "Hermes Skills Inventory",
            f"Available Hermes skills: {', '.join(skills)}",
            category="knowledge", importance=6,
            tags=["hermes", "skills"]
        )
        imported += 1

    # 7. Import OpenClaw agents
    agents_dir = os.path.join(openclaw_dir, "agents")
    if os.path.exists(agents_dir):
        agents = os.listdir(agents_dir)
        memory.remember_long(
            "OpenClaw Agents",
            f"Available agents: {', '.join(agents)}",
            category="knowledge", importance=5,
            tags=["openclaw", "agents"]
        )
        imported += 1

    logger.info(f"Knowledge import complete: {imported} items imported")
    return {"imported": imported}


def import_project_knowledge(memory, project_dir: str):
    """Import knowledge about the NAOMI project itself."""
    # Read own README
    readme = os.path.join(project_dir, "README.md")
    if os.path.exists(readme):
        with open(readme, 'r') as f:
            content = f.read()[:1500]
        memory.remember_long(
            "NAOMI Project README",
            content,
            category="self", importance=8,
            tags=["naomi", "readme"]
        )

    # Inventory own source files
    source_files = []
    for root, dirs, files in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in ('data', '__pycache__', '.git')]
        for f in files:
            if f.endswith('.py'):
                rel = os.path.relpath(os.path.join(root, f), project_dir)
                source_files.append(rel)

    memory.remember_long(
        "NAOMI Source Files",
        f"My source code files:\n" + "\n".join(source_files),
        category="self", importance=8,
        tags=["naomi", "source"]
    )

    return {"source_files": len(source_files)}
