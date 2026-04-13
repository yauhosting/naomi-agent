"""
NAOMI Agent - Self-Learning Skill System
Inspired by Hermes Agent's skill creation nudge + Claude Code's continuous learning.

Closed loop:
1. NAOMI completes a complex task (3+ tool calls)
2. Background review extracts reusable patterns
3. Skill saved as SKILL.md with structured metadata
4. Next time a similar task comes, skill is loaded into context
5. Cross-session experience accumulation via skills directory

Skills are stored as markdown files in skills/ directory:
  skills/
    fetch-crypto-price/
      SKILL.md
    generate-qr-code/
      SKILL.md
"""
import os
import json
import time
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("naomi.skills")

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
os.makedirs(SKILLS_DIR, exist_ok=True)

# Skill creation prompt — given to brain after complex task completion
SKILL_EXTRACTION_PROMPT = """You just completed a task. Analyze the steps taken and extract a reusable skill.

Task: {task}
Steps taken:
{steps}
Result: {result}

If this task involved a reusable pattern (can be applied to similar tasks in the future),
create a skill document. Reply in JSON:
{{
  "should_create": true,
  "name": "kebab-case-name",
  "description": "One-line description",
  "tags": ["tag1", "tag2"],
  "prerequisites": ["pip package or tool needed"],
  "when_to_use": "When user asks for...",
  "procedure": "Step-by-step instructions for how to do this task",
  "example_commands": ["shell command 1", "shell command 2"],
  "learned_from": "What went wrong and was fixed during execution"
}}

If this task is too specific or trivial (< 3 steps, or purely informational), reply:
{{"should_create": false, "reason": "why"}}

Reply with ONLY the JSON."""

# Skill matching prompt — find relevant skills for a new task
SKILL_MATCH_PROMPT = """Available skills:
{skills_index}

New task: {task}

Which skills are relevant? Reply with a JSON array of skill names, or [] if none match.
Reply with ONLY the JSON array."""


class SkillManager:
    """Manages NAOMI's learned skills — create, store, retrieve, match."""

    def __init__(self, brain=None):
        self.brain = brain
        self._skills_cache = {}
        self._load_skills()

    def _load_skills(self):
        """Load all skills from disk into cache."""
        self._skills_cache = {}
        if not os.path.exists(SKILLS_DIR):
            return

        for dirname in os.listdir(SKILLS_DIR):
            skill_dir = os.path.join(SKILLS_DIR, dirname)
            skill_file = os.path.join(skill_dir, "SKILL.md")
            if os.path.isdir(skill_dir) and os.path.exists(skill_file):
                try:
                    with open(skill_file, 'r') as f:
                        content = f.read()
                    meta = self._parse_frontmatter(content)
                    meta["_dir"] = skill_dir
                    meta["_content"] = content
                    self._skills_cache[dirname] = meta
                except Exception as e:
                    logger.warning(f"Failed to load skill {dirname}: {e}")

        logger.info(f"Loaded {len(self._skills_cache)} skills")

    def _parse_frontmatter(self, content: str) -> dict:
        """Parse YAML frontmatter from SKILL.md."""
        if not content.startswith("---"):
            return {"name": "", "description": "", "body": content}

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {"name": "", "description": "", "body": content}

        frontmatter = parts[1].strip()
        body = parts[2].strip()

        meta = {"body": body}
        for line in frontmatter.split("\n"):
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                # Parse list values
                if val.startswith("[") and val.endswith("]"):
                    try:
                        val = json.loads(val)
                    except json.JSONDecodeError:
                        val = [v.strip() for v in val[1:-1].split(",")]
                meta[key] = val
        return meta

    # === Skill Creation (Self-Learning) ===

    def extract_skill_from_task(self, task: str, steps: List[Dict],
                                result: str) -> Optional[Dict]:
        """After completing a complex task, extract a reusable skill."""
        if not self.brain:
            return None

        # Only extract from tasks with 3+ successful tool calls
        successful_steps = [s for s in steps if s.get("success") or s.get("tool") == "task_complete"]
        if len(successful_steps) < 3:
            return None

        # Format steps for the prompt
        steps_text = "\n".join(
            f"  {i+1}. [{s.get('tool', '?')}] {json.dumps(s.get('input', s.get('result', '')))[:150]}"
            for i, s in enumerate(steps)
        )

        prompt = SKILL_EXTRACTION_PROMPT.format(
            task=task[:300],
            steps=steps_text[:2000],
            result=str(result)[:500],
        )

        response = self.brain._call_claude_cli(prompt, system_prompt=(
            "You are a skill extraction engine. Analyze completed tasks and output reusable skills as JSON. "
            "Reply with ONLY JSON, no other text."
        ), bare=True)

        if not response:
            return None

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            raw = response.strip()
            if not raw.startswith("{"):
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]
            data = json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            logger.debug(f"Skill extraction parse failed: {response[:200]}")
            return None

        if not data.get("should_create"):
            logger.debug(f"Skill extraction declined: {data.get('reason', '?')}")
            return None

        # Save the skill
        return self._save_skill(data)

    def _save_skill(self, data: dict) -> Dict:
        """Save an extracted skill to disk."""
        name = data.get("name", "")
        if not name:
            return {"success": False, "error": "No skill name"}

        # Sanitize name
        name = re.sub(r'[^a-z0-9\-]', '-', name.lower()).strip('-')
        if not name:
            return {"success": False, "error": "Invalid skill name"}

        # Check if skill already exists
        if name in self._skills_cache:
            # Update existing skill
            skill_dir = self._skills_cache[name]["_dir"]
            logger.info(f"Updating existing skill: {name}")
        else:
            skill_dir = os.path.join(SKILLS_DIR, name)
            os.makedirs(skill_dir, exist_ok=True)

        # Build SKILL.md content
        tags_str = json.dumps(data.get("tags", []))
        prereqs_str = json.dumps(data.get("prerequisites", []))
        examples = data.get("example_commands", [])

        content = f"""---
name: {name}
description: {data.get('description', '')}
version: 1.0.0
author: NAOMI (auto-learned)
tags: {tags_str}
prerequisites: {prereqs_str}
created: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

# {data.get('description', name)}

## When to Use

{data.get('when_to_use', 'When a similar task is requested.')}

## Procedure

{data.get('procedure', 'No procedure documented.')}

## Example Commands

{chr(10).join(f'```bash{chr(10)}{cmd}{chr(10)}```' for cmd in examples) if examples else 'No examples.'}

## Lessons Learned

{data.get('learned_from', 'No issues encountered.')}
"""

        skill_file = os.path.join(skill_dir, "SKILL.md")
        with open(skill_file, 'w') as f:
            f.write(content)

        # Update cache
        meta = self._parse_frontmatter(content)
        meta["_dir"] = skill_dir
        meta["_content"] = content
        self._skills_cache[name] = meta

        logger.info(f"Skill saved: {name} ({data.get('description', '')})")
        return {"success": True, "name": name, "description": data.get("description", ""),
                "path": skill_file}

    # === Skill Retrieval ===

    def get_skill(self, name: str) -> Optional[Dict]:
        """Get a skill by name."""
        return self._skills_cache.get(name)

    def list_skills(self) -> List[Dict]:
        """List all available skills."""
        return [
            {"name": name, "description": meta.get("description", ""),
             "tags": meta.get("tags", []), "created": meta.get("created", "")}
            for name, meta in self._skills_cache.items()
        ]

    def find_relevant_skills(self, task: str) -> List[Dict]:
        """Find skills relevant to a task using keyword matching + brain."""
        if not self._skills_cache:
            return []

        # Quick keyword match first
        task_lower = task.lower()
        matches = []
        for name, meta in self._skills_cache.items():
            desc = meta.get("description", "").lower()
            tags = [t.lower() if isinstance(t, str) else str(t).lower()
                    for t in meta.get("tags", [])]
            when = meta.get("body", "").lower()

            # Check if any keyword matches
            score = 0
            for word in task_lower.split():
                if len(word) < 3:
                    continue
                if word in name:
                    score += 3
                if word in desc:
                    score += 2
                if any(word in t for t in tags):
                    score += 2
                if word in when:
                    score += 1

            if score > 0:
                matches.append({"name": name, "score": score, **meta})

        # Sort by score, return top 3
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:3]

    def get_skill_context(self, task: str) -> str:
        """Get relevant skill content to inject into agent context."""
        relevant = self.find_relevant_skills(task)
        if not relevant:
            return ""

        parts = ["=== Relevant Skills ==="]
        for skill in relevant:
            body = skill.get("body", "")
            parts.append(f"\n### {skill['name']}: {skill.get('description', '')}\n{body[:500]}")

        return "\n".join(parts)

    # === Skill Management ===

    def delete_skill(self, name: str) -> Dict:
        """Delete a skill."""
        if name not in self._skills_cache:
            return {"success": False, "error": f"Skill not found: {name}"}

        import shutil
        skill_dir = self._skills_cache[name]["_dir"]
        shutil.rmtree(skill_dir, ignore_errors=True)
        del self._skills_cache[name]
        logger.info(f"Skill deleted: {name}")
        return {"success": True, "name": name}

    def get_status(self) -> Dict:
        """Get skill system status."""
        return {
            "total_skills": len(self._skills_cache),
            "skills_dir": SKILLS_DIR,
            "skills": [{"name": n, "description": m.get("description", "")}
                       for n, m in self._skills_cache.items()],
        }
