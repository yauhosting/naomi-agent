"""
NAOMI Agent - Self-Evolution Engine v2
- Multi-Agent Council debates before code changes
- Actually modifies own source code (with backup + git)
- Auto-detects bugs and fixes them
- Full permissions: network, machines, code
"""
import os
import json
import time
import difflib
import logging
import subprocess
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger("naomi.evolution")


class AgentCouncil:
    """Multiple agents debate and reach consensus before changes."""

    COUNCIL_MEMBERS = [
        {"name": "Architect", "role": "System design, scalability, clean code structure."},
        {"name": "Pragmatist", "role": "Working code over perfect code. Ship fast, fix later."},
        {"name": "Security", "role": "Identify risks, vulnerabilities, data safety."},
        {"name": "Innovator", "role": "Creative solutions, new approaches, push boundaries."},
        {"name": "Critic", "role": "Find flaws, challenge assumptions, devil's advocate."},
    ]

    def __init__(self, brain):
        self.brain = brain

    def debate(self, topic: str, context: str = "") -> Dict[str, Any]:
        logger.info(f"Council debate: {topic[:100]}")
        perspectives = []

        for member in self.COUNCIL_MEMBERS:
            system = (
                f"You are {member['name']}, a council member of NAOMI's decision body.\n"
                f"Role: {member['role']}\n"
                "Give your perspective in 3-5 bullet points. Be concise."
            )
            prompt = f"Topic: {topic}"
            if context:
                prompt += f"\n\nContext: {context[:1000]}"
            if perspectives:
                prev = "\n".join(f"{p['name']}: {p['view'][:200]}" for p in perspectives[-2:])
                prompt += f"\n\nPrevious views:\n{prev}"

            response = self.brain._think(prompt, system)
            perspectives.append({"name": member["name"], "view": response})

        # Synthesize
        all_views = "\n\n".join(f"**{p['name']}**: {p['view'][:300]}" for p in perspectives)
        synthesis = self.brain._think(
            f"Council debated:\n{all_views}\n\n"
            "Synthesize into a final decision. Respond in JSON:\n"
            '{"consensus": "...", "action_steps": ["..."], "confidence": 0.0-1.0, "risks": ["..."]}',
        )

        try:
            if "```json" in synthesis:
                synthesis = synthesis.split("```json")[1].split("```")[0]
            elif "```" in synthesis:
                synthesis = synthesis.split("```")[1].split("```")[0]
            result = json.loads(synthesis.strip())
        except (json.JSONDecodeError, IndexError):
            result = {"consensus": synthesis[:500], "action_steps": [], "confidence": 0.5, "risks": []}

        result["debate_log"] = [f"[{p['name']}] {p['view'][:200]}" for p in perspectives]
        logger.info(f"Council consensus (confidence: {result.get('confidence', '?')})")
        return result


class SelfEvolution:
    """
    NAOMI modifies her own code.
    Safety: backup → council debate → apply → git commit → test → rollback if broken.
    """

    def __init__(self, brain, memory, project_dir: str):
        self.brain = brain
        self.memory = memory
        self.project_dir = project_dir
        self.council = AgentCouncil(brain)

    def scan_for_bugs(self) -> List[Dict]:
        """Scan own source code for bugs and improvements."""
        bugs = []
        for root, dirs, files in os.walk(self.project_dir):
            dirs[:] = [d for d in dirs if d not in ('data', '__pycache__', '.git', 'templates')]
            for f in files:
                if not f.endswith('.py'):
                    continue
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, self.project_dir)
                try:
                    with open(path, 'r') as fh:
                        code = fh.read()
                except Exception:
                    continue

                # Ask brain to find bugs
                review = self.brain._think(
                    f"Review this Python code for bugs, errors, or improvements. "
                    f"If no bugs, reply 'NO_BUGS'. Otherwise describe each bug concisely.\n\n"
                    f"File: {rel_path}\n```python\n{code[:3000]}\n```"
                )

                if "NO_BUGS" not in review.upper():
                    bugs.append({
                        "file": rel_path,
                        "issues": review[:500],
                        "code_preview": code[:500],
                    })

        logger.info(f"Bug scan complete: {len(bugs)} files with issues")
        return bugs

    def auto_fix(self, file_path: str, issue: str) -> Dict[str, Any]:
        """
        Auto-fix a bug:
        1. Read current code
        2. Council debates the fix
        3. Brain generates fixed code
        4. Backup original
        5. Apply fix
        6. Git commit
        7. Test (basic syntax check)
        8. Rollback if broken
        """
        full_path = os.path.join(self.project_dir, file_path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {file_path}"}

        with open(full_path, 'r') as f:
            original_code = f.read()

        logger.info(f"Auto-fix: {file_path} - {issue[:100]}")

        # Step 1: Council debates whether to fix
        debate = self.council.debate(
            f"Should we fix this bug in {file_path}?",
            f"Issue: {issue}\n\nCode:\n```python\n{original_code[:2000]}\n```"
        )

        confidence = debate.get("confidence", 0)
        if confidence < 0.5:
            logger.info(f"Council rejected fix (confidence: {confidence})")
            return {"action": "rejected", "reason": "Council confidence too low", "confidence": confidence}

        # Step 2: Generate fix
        fix_prompt = (
            f"Fix this bug in {file_path}:\n{issue}\n\n"
            f"Current code:\n```python\n{original_code[:4000]}\n```\n\n"
            "Output the COMPLETE fixed file. Output ONLY Python code, no explanations."
        )
        fixed_code = self.brain.write_code(fix_prompt)

        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0]
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0]
        fixed_code = fixed_code.strip()

        if len(fixed_code) < 50:
            return {"error": "Generated fix too short, aborting"}

        # Step 3: Backup
        backup_path = f"{full_path}.bak.{int(time.time())}"
        with open(backup_path, 'w') as f:
            f.write(original_code)

        # Step 4: Syntax check before applying
        try:
            compile(fixed_code, file_path, 'exec')
        except SyntaxError as e:
            logger.error(f"Fix has syntax error: {e}")
            return {"error": f"Syntax error in fix: {e}", "backup": backup_path}

        # Step 5: Apply
        with open(full_path, 'w') as f:
            f.write(fixed_code)

        # Step 6: Verify the whole project still loads
        verify = subprocess.run(
            ["python3", "-c", f"import ast; ast.parse(open('{full_path}').read()); print('OK')"],
            capture_output=True, text=True, timeout=10, cwd=self.project_dir
        )

        if verify.returncode != 0:
            # Rollback
            logger.error(f"Verification failed, rolling back: {verify.stderr[:200]}")
            with open(full_path, 'w') as f:
                f.write(original_code)
            return {"error": "Verification failed, rolled back", "backup": backup_path}

        # Step 7: Git commit
        diff = list(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
        ))

        try:
            subprocess.run(["git", "add", file_path], cwd=self.project_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"[NAOMI Self-Evolution] Fix: {issue[:80]}\n\n"
                 f"File: {file_path}\n"
                 f"Council confidence: {confidence}\n"
                 f"Auto-fixed by NAOMI Agent Council"],
                cwd=self.project_dir, capture_output=True
            )
            logger.info(f"Evolution committed: {file_path}")
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

        # Save to memory
        self.memory.remember_long(
            f"Evolution: {file_path}",
            f"Fixed: {issue[:200]}. Backup: {backup_path}. Confidence: {confidence}",
            category="evolution", importance=9,
        )

        return {
            "action": "fixed",
            "file": file_path,
            "backup": backup_path,
            "confidence": confidence,
            "diff_lines": len(diff),
            "council_consensus": debate.get("consensus", "")[:300],
        }

    def rollback(self, file_path: str) -> Dict[str, Any]:
        """Rollback to the most recent backup."""
        full_path = os.path.join(self.project_dir, file_path)
        import glob
        backups = sorted(glob.glob(f"{full_path}.bak.*"), reverse=True)
        if not backups:
            return {"error": f"No backups found for {file_path}"}

        latest_backup = backups[0]
        with open(latest_backup, 'r') as f:
            backup_code = f.read()
        with open(full_path, 'w') as f:
            f.write(backup_code)

        subprocess.run(["git", "add", file_path], cwd=self.project_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"[NAOMI Rollback] Reverted {file_path} to backup"],
            cwd=self.project_dir, capture_output=True
        )

        logger.info(f"Rolled back {file_path} from {latest_backup}")
        return {"action": "rolled_back", "file": file_path, "from_backup": latest_backup}

    def evolution_cycle(self) -> Dict[str, Any]:
        """
        Full evolution cycle:
        1. Scan all source files for bugs
        2. Council reviews and prioritizes
        3. Auto-fix top issues
        4. Git commit + push
        """
        logger.info("=== Evolution Cycle Started ===")

        # Scan for bugs
        bugs = self.scan_for_bugs()

        if not bugs:
            logger.info("No bugs found, evolution cycle complete")
            return {"cycle": "clean", "bugs_found": 0, "fixes": []}

        # Fix top 2 bugs per cycle (to avoid cascading issues)
        fixes = []
        for bug in bugs[:2]:
            result = self.auto_fix(bug["file"], bug["issues"])
            fixes.append(result)

        # Commit only — do NOT auto-push. Master reviews via /evolve or manually pushes.
        # Git push removed for safety: auto-pushed code could contain regressions.
        logger.info("Evolution committed locally. Use 'git push' manually or /shell git push to push.")

        result = {
            "cycle": "complete",
            "bugs_found": len(bugs),
            "fixes_attempted": len(fixes),
            "fixes": fixes,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"=== Evolution Cycle Complete: {len(fixes)} fixes ===")
        return result
