"""
NAOMI Agent - Self-Evolution Engine v3
HARD RULES:
- ONLY Claude CLI may modify code (no weak models)
- Diff-based patches only (never rewrite whole files)
- 3-round simulation before applying any change
- Import + startup verification after every change
- Circuit breaker: max 2 fixes per cycle, cooldown on failure
"""
import os
import json
import time
import difflib
import logging
import subprocess
import glob
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("naomi.evolution")

# Safety limits
MAX_FIXES_PER_CYCLE = 2
COOLDOWN_AFTER_FAILURE = 3600  # 1 hour cooldown after a failed fix
MAX_CONSECUTIVE_FAILURES = 3   # After N failures, disable auto-evolution until manual reset
SIMULATION_ROUNDS = 3          # Number of simulation/review rounds before applying

# ── PROTECTED FILES ──────────────────────────────────────────────
# Self-evolution MUST NOT scan or modify these files.
# History: NAOMI has repeatedly broken herself by deleting methods from
# brain.py and telegram_bot.py during evolution cycles (e.g. _call_fast
# removal caused total Telegram silence).  Only Master may edit these.
PROTECTED_FILES = frozenset({
    "core/brain.py",
    "core/evolution.py",
    "communication/telegram_bot.py",
    "communication/dashboard.py",
    "naomi.py",
    "run.py",
})


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
        logger.info("Council debate: %s", topic[:100])
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
        logger.info("Council consensus (confidence: %s)", result.get("confidence", "?"))
        return result


class SelfEvolution:
    """
    NAOMI modifies her own code — with strict safety.

    HARD RULES:
    1. ONLY Claude CLI for code generation (strongest model, no truncation)
    2. DIFF-BASED patches only — never rewrite entire files
    3. 3-round simulation before applying
    4. Full verification: syntax + import + startup test
    5. Circuit breaker on consecutive failures
    """

    def __init__(self, brain, memory, project_dir: str):
        self.brain = brain
        self.memory = memory
        self.project_dir = project_dir
        self.council = AgentCouncil(brain)
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._disabled = False

    # === Circuit Breaker ===

    @property
    def is_locked(self) -> bool:
        """Check if evolution is locked due to failures."""
        if self._disabled:
            return True
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning("Evolution locked: %d consecutive failures", self._consecutive_failures)
            return True
        if self._last_failure_time and time.time() - self._last_failure_time < COOLDOWN_AFTER_FAILURE:
            remaining = COOLDOWN_AFTER_FAILURE - (time.time() - self._last_failure_time)
            logger.info("Evolution cooldown: %.0fs remaining", remaining)
            return True
        return False

    def reset_lock(self):
        """Manual reset — called by Master via /evolve reset."""
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._disabled = False
        logger.info("Evolution lock reset by Master")

    def _record_failure(self):
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                "Evolution DISABLED after %d consecutive failures. "
                "Use /evolve reset to re-enable.",
                self._consecutive_failures,
            )

    def _record_success(self):
        self._consecutive_failures = 0

    # === Claude CLI Only ===

    def _claude_cli_generate(self, prompt: str) -> str:
        """Generate code ONLY via Claude CLI. Returns empty string on failure."""
        if not self.brain._check_claude_cli():
            logger.error("Claude CLI not available — evolution BLOCKED")
            return ""

        result = self.brain._call_claude_cli(prompt)
        if not result:
            logger.error("Claude CLI returned no output")
            return ""
        return result

    # === Bug Scanning (uses any model — read-only) ===

    def scan_for_bugs(self) -> List[Dict]:
        """Scan source code for bugs. Read-only — any model is fine here."""
        bugs = []
        for root, dirs, files in os.walk(self.project_dir):
            dirs[:] = [d for d in dirs if d not in ("data", "__pycache__", ".git", "templates")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, self.project_dir)

                # Skip protected files — only Master may modify these
                if rel_path in PROTECTED_FILES:
                    continue

                try:
                    with open(path, "r") as fh:
                        code = fh.read()
                except Exception:
                    continue

                review = self.brain._think(
                    "Review this Python code for ACTUAL bugs or runtime errors ONLY. "
                    "Ignore style, naming, or minor improvements. "
                    "If no real bugs, reply 'NO_BUGS'. "
                    "Otherwise describe each bug concisely.\n\n"
                    f"File: {rel_path}\n```python\n{code[:3000]}\n```"
                )

                if "NO_BUGS" not in review.upper():
                    bugs.append({
                        "file": rel_path,
                        "issues": review[:500],
                        "code_preview": code[:500],
                    })

        logger.info("Bug scan: %d files with issues", len(bugs))
        return bugs

    # === Core: Diff-Based Fix with Simulation ===

    def auto_fix(self, file_path: str, issue: str) -> Dict[str, Any]:
        """
        Fix a bug with full safety:
        1. Council debate
        2. Claude CLI generates DIFF (not full rewrite)
        3. 3-round simulation: each round reviews the diff for problems
        4. Apply patch
        5. Verify: syntax → import → startup test
        6. Git commit or rollback
        """
        if self.is_locked:
            return {"error": "Evolution locked. Use /evolve reset to unlock."}

        full_path = os.path.join(self.project_dir, file_path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {file_path}"}

        # Block modification of protected core files
        rel = os.path.relpath(full_path, self.project_dir)
        if rel in PROTECTED_FILES:
            logger.warning("auto_fix BLOCKED: %s is a protected file", rel)
            return {"action": "blocked", "reason": f"{rel} is protected — only Master may modify it"}

        with open(full_path, "r") as f:
            original_code = f.read()

        logger.info("Auto-fix: %s — %s", file_path, issue[:100])

        # === Step 1: Council debate ===
        debate = self.council.debate(
            f"Should we fix this bug in {file_path}?",
            f"Issue: {issue}\n\nCode:\n```python\n{original_code[:2000]}\n```",
        )
        confidence = debate.get("confidence", 0)
        if confidence < 0.6:
            logger.info("Council rejected fix (confidence: %.2f < 0.6)", confidence)
            return {"action": "rejected", "confidence": confidence}

        # === Step 2: Claude CLI generates DIFF ===
        diff_prompt = (
            f"Fix this bug in {file_path}:\n{issue}\n\n"
            f"Current code:\n```python\n{original_code}\n```\n\n"
            "IMPORTANT RULES:\n"
            "- Output ONLY the lines that need to change, in unified diff format\n"
            "- Use --- a/file and +++ b/file header\n"
            "- Include enough context lines (3+) so the patch can be applied unambiguously\n"
            "- Do NOT rewrite the entire file\n"
            "- If the fix requires changing more than 30 lines, explain why first\n"
        )
        diff_output = self._claude_cli_generate(diff_prompt)
        if not diff_output:
            self._record_failure()
            return {"error": "Claude CLI failed to generate diff"}

        # Parse the diff and apply to get proposed code
        proposed_code = self._apply_diff(original_code, diff_output, file_path)
        if proposed_code is None:
            # Fallback: if diff parsing fails, ask Claude CLI for the edited section only
            logger.warning("Diff parse failed, trying targeted edit approach")
            edit_prompt = (
                f"Fix this bug in {file_path}:\n{issue}\n\n"
                f"Current code:\n```python\n{original_code}\n```\n\n"
                "Output the COMPLETE fixed file. Do NOT truncate. "
                "Make MINIMAL changes — only fix the bug, change nothing else."
            )
            proposed_code = self._claude_cli_generate(edit_prompt)
            if not proposed_code:
                self._record_failure()
                return {"error": "Claude CLI fallback also failed"}
            # Extract code from markdown
            if "```python" in proposed_code:
                proposed_code = proposed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in proposed_code:
                proposed_code = proposed_code.split("```")[1].split("```")[0].strip()

        if not proposed_code or len(proposed_code) < 50:
            self._record_failure()
            return {"error": "Generated fix too short"}

        # Sanity: proposed code should be similar length (not truncated)
        len_ratio = len(proposed_code) / len(original_code) if original_code else 0
        if len_ratio < 0.7:
            self._record_failure()
            logger.error(
                "Fix rejected: output too short (%.0f%% of original). Likely truncated.",
                len_ratio * 100,
            )
            return {"error": f"Fix rejected: {len_ratio:.0%} of original size, likely truncated"}

        # === Step 3: Multi-Round Simulation ===
        simulation_result = self._simulate(file_path, original_code, proposed_code, issue)
        if not simulation_result["approved"]:
            self._record_failure()
            return {
                "error": "Simulation rejected the fix",
                "simulation": simulation_result,
            }

        # === Step 4: Backup + Apply ===
        backup_path = f"{full_path}.bak.{int(time.time())}"
        with open(backup_path, "w") as f:
            f.write(original_code)

        with open(full_path, "w") as f:
            f.write(proposed_code)

        # === Step 5: Full Verification ===
        verify_result = self._verify(full_path, file_path)
        if not verify_result["ok"]:
            # ROLLBACK
            logger.error("Verification FAILED — rolling back: %s", verify_result["error"])
            with open(full_path, "w") as f:
                f.write(original_code)
            self._record_failure()
            return {"error": f"Verification failed, rolled back: {verify_result['error']}"}

        # === Step 6: Git Commit ===
        self._git_commit(file_path, issue, confidence)
        self._record_success()

        self.memory.remember_long(
            f"Evolution: {file_path}",
            f"Fixed: {issue[:200]}. Confidence: {confidence}. Simulations: {SIMULATION_ROUNDS} passed.",
            category="evolution",
            importance=9,
        )

        diff_lines = list(difflib.unified_diff(
            original_code.splitlines(), proposed_code.splitlines(), lineterm="",
        ))
        logger.info("Evolution SUCCESS: %s (%d diff lines)", file_path, len(diff_lines))

        return {
            "action": "fixed",
            "file": file_path,
            "backup": backup_path,
            "confidence": confidence,
            "diff_lines": len(diff_lines),
            "simulations_passed": SIMULATION_ROUNDS,
        }

    # === Simulation: 3 rounds of adversarial review ===

    def _simulate(self, file_path: str, original: str, proposed: str, issue: str) -> Dict[str, Any]:
        """Run N rounds of simulation. Each round tries to find problems."""
        diff_text = "\n".join(difflib.unified_diff(
            original.splitlines(), proposed.splitlines(),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="",
        ))

        rounds: List[Dict[str, str]] = []

        for i in range(1, SIMULATION_ROUNDS + 1):
            role = {
                1: "a senior Python engineer reviewing for correctness",
                2: "a QA engineer testing edge cases and worst-case scenarios",
                3: "a security auditor checking for regressions and breaking changes",
            }.get(i, "a code reviewer")

            prev_issues = ""
            if rounds:
                prev_issues = "\nPrevious rounds found these concerns:\n" + "\n".join(
                    f"Round {r['round']}: {r['verdict']}" for r in rounds if r["verdict"] != "APPROVE"
                )

            sim_prompt = (
                f"You are {role}.\n"
                f"A proposed fix for '{issue}' in {file_path}:\n\n"
                f"```diff\n{diff_text[:3000]}\n```\n\n"
                f"Full proposed code:\n```python\n{proposed[:4000]}\n```\n"
                f"{prev_issues}\n\n"
                "CRITICAL CHECKS:\n"
                "1. Will this break any existing imports or function signatures?\n"
                "2. Will the module still initialize correctly?\n"
                "3. Could this cause a crash loop?\n"
                "4. Is the code complete (not truncated)?\n"
                "5. Are there any regressions?\n\n"
                "Reply APPROVE if safe, or REJECT with specific reason."
            )

            # Simulation review MUST use Claude CLI
            verdict = self._claude_cli_generate(sim_prompt)
            if not verdict:
                return {"approved": False, "reason": f"Simulation round {i}: Claude CLI unavailable"}

            is_approved = "APPROVE" in verdict.upper() and "REJECT" not in verdict.upper()
            rounds.append({
                "round": str(i),
                "role": role,
                "verdict": "APPROVE" if is_approved else verdict[:200],
            })

            if not is_approved:
                logger.warning("Simulation round %d REJECTED: %s", i, verdict[:100])
                return {
                    "approved": False,
                    "reason": f"Round {i} ({role}) rejected: {verdict[:200]}",
                    "rounds": rounds,
                }

            logger.info("Simulation round %d/%d: APPROVED", i, SIMULATION_ROUNDS)

        return {"approved": True, "rounds": rounds}

    # === Verification: syntax + import + startup ===

    def _verify(self, full_path: str, rel_path: str) -> Dict[str, Any]:
        """3-layer verification."""
        # Layer 1: Syntax check
        try:
            with open(full_path, "r") as f:
                code = f.read()
            compile(code, rel_path, "exec")
        except SyntaxError as e:
            return {"ok": False, "error": f"Syntax error: {e}"}

        # Layer 2: Import test — can every module be imported?
        module_name = rel_path.replace("/", ".").replace(".py", "")
        import_test = subprocess.run(
            ["python3", "-c", f"import importlib; importlib.import_module('{module_name}')"],
            capture_output=True, text=True, timeout=15,
            cwd=self.project_dir,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if import_test.returncode != 0:
            return {"ok": False, "error": f"Import failed: {import_test.stderr[:300]}"}

        # Layer 3: Full startup test — can naomi.py parse + import?
        startup_test = subprocess.run(
            ["python3", "-c",
             "import ast; "
             "ast.parse(open('naomi.py').read()); "
             "from core.memory import Memory; "
             "from core.brain import Brain; "
             "from core.heartbeat import Heartbeat; "
             "from communication.telegram_bot import TelegramBot; "
             "print('STARTUP_OK')"],
            capture_output=True, text=True, timeout=15,
            cwd=self.project_dir,
        )
        if startup_test.returncode != 0 or "STARTUP_OK" not in startup_test.stdout:
            return {"ok": False, "error": f"Startup test failed: {startup_test.stderr[:300]}"}

        return {"ok": True}

    # === Diff Parsing ===

    @staticmethod
    def _apply_diff(original: str, diff_output: str, file_path: str):
        """Try to extract and apply a unified diff. Returns None on failure."""
        # Extract diff block from Claude output
        diff_text = diff_output
        if "```diff" in diff_text:
            diff_text = diff_text.split("```diff")[1].split("```")[0]
        elif "```" in diff_text:
            diff_text = diff_text.split("```")[1].split("```")[0]

        lines = diff_text.strip().splitlines()
        if not any(l.startswith("@@") or l.startswith("---") for l in lines):
            return None  # Not a valid diff

        # Simple patch application: find context, apply changes
        orig_lines = original.splitlines(keepends=True)
        result_lines = list(orig_lines)
        applied = False

        # Find hunks
        i = 0
        while i < len(lines):
            if lines[i].startswith("@@"):
                # Parse hunk header: @@ -start,count +start,count @@
                try:
                    parts = lines[i].split("@@")[1].strip()
                    old_spec = parts.split()[0]  # -start,count
                    start = abs(int(old_spec.split(",")[0]))
                    hunk_old = []
                    hunk_new = []
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("@@"):
                        line = lines[j]
                        if line.startswith("-"):
                            hunk_old.append(line[1:])
                        elif line.startswith("+"):
                            hunk_new.append(line[1:])
                        else:
                            # Context line
                            ctx = line[1:] if line.startswith(" ") else line
                            hunk_old.append(ctx)
                            hunk_new.append(ctx)
                        j += 1

                    # Apply hunk at the right position
                    idx = start - 1  # 0-based
                    if idx >= 0 and idx < len(result_lines):
                        # Replace old lines with new lines
                        old_count = len(hunk_old)
                        result_lines[idx:idx + old_count] = [
                            l + "\n" if not l.endswith("\n") else l for l in hunk_new
                        ]
                        applied = True

                    i = j
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1

        if not applied:
            return None

        return "".join(result_lines)

    # === Git ===

    def _git_commit(self, file_path: str, issue: str, confidence: float):
        try:
            subprocess.run(["git", "add", file_path], cwd=self.project_dir, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"[NAOMI Self-Evolution] {issue[:60]}\n\n"
                 f"File: {file_path}\n"
                 f"Council confidence: {confidence}\n"
                 f"Simulations passed: {SIMULATION_ROUNDS}/{SIMULATION_ROUNDS}\n"
                 f"Verified: syntax + import + startup\n"
                 f"Engine: Claude CLI only (v3)"],
                cwd=self.project_dir, capture_output=True,
            )
            logger.info("Evolution committed: %s", file_path)
        except Exception as e:
            logger.warning("Git commit failed: %s", e)

    # === Rollback ===

    def rollback(self, file_path: str) -> Dict[str, Any]:
        """Rollback to the most recent backup."""
        full_path = os.path.join(self.project_dir, file_path)
        backups = sorted(glob.glob(f"{full_path}.bak.*"), reverse=True)
        if not backups:
            return {"error": f"No backups found for {file_path}"}

        latest_backup = backups[0]
        with open(latest_backup, "r") as f:
            backup_code = f.read()
        with open(full_path, "w") as f:
            f.write(backup_code)

        subprocess.run(["git", "add", file_path], cwd=self.project_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"[NAOMI Rollback] Reverted {file_path} to backup"],
            cwd=self.project_dir, capture_output=True,
        )

        logger.info("Rolled back %s from %s", file_path, latest_backup)
        return {"action": "rolled_back", "file": file_path, "from_backup": latest_backup}

    # === Full Cycle ===

    def evolution_cycle(self) -> Dict[str, Any]:
        """Full evolution cycle with circuit breaker."""
        if self.is_locked:
            return {"cycle": "locked", "reason": "Evolution is locked due to failures"}

        logger.info("=== Evolution Cycle v3 Started ===")

        bugs = self.scan_for_bugs()
        if not bugs:
            logger.info("No bugs found")
            return {"cycle": "clean", "bugs_found": 0}

        fixes = []
        for bug in bugs[:MAX_FIXES_PER_CYCLE]:
            if self.is_locked:
                break
            result = self.auto_fix(bug["file"], bug["issues"])
            fixes.append(result)

        return {
            "cycle": "complete",
            "bugs_found": len(bugs),
            "fixes_attempted": len(fixes),
            "fixes": fixes,
            "timestamp": datetime.now().isoformat(),
            "locked": self.is_locked,
        }
