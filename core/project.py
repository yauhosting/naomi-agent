"""
NAOMI Agent - Project Pipeline (0-to-1 Delivery)
Takes a high-level goal and autonomously delivers it through phases.

Flow:
  1. User gives goal: "做一個卡牌遊戲"
  2. NAOMI decomposes into phases with milestones
  3. Each phase executes via agent loop
  4. Phase verification before advancing
  5. State persists to disk — can resume after restart
  6. Final delivery + verification

Example phases for a card game:
  Phase 1: Design — game rules, card types, mechanics doc
  Phase 2: Assets — generate card art, UI design
  Phase 3: Engine — code the game logic, card system
  Phase 4: UI — build the interface
  Phase 5: Integration — connect everything
  Phase 6: Test — play test, fix bugs
  Phase 7: Polish — final adjustments, packaging
  Phase 8: Deliver — output + verification report
"""
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("naomi.project")

PROJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)


class ProjectPipeline:
    """Multi-phase project executor with persistent state."""

    def __init__(self, brain, executor, discovery=None):
        self.brain = brain
        self.executor = executor
        self.discovery = discovery

    def _project_file(self, project_id: str) -> str:
        return os.path.join(PROJECTS_DIR, f"{project_id}.json")

    def _load_project(self, project_id: str) -> Optional[Dict]:
        path = self._project_file(project_id)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _save_project(self, project: Dict):
        path = self._project_file(project["id"])
        with open(path, 'w') as f:
            json.dump(project, f, indent=2, ensure_ascii=False)

    # === Phase 1: Decompose goal into phases ===

    async def create(self, goal: str, work_dir: str = None) -> Dict:
        """Create a new project from a high-level goal."""
        import re
        project_id = re.sub(r'[^a-z0-9]', '-', goal[:40].lower()).strip('-')
        project_id = f"{project_id}-{int(time.time()) % 10000}"

        if not work_dir:
            work_dir = os.path.expanduser(f"~/Projects/{project_id}")

        logger.info(f"Creating project: {goal[:80]}")

        # Ask brain to decompose into phases
        decompose_schema = {
            "type": "object",
            "properties": {
                "project_name": {"type": "string"},
                "description": {"type": "string"},
                "tech_stack": {"type": "array", "items": {"type": "string"}},
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "tasks": {"type": "array", "items": {"type": "string"}},
                            "deliverables": {"type": "array", "items": {"type": "string"}},
                            "verification": {"type": "string"},
                        },
                        "required": ["id", "name", "tasks", "deliverables", "verification"],
                    },
                },
                "estimated_phases": {"type": "integer"},
            },
            "required": ["project_name", "phases"],
        }

        # Detect available tools for smarter planning
        available_engines = []
        if self.discovery:
            for name in ["godot", "unity-hub", "blender"]:
                if self.discovery.check_app(name):
                    available_engines.append(name)

        engine_hint = ""
        if available_engines:
            engine_hint = f"\nAvailable engines/tools on this machine: {', '.join(available_engines)}"
        else:
            engine_hint = (
                "\nNo game engines installed yet. Available to install: "
                "Godot (brew install --cask godot), Unity Hub (brew install --cask unity-hub). "
                "For simpler projects, use Python (pygame/arcade), web (HTML5/JS), or Godot. "
                "Include an installation phase if an engine is needed."
            )

        # Web development context
        web_hint = (
            "\nWeb development tools available: "
            "node/npm/npx (v22), python3 (3.14), git. "
            "Frameworks: can install Next.js, React, Vue, Vite, Tailwind via npx. "
            "Image generation: ComfyUI API at http://127.0.0.1:18801/api/generate (AI art on RTX 5070Ti GPU). "
            "Deployment: vercel, netlify, gh-pages, or local http server. "
            "For web projects, include: scaffold → code → assets → test → deploy phases."
        )

        result = self.brain._call_claude_cli(
            f"Decompose this project into 5-8 phases. Each phase should be independently executable.\n\n"
            f"Project goal: {goal}\n"
            f"{engine_hint}\n{web_hint}\n\n"
            f"For each phase, define: tasks (concrete steps), deliverables (files/outputs), "
            f"and verification (how to check it's done). "
            f"Use real tools and frameworks. Be specific about file paths relative to the project root. "
            f"Include tool/dependency installation as Phase 1 if needed.",
            json_schema=decompose_schema,
        )

        if not result:
            return {"success": False, "error": "Failed to decompose project"}

        try:
            plan = json.loads(result)
        except json.JSONDecodeError:
            return {"success": False, "error": "Failed to parse project plan"}

        # Build project state
        project = {
            "id": project_id,
            "goal": goal,
            "name": plan.get("project_name", goal[:40]),
            "description": plan.get("description", ""),
            "tech_stack": plan.get("tech_stack", []),
            "work_dir": work_dir,
            "status": "created",
            "current_phase": 0,
            "phases": [],
            "created_at": time.time(),
            "updated_at": time.time(),
            "total_steps_executed": 0,
        }

        for phase in plan.get("phases", []):
            project["phases"].append({
                "id": phase["id"],
                "name": phase["name"],
                "description": phase.get("description", ""),
                "tasks": phase.get("tasks", []),
                "deliverables": phase.get("deliverables", []),
                "verification": phase.get("verification", ""),
                "status": "pending",
                "result": None,
                "started_at": None,
                "completed_at": None,
            })

        self._save_project(project)
        logger.info(f"Project created: {project_id} ({len(project['phases'])} phases)")

        return {
            "success": True,
            "project_id": project_id,
            "name": project["name"],
            "phases": len(project["phases"]),
            "phase_names": [p["name"] for p in project["phases"]],
            "work_dir": work_dir,
        }

    # === Phase execution ===

    async def execute_next_phase(self, project_id: str) -> Dict:
        """Execute the next pending phase of a project."""
        project = self._load_project(project_id)
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}

        # Find next pending phase
        phase = None
        phase_idx = None
        for i, p in enumerate(project["phases"]):
            if p["status"] == "pending":
                phase = p
                phase_idx = i
                break

        if not phase:
            return {"success": True, "status": "all_phases_complete",
                    "project": project["name"]}

        logger.info(f"Executing phase {phase['id']}: {phase['name']}")

        # Create work directory
        os.makedirs(project["work_dir"], exist_ok=True)

        # Build context from previous phases
        prev_results = []
        for p in project["phases"][:phase_idx]:
            if p["status"] == "completed" and p.get("result"):
                prev_results.append(f"Phase {p['id']} ({p['name']}): {p['result'][:200]}")

        prev_context = "\n".join(prev_results) if prev_results else "This is the first phase."

        # Execute phase via agent loop
        task_prompt = (
            f"Project: {project['name']}\n"
            f"Goal: {project['goal']}\n"
            f"Work directory: {project['work_dir']}\n"
            f"Tech stack: {', '.join(project.get('tech_stack', []))}\n\n"
            f"Previous phases completed:\n{prev_context}\n\n"
            f"CURRENT PHASE {phase['id']}: {phase['name']}\n"
            f"Description: {phase.get('description', '')}\n"
            f"Tasks:\n" + "\n".join(f"  - {t}" for t in phase["tasks"]) + "\n"
            f"Deliverables:\n" + "\n".join(f"  - {d}" for d in phase["deliverables"]) + "\n"
            f"Verification: {phase['verification']}\n\n"
            f"Execute ALL tasks in this phase. Create all deliverable files. "
            f"Work in the project directory: {project['work_dir']}"
        )

        phase["status"] = "running"
        phase["started_at"] = time.time()
        project["current_phase"] = phase_idx
        project["status"] = "running"
        self._save_project(project)

        result = await self.brain.agent_loop(
            task=task_prompt,
            executor=self.executor,
            system_prompt=(
                f"You are NAOMI executing phase {phase['id']} of project '{project['name']}'. "
                f"Create all files in {project['work_dir']}. "
                f"Use shell commands to create directories, write files, install dependencies. "
                f"Be thorough — create real, working code, not placeholders."
            ),
            max_iterations=1,  # CLI does multi-step internally
        )

        # Update phase result
        phase["result"] = result.get("result", "")[:2000]
        phase["completed_at"] = time.time()
        project["total_steps_executed"] += len(result.get("steps", []))

        # Verify phase completion
        verification = await self._verify_phase(project, phase)

        if verification.get("passed"):
            phase["status"] = "completed"
            logger.info(f"Phase {phase['id']} completed: {phase['name']}")
        else:
            phase["status"] = "needs_review"
            phase["verification_notes"] = verification.get("notes", "")
            logger.warning(f"Phase {phase['id']} needs review: {verification.get('notes', '')}")

        # Check if all phases done
        all_done = all(p["status"] == "completed" for p in project["phases"])
        project["status"] = "completed" if all_done else "in_progress"
        project["updated_at"] = time.time()
        self._save_project(project)

        return {
            "success": True,
            "phase": phase["name"],
            "phase_status": phase["status"],
            "result": phase["result"][:500],
            "verification": verification,
            "project_status": project["status"],
            "phases_remaining": sum(1 for p in project["phases"] if p["status"] == "pending"),
        }

    async def _verify_phase(self, project: Dict, phase: Dict) -> Dict:
        """Verify a phase's deliverables exist."""
        missing = []
        found = []
        for deliverable in phase.get("deliverables", []):
            # Check if it looks like a file path
            if "/" in deliverable or "." in deliverable:
                path = os.path.join(project["work_dir"], deliverable)
                if os.path.exists(path):
                    found.append(deliverable)
                else:
                    missing.append(deliverable)
            else:
                found.append(deliverable)  # Non-file deliverable, assume done

        passed = len(missing) == 0
        return {
            "passed": passed,
            "found": found,
            "missing": missing,
            "notes": f"Missing: {', '.join(missing)}" if missing else "All deliverables verified",
        }

    # === Run all phases ===

    async def run_all(self, project_id: str, notify_callback=None) -> Dict:
        """Run all remaining phases of a project."""
        project = self._load_project(project_id)
        if not project:
            return {"success": False, "error": f"Project not found: {project_id}"}

        results = []
        for phase in project["phases"]:
            if phase["status"] != "pending":
                continue

            if notify_callback:
                await notify_callback(
                    f"🔨 Phase {phase['id']}/{len(project['phases'])}: {phase['name']}..."
                )

            result = await self.execute_next_phase(project_id)
            results.append(result)

            if notify_callback:
                status = "✅" if result.get("phase_status") == "completed" else "⚠️"
                await notify_callback(
                    f"{status} Phase {phase['id']}: {phase['name']} — {result.get('phase_status', '?')}"
                )

            # Reload project (may have been updated)
            project = self._load_project(project_id)

        return {
            "success": True,
            "project_id": project_id,
            "phases_executed": len(results),
            "project_status": project["status"],
            "results": results,
        }

    # === Project management ===

    def list_projects(self) -> List[Dict]:
        """List all projects."""
        projects = []
        for f in os.listdir(PROJECTS_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(PROJECTS_DIR, f), 'r') as fh:
                        p = json.load(fh)
                    done = sum(1 for ph in p["phases"] if ph["status"] == "completed")
                    total = len(p["phases"])
                    projects.append({
                        "id": p["id"],
                        "name": p["name"],
                        "status": p["status"],
                        "progress": f"{done}/{total}",
                        "work_dir": p["work_dir"],
                    })
                except Exception:
                    pass
        return projects

    def get_project(self, project_id: str) -> Optional[Dict]:
        return self._load_project(project_id)

    def delete_project(self, project_id: str) -> Dict:
        path = self._project_file(project_id)
        if os.path.exists(path):
            os.unlink(path)
            return {"success": True}
        return {"success": False, "error": "Not found"}
