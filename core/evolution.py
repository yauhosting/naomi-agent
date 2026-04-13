"""
NAOMI Agent - Self-Evolution Engine
Multi-Agent Council: Multiple agents debate and reach consensus
Self-Modification: Reviews and improves its own code
Continuous Learning: Evolves towards better performance
"""
import os
import json
import time
import difflib
import logging
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("naomi.evolution")


class AgentCouncil:
    """
    Simulates multiple agents with different perspectives debating a topic.
    Each agent has a role and argues from that perspective.
    The council reaches consensus through structured debate.
    """

    COUNCIL_MEMBERS = [
        {
            "name": "Architect",
            "role": "System architect focused on clean design, scalability, and maintainability.",
            "bias": "Prefers elegant, well-structured solutions. May over-engineer.",
        },
        {
            "name": "Pragmatist",
            "role": "Practical engineer who values working code over perfect code.",
            "bias": "Prefers simple, quick solutions. May cut corners.",
        },
        {
            "name": "Security",
            "role": "Security expert who identifies vulnerabilities and risks.",
            "bias": "Conservative about changes. Always thinks about attack vectors.",
        },
        {
            "name": "Innovator",
            "role": "Creative thinker who pushes boundaries and tries new approaches.",
            "bias": "Loves novel solutions. May suggest untested approaches.",
        },
        {
            "name": "Critic",
            "role": "Devil's advocate who challenges assumptions and finds flaws.",
            "bias": "Skeptical of all proposals. Forces thorough consideration.",
        },
    ]

    def __init__(self, brain):
        self.brain = brain

    def debate(self, topic: str, context: str = "", rounds: int = 2) -> Dict[str, Any]:
        """
        Run a multi-agent debate on a topic.
        Each council member presents their view, then they synthesize.
        """
        logger.info(f"Council debate started: {topic}")
        debate_log = []

        # Round 1: Each member presents their perspective
        perspectives = []
        for member in self.COUNCIL_MEMBERS:
            system = f"""You are {member['name']}, a council member of NAOMI's decision-making body.
Role: {member['role']}
Tendency: {member['bias']}

Provide your perspective on the topic. Be concise (3-5 bullet points).
Consider both benefits and drawbacks of potential approaches."""

            prompt = f"Topic for discussion: {topic}"
            if context:
                prompt += f"\n\nContext: {context}"
            if perspectives:
                prompt += f"\n\nPrevious perspectives:\n"
                for p in perspectives:
                    prompt += f"\n{p['name']}: {p['view'][:300]}"

            response = self.brain.think(prompt)
            perspective = {"name": member["name"], "view": response}
            perspectives.append(perspective)
            debate_log.append(f"[{member['name']}] {response}")
            logger.info(f"Council - {member['name']} spoke")

        # Synthesis: Combine all perspectives into a decision
        all_views = "\n\n".join(
            f"**{p['name']}**:\n{p['view']}" for p in perspectives
        )

        synthesis_prompt = f"""You are NAOMI's decision synthesizer.
Multiple council members have debated this topic:

{all_views}

Your job:
1. Identify points of agreement
2. Resolve conflicts by weighing each perspective
3. Produce a FINAL DECISION with concrete action steps
4. Note any unresolved risks

Respond in JSON:
{{
  "consensus": "the agreed decision",
  "action_steps": ["step1", "step2"],
  "key_insights": ["insight1"],
  "unresolved_risks": ["risk1"],
  "confidence": 0.0-1.0
}}"""

        synthesis = self.brain.think(synthesis_prompt)
        try:
            if "```json" in synthesis:
                synthesis = synthesis.split("```json")[1].split("```")[0]
            elif "```" in synthesis:
                synthesis = synthesis.split("```")[1].split("```")[0]
            result = json.loads(synthesis)
        except (json.JSONDecodeError, IndexError):
            result = {
                "consensus": synthesis,
                "action_steps": [],
                "key_insights": [],
                "unresolved_risks": [],
                "confidence": 0.5,
            }

        result["debate_log"] = debate_log
        result["participants"] = [m["name"] for m in self.COUNCIL_MEMBERS]
        logger.info(f"Council consensus reached (confidence: {result.get('confidence', '?')})")
        return result


class SelfEvolution:
    """
    NAOMI's self-modification engine.
    Reviews its own code, proposes improvements, and applies them.
    """

    def __init__(self, brain, memory, project_dir: str):
        self.brain = brain
        self.memory = memory
        self.project_dir = project_dir
        self.council = AgentCouncil(brain)
        self.evolution_log = []

    def review_own_code(self, file_path: str) -> Dict[str, Any]:
        """Review a specific source file and suggest improvements."""
        full_path = os.path.join(self.project_dir, file_path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}"}

        with open(full_path, 'r') as f:
            code = f.read()

        # Ask the council to review
        topic = f"Code review of {file_path}"
        context = f"Current code:\n```python\n{code[:3000]}\n```"

        review = self.council.debate(topic, context)
        return {
            "file": file_path,
            "review": review,
            "timestamp": time.time(),
        }

    def propose_improvement(self, file_path: str, goal: str) -> Dict[str, Any]:
        """Propose a specific code improvement."""
        full_path = os.path.join(self.project_dir, file_path)
        if not os.path.exists(full_path):
            return {"error": f"File not found: {full_path}"}

        with open(full_path, 'r') as f:
            original = f.read()

        prompt = f"""Review this code and propose improvements for the goal: {goal}

File: {file_path}
```python
{original[:4000]}
```

Respond with the complete improved code. Only output the code, no explanations."""

        improved = self.brain.write_code(prompt)

        # Clean up the response
        if "```python" in improved:
            improved = improved.split("```python")[1].split("```")[0]
        elif "```" in improved:
            improved = improved.split("```")[1].split("```")[0]

        # Generate diff
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            improved.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        ))

        return {
            "file": file_path,
            "goal": goal,
            "diff": "".join(diff),
            "original_lines": len(original.splitlines()),
            "improved_lines": len(improved.splitlines()),
            "improved_code": improved,
            "timestamp": time.time(),
        }

    def apply_improvement(self, file_path: str, improved_code: str, reason: str) -> Dict[str, Any]:
        """Apply an improvement after council approval."""
        full_path = os.path.join(self.project_dir, file_path)

        # Always backup first
        backup_path = f"{full_path}.bak.{int(time.time())}"
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                original = f.read()
            with open(backup_path, 'w') as f:
                f.write(original)

        # Write improved code
        with open(full_path, 'w') as f:
            f.write(improved_code)

        # Log the evolution
        evolution_entry = {
            "file": file_path,
            "reason": reason,
            "backup": backup_path,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }
        self.evolution_log.append(evolution_entry)

        # Save to memory
        self.memory.remember_long(
            f"Evolution: {file_path}",
            f"Modified {file_path}: {reason}. Backup at {backup_path}",
            category="evolution",
            importance=8,
        )

        # Git commit the change
        try:
            subprocess.run(
                ["git", "add", file_path],
                cwd=self.project_dir, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", f"[NAOMI Evolution] {reason}\n\nFile: {file_path}\nAuto-improved by NAOMI Agent Council"],
                cwd=self.project_dir, capture_output=True
            )
            logger.info(f"Evolution committed: {file_path}")
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

        return evolution_entry

    def evolution_cycle(self) -> Dict[str, Any]:
        """
        Run a full evolution cycle:
        1. Scan all source files
        2. Council reviews current state
        3. Propose improvements
        4. Apply approved changes
        """
        logger.info("Starting evolution cycle...")

        # Find all Python source files
        source_files = []
        for root, dirs, files in os.walk(self.project_dir):
            # Skip data, __pycache__, .git
            dirs[:] = [d for d in dirs if d not in ('data', '__pycache__', '.git', 'templates')]
            for f in files:
                if f.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, f), self.project_dir)
                    source_files.append(rel_path)

        # Council decides what to improve
        file_list = "\n".join(f"- {f}" for f in source_files)
        topic = "What should NAOMI improve about itself next?"
        context = f"""NAOMI's source files:
{file_list}

Recent evolution log:
{json.dumps(self.evolution_log[-5:], indent=2, default=str)}

Guidelines:
- Focus on improvements that make NAOMI more autonomous
- Prioritize reliability and capability expansion
- Don't change things just for the sake of changing them
"""

        decision = self.council.debate(topic, context)

        result = {
            "cycle_time": datetime.now().isoformat(),
            "files_scanned": len(source_files),
            "council_decision": decision,
            "improvements_applied": [],
        }

        # Apply top action steps if council is confident
        if decision.get("confidence", 0) >= 0.7:
            for step in decision.get("action_steps", [])[:2]:  # Max 2 changes per cycle
                logger.info(f"Applying evolution step: {step}")
                self.memory.remember_short(f"Evolution: {step}", category="evolution")

        logger.info(f"Evolution cycle complete: {len(result['improvements_applied'])} changes")
        return result

    def get_evolution_history(self) -> List[Dict]:
        """Get the evolution history."""
        return self.evolution_log
