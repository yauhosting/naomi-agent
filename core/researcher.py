"""NAOMI Agent - Deep Research Pipeline
Plan-and-execute research: decompose question -> parallel search -> cross-reference -> structured report.
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("naomi.researcher")


class DeepResearcher:
    """Autonomous research agent that produces structured reports."""

    def __init__(self, brain: Any, executor: Any) -> None:
        self.brain = brain
        self.executor = executor

    async def research(
        self,
        topic: str,
        depth: int = 3,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Full research loop:
        1. Decompose topic into sub-questions using brain._think()
        2. For each sub-question, run web_search + web_fetch in parallel
        3. Cross-reference findings, identify agreements and contradictions
        4. Produce structured report with citations

        Args:
            topic: Research topic or question
            depth: Number of sub-questions to generate (3-5, clamped)
            progress_callback: Optional async callable for progress updates
        Returns:
            {"success": bool, "topic": str, "sub_questions": [...],
             "findings": [{"question": str, "answer": str, "sources": [...]}],
             "report": str, "duration_seconds": float}
        """
        start_time = time.time()
        depth = max(2, min(depth, 5))

        # Step 1: Decompose topic into sub-questions
        decompose_prompt = (
            f"You are a research planner. Given the topic below, generate exactly {depth} "
            "specific sub-questions that, when answered together, would provide a comprehensive "
            "understanding of the topic. Output ONLY a JSON array of strings, no other text.\n\n"
            f"Topic: {topic}\n\n"
            f"Example output: [\"What is X?\", \"How does X compare to Y?\", \"What are the risks of X?\"]"
        )

        raw_questions = self.brain._think(decompose_prompt)
        sub_questions = self._parse_questions(raw_questions, topic, depth)

        if not sub_questions:
            return {
                "success": False,
                "topic": topic,
                "sub_questions": [],
                "findings": [],
                "report": "Failed to decompose topic into sub-questions.",
                "duration_seconds": time.time() - start_time,
            }

        # Step 2: Research each sub-question in parallel
        findings: List[Dict[str, Any]] = []
        tasks = []
        for i, question in enumerate(sub_questions):
            tasks.append(self._search_question(question, i, len(sub_questions), progress_callback))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error("Research sub-task failed: %s", result)
                continue
            if result:
                findings.append(result)

        # Step 3: Cross-reference and generate report
        if progress_callback:
            await progress_callback("Synthesizing findings into report...")

        report = await asyncio.to_thread(self._generate_report, topic, findings)

        duration = time.time() - start_time
        return {
            "success": len(findings) > 0,
            "topic": topic,
            "sub_questions": sub_questions,
            "findings": findings,
            "report": report,
            "duration_seconds": round(duration, 1),
        }

    async def _search_question(
        self,
        question: str,
        index: int,
        total: int,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Search for answers to a single sub-question.
        Uses web_search to find sources, web_fetch to read top results,
        then brain._think to synthesize an answer with citations.
        """
        if progress_callback:
            await progress_callback(f"Researching sub-question {index + 1}/{total}: {question[:60]}...")

        # Search for the question
        search_result = self.executor.web_search(question)
        if not search_result.get("success") or not search_result.get("results"):
            return {
                "question": question,
                "answer": "No search results found.",
                "sources": [],
                "raw_content": "",
            }

        search_results = search_result["results"][:4]
        sources = []
        fetched_content_parts: List[str] = []

        # Fetch top results in parallel
        fetch_tasks = []
        for sr in search_results:
            url = sr.get("href") or sr.get("url") or sr.get("link", "")
            title = sr.get("title", "")
            snippet = sr.get("body", sr.get("snippet", ""))
            if url:
                sources.append({"url": url, "title": title, "snippet": snippet})
                fetch_tasks.append(asyncio.to_thread(self.executor.web_fetch, url))
            elif snippet:
                sources.append({"url": "", "title": title, "snippet": snippet})
                fetched_content_parts.append(f"[{title}]: {snippet}")

        if fetch_tasks:
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for i, fetch_res in enumerate(fetch_results):
                if isinstance(fetch_res, Exception):
                    logger.debug("Fetch failed for source %d: %s", i, fetch_res)
                    continue
                if isinstance(fetch_res, dict) and fetch_res.get("success"):
                    content = fetch_res.get("content", "")[:3000]
                    title = fetch_res.get("title", sources[i].get("title", ""))
                    fetched_content_parts.append(f"[{title}]: {content}")

        # If no content fetched, use snippets from search results
        if not fetched_content_parts:
            for sr in search_results:
                snippet = sr.get("body", sr.get("snippet", ""))
                if snippet:
                    fetched_content_parts.append(f"[{sr.get('title', '')}]: {snippet}")

        combined_content = "\n\n---\n\n".join(fetched_content_parts)[:8000]

        # Synthesize answer using brain
        synthesis_prompt = (
            f"Based on the following sources, answer this question concisely:\n"
            f"Question: {question}\n\n"
            f"Sources:\n{combined_content}\n\n"
            "Provide a clear, factual answer in 2-4 paragraphs. "
            "Cite sources by their title in square brackets. "
            "Note any contradictions between sources. "
            "Respond in the same language as the question."
        )

        answer = await asyncio.to_thread(self.brain._think, synthesis_prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "raw_content": combined_content[:2000],
        }

    def _generate_report(self, topic: str, findings: List[Dict[str, Any]]) -> str:
        """Use brain._think to generate a final structured report."""
        if not findings:
            return f"No findings available for topic: {topic}"

        findings_text_parts = []
        all_sources: List[Dict[str, str]] = []
        for i, f in enumerate(findings, 1):
            findings_text_parts.append(
                f"## Sub-question {i}: {f['question']}\n{f['answer']}"
            )
            for src in f.get("sources", []):
                if src not in all_sources:
                    all_sources.append(src)

        findings_text = "\n\n".join(findings_text_parts)
        source_list = "\n".join(
            f"- [{s.get('title', 'Untitled')}]({s.get('url', '')})"
            for s in all_sources if s.get("url")
        )

        report_prompt = (
            f"You are writing a structured research report. "
            f"Compile the following research findings into a cohesive report.\n\n"
            f"Topic: {topic}\n\n"
            f"Findings:\n{findings_text[:6000]}\n\n"
            f"Sources:\n{source_list[:2000]}\n\n"
            "Write a structured report with:\n"
            "1. Executive Summary (2-3 sentences)\n"
            "2. Key Findings (bullet points)\n"
            "3. Areas of Agreement across sources\n"
            "4. Contradictions or Uncertainties\n"
            "5. Conclusion\n"
            "6. Sources list\n\n"
            "Respond in the same language as the topic. Be concise and factual."
        )

        report = self.brain._think(report_prompt)
        return report or f"Failed to generate report for: {topic}"

    @staticmethod
    def _parse_questions(raw: str, topic: str, depth: int) -> List[str]:
        """Parse LLM output into a list of sub-questions."""
        import json as _json

        # Try JSON parse first
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find JSON array in text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                questions = _json.loads(text[start : end + 1])
                if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                    return questions[:depth]
            except _json.JSONDecodeError:
                pass

        # Fallback: split by newlines and numbered items
        lines = text.split("\n")
        questions = []
        for line in lines:
            line = line.strip().lstrip("0123456789.-) ").strip('"').strip()
            if len(line) > 10 and "?" in line:
                questions.append(line)

        if questions:
            return questions[:depth]

        # Last resort: generate generic sub-questions
        return [
            f"What is {topic}?",
            f"What are the key aspects of {topic}?",
            f"What are the latest developments regarding {topic}?",
        ][:depth]
