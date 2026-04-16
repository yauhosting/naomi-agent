"""Structured output validation with auto-retry on parse failure."""
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("naomi.validator")


def parse_json_response(
    text: str,
    brain=None,
    retry_prompt: str = "",
    max_retries: int = 1,
) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response. On failure, retry with corrective prompt.

    Steps:
    1. Try to extract JSON from text (handle ```json blocks, find {} boundaries)
    2. If parse fails AND brain is provided, retry with error message
    3. Return parsed dict or None
    """
    parsed = _extract_json(text)
    if parsed is not None:
        return parsed

    # Retry with brain if available
    if brain is not None and max_retries > 0:
        error_msg = f"Failed to parse JSON from response. Raw text: {text[:500]}"
        corrective = retry_prompt or (
            "Your previous response was not valid JSON. "
            "Please respond with ONLY valid JSON, no markdown, no explanation. "
            f"Parse error context: {error_msg}"
        )
        logger.info("JSON parse failed, retrying with corrective prompt")
        try:
            retry_text = brain._think(corrective)
            if retry_text:
                parsed = _extract_json(retry_text)
                if parsed is not None:
                    return parsed
        except Exception as e:
            logger.debug("Retry parse failed: %s", e)

    logger.warning("JSON extraction failed after all attempts")
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try multiple strategies to extract JSON from text."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` blocks
    if "```json" in cleaned:
        try:
            block = cleaned.split("```json", 1)[1].split("```", 1)[0]
            result = json.loads(block.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, IndexError):
            pass

    # Strategy 3: Extract from ``` ... ``` blocks
    if "```" in cleaned:
        try:
            block = cleaned.split("```", 1)[1].split("```", 1)[0]
            result = json.loads(block.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, IndexError):
            pass

    # Strategy 4: Find outermost { ... } boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 5: Strip common prefixes/suffixes and retry
    # Some models add text like "Here is the JSON:" before the actual JSON
    for pattern in [
        r"^[^{]*({.*})[^}]*$",
    ]:
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    return None


def validate_against_schema(
    data: dict,
    required_fields: List[str],
    field_types: Optional[Dict[str, type]] = None,
) -> Dict[str, Any]:
    """Validate a parsed dict against expected schema.

    Returns {"valid": bool, "errors": [...], "data": cleaned_data}
    """
    errors: List[str] = []
    cleaned = dict(data)

    # Check required fields
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check field types
    if field_types:
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                actual = type(data[field]).__name__
                expected = expected_type.__name__
                errors.append(
                    f"Field '{field}' expected {expected}, got {actual}"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": cleaned,
    }
