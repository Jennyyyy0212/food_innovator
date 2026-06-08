import json
import re
import time
from typing import Any, Callable, Optional


def _extract_text(response) -> str:
    """Extract text from a Gemini response, skipping thought/signature parts.

    Thinking models (gemini-flash-latest, gemini-2.5-*) include thought_signature
    parts in the response. Iterating parts directly avoids the SDK warning that
    fires when response.text concatenates non-text parts.
    """
    candidates = getattr(response, "candidates", None)
    if candidates:
        parts = getattr(candidates[0].content, "parts", []) or []
        text_parts = [
            p.text
            for p in parts
            if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
        ]
        if text_parts:
            return "".join(text_parts)
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return ""


def gemini_generate(
    client,
    model_name: str,
    prompt,
    max_retries: int = 4,
    on_retry: Optional[Callable[[int, int], None]] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> str:
    """Call Gemini with exponential-backoff retry on 503 / transient errors.

    on_retry(attempt, delay) — called before sleeping on a transient error.
    on_error(exception)      — called once on final failure.
    Returns the response text, or "" on failure.
    """
    delay = 2
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text_out = _extract_text(response)
            return text_out.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            last_error = e
            err_str = str(e)
            is_transient = any(
                code in err_str for code in ("503", "429", "UNAVAILABLE", "ResourceExhausted")
            )
            if is_transient and attempt < max_retries - 1:
                if on_retry:
                    on_retry(attempt, delay)
                time.sleep(delay)
                delay *= 2
            else:
                break
    if on_error:
        on_error(last_error)
    return ""


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in Markdown fences or contain extra whitespace."""
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return {}
