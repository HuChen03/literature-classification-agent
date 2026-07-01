from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .config import get_env, get_env_int
from .schema import ClassificationInput, ClassificationResult, Evidence


class OpenAICompatibleClient:
    def __init__(self) -> None:
        self.api_key = get_env("OPENAI_API_KEY")
        self.base_url = get_env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = get_env("OPENAI_MODEL", "gpt-4.1-mini")
        self.timeout_s = get_env_int("OPENAI_TIMEOUT_S", 60, 1, 600)
        self.max_retries = get_env_int("OPENAI_MAX_RETRIES", 2, 0, 10)
        self.retry_backoff_s = get_env_int("OPENAI_RETRY_BACKOFF_S", 2, 0, 60)

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a literature classification agent. "
                        "Return one valid JSON object only. Do not include Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        payload = self._post_with_retries(body)
        content = payload["choices"][0]["message"]["content"]
        parsed = parse_json_object(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("llm_returned_non_object_json")
        return parsed

    def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._post_once(body)
            except RuntimeError as error:
                last_error = error
                if not _is_retryable_error(str(error)) or attempt >= self.max_retries:
                    raise
            if self.retry_backoff_s > 0:
                time.sleep(self.retry_backoff_s * (2 ** attempt))
        raise last_error or RuntimeError("llm_request_failed")

    def _post_once(self, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf8"),
            headers={
                "authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf8", errors="replace")[:500]
            raise RuntimeError(f"llm_http_{error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"llm_request_failed: {error.reason}") from error


class LlmClassifier:
    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self._client = client or OpenAICompatibleClient()

    def classify(self, request: ClassificationInput, prompt: str | None = None) -> ClassificationResult:
        if not prompt:
            raise ValueError("prompt is required for LLM classification")
        parsed = self._client.complete_json(prompt)
        result = ClassificationResult.from_dict(parsed, fallback_mode=request.mode or "general", fallback_paper_id=request.paper.paper_id)
        return self._validate(result, request)

    def _validate(self, result: ClassificationResult, request: ClassificationInput) -> ClassificationResult:
        review_reasons = list(result.review_reasons)
        primary = result.primary_category
        secondary = result.secondary_categories
        if request.mode == "custom" and request.taxonomy:
            allowed = {item.id: item.name for item in request.taxonomy.categories}
            allowed_names = {item.name: item.id for item in request.taxonomy.categories}

            def normalize_category(category):
                if category.id in allowed:
                    return category
                if category.name in allowed_names:
                    return type(category)(id=allowed_names[category.name], name=category.name)
                return None

            if primary:
                normalized_primary = normalize_category(primary)
                if normalized_primary is None:
                    review_reasons.append("llm_primary_category_outside_taxonomy")
                primary = normalized_primary
            normalized_secondary = []
            for category in secondary:
                normalized = normalize_category(category)
                if normalized:
                    normalized_secondary.append(normalized)
                else:
                    review_reasons.append("llm_secondary_category_outside_taxonomy")
            secondary = normalized_secondary
            if primary is None:
                review_reasons.append("missing_allowed_primary_category")

        if not result.evidence:
            review_reasons.append("missing_evidence")
        if result.confidence < 0.7:
            review_reasons.append("low_confidence")
        return ClassificationResult(
            mode=request.mode or result.mode,
            paper_id=result.paper_id or request.paper.paper_id,
            primary_category=primary,
            secondary_categories=secondary,
            paper_type=result.paper_type,
            research_methods=result.research_methods,
            domains=result.domains,
            application_areas=result.application_areas,
            data_types=result.data_types,
            generated_keywords=result.generated_keywords,
            confidence=result.confidence,
            evidence=result.evidence[:5] or [Evidence(text=request.paper.title, reason="LLM did not provide evidence.")],
            needs_human_review=bool(review_reasons) or result.needs_human_review,
            review_reasons=_dedupe(review_reasons),
        )


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = json.loads(_extract_json_object_text(text))
    if not isinstance(parsed, dict):
        raise RuntimeError("json_repair_non_object")
    return parsed


def _extract_json_object_text(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("json_repair_no_object_found")
    return text[start : end + 1]


def _is_retryable_error(message: str) -> bool:
    retryable_markers = [
        "llm_http_408",
        "llm_http_409",
        "llm_http_429",
        "llm_http_500",
        "llm_http_502",
        "llm_http_503",
        "llm_http_504",
        "timed out",
        "timeout",
        "temporarily",
        "connection reset",
    ]
    lowered = message.lower()
    return any(marker in lowered for marker in retryable_markers)
