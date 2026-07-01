from __future__ import annotations

import json
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

    def complete_json(self, prompt: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when CLASSIFIER_BACKEND=llm")
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
                payload = json.loads(response.read().decode("utf8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf8", errors="replace")[:500]
            raise RuntimeError(f"llm_http_{error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"llm_request_failed: {error.reason}") from error
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("llm_returned_non_object_json")
        return parsed


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
