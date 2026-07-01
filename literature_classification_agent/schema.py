from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ClassificationMode = Literal["custom", "general"]
SourceType = Literal["inline", "file", "directory"]
OutputFormat = Literal["json", "jsonl", "csv"]


@dataclass(frozen=True)
class LiteraturePaper:
    title: str
    paper_id: str | None = None
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiteraturePaper":
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError("paper.title is required")
        keywords_raw = payload.get("keywords") or []
        keywords = [str(item).strip() for item in keywords_raw if str(item).strip()] if isinstance(keywords_raw, list) else []
        metadata_raw = payload.get("metadata") or {}
        return cls(
            paper_id=str(payload.get("paper_id") or payload.get("paperId") or "").strip() or None,
            title=title,
            abstract=str(payload.get("abstract") or "").strip(),
            keywords=keywords,
            text=str(payload.get("text") or "").strip(),
            metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
        )


@dataclass(frozen=True)
class TaxonomyCategory:
    id: str
    name: str
    definition: str = ""
    examples: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaxonomyCategory":
        category_id = str(payload.get("id") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not category_id:
            raise ValueError("taxonomy.categories[].id is required")
        if not name:
            raise ValueError("taxonomy.categories[].name is required")
        return cls(
            id=category_id,
            name=name,
            definition=str(payload.get("definition") or "").strip(),
            examples=_string_list(payload.get("examples")),
            exclusions=_string_list(payload.get("exclusions")),
            keywords=_string_list(payload.get("keywords")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "definition": self.definition,
            "examples": self.examples,
            "exclusions": self.exclusions,
            "keywords": self.keywords,
        }


@dataclass(frozen=True)
class TaxonomyRules:
    single_primary_category: bool = True
    allow_multiple_secondary_categories: bool = False
    allow_unknown: bool = False
    min_confidence_for_auto_accept: float = 0.70

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TaxonomyRules":
        if not payload:
            return cls()
        return cls(
            single_primary_category=bool(payload.get("single_primary_category", payload.get("singlePrimaryCategory", True))),
            allow_multiple_secondary_categories=bool(
                payload.get("allow_multiple_secondary_categories", payload.get("allowMultipleSecondaryCategories", False))
            ),
            allow_unknown=bool(payload.get("allow_unknown", payload.get("allowUnknown", False))),
            min_confidence_for_auto_accept=_bounded_float(
                payload.get("min_confidence_for_auto_accept", payload.get("minConfidenceForAutoAccept", 0.70)),
                0.0,
                1.0,
                0.70,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_primary_category": self.single_primary_category,
            "allow_multiple_secondary_categories": self.allow_multiple_secondary_categories,
            "allow_unknown": self.allow_unknown,
            "min_confidence_for_auto_accept": self.min_confidence_for_auto_accept,
        }


@dataclass(frozen=True)
class Taxonomy:
    categories: list[TaxonomyCategory]
    rules: TaxonomyRules = field(default_factory=TaxonomyRules)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Taxonomy":
        categories_raw = payload.get("categories") or []
        if not isinstance(categories_raw, list) or not categories_raw:
            raise ValueError("taxonomy.categories must be a non-empty list")
        categories = [TaxonomyCategory.from_dict(item) for item in categories_raw if isinstance(item, dict)]
        if not categories:
            raise ValueError("taxonomy.categories must contain valid categories")
        return cls(categories=categories, rules=TaxonomyRules.from_dict(payload.get("rules") if isinstance(payload.get("rules"), dict) else None))

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": [item.to_dict() for item in self.categories],
            "rules": self.rules.to_dict(),
        }


@dataclass(frozen=True)
class ClassificationInput:
    paper: LiteraturePaper
    mode: ClassificationMode | None = None
    taxonomy: Taxonomy | None = None
    user_instruction: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClassificationInput":
        paper_raw = payload.get("paper")
        if not isinstance(paper_raw, dict):
            raise ValueError("paper object is required")
        taxonomy_raw = payload.get("taxonomy")
        taxonomy = Taxonomy.from_dict(taxonomy_raw) if isinstance(taxonomy_raw, dict) else None
        raw_mode = str(payload.get("mode") or "").strip().lower()
        mode: ClassificationMode | None
        if raw_mode in {"custom", "general"}:
            mode = raw_mode  # type: ignore[assignment]
        else:
            mode = "custom" if taxonomy else "general"
        if mode == "custom" and taxonomy is None:
            raise ValueError("custom mode requires taxonomy")
        return cls(
            paper=LiteraturePaper.from_dict(paper_raw),
            mode=mode,
            taxonomy=taxonomy,
            user_instruction=str(payload.get("user_instruction") or payload.get("userInstruction") or "").strip(),
        )


@dataclass(frozen=True)
class ClassificationIntent:
    mode: ClassificationMode
    source_type: SourceType
    source_path: str | None = None
    papers: list[LiteraturePaper] = field(default_factory=list)
    taxonomy: Taxonomy | None = None
    user_instruction: str = ""
    output_format: OutputFormat = "json"
    max_workers: int = 4
    needs_clarification: bool = False
    clarification_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "paper_count": len(self.papers),
            "taxonomy": self.taxonomy.to_dict() if self.taxonomy else None,
            "user_instruction": self.user_instruction,
            "output_format": self.output_format,
            "max_workers": self.max_workers,
            "needs_clarification": self.needs_clarification,
            "clarification_questions": self.clarification_questions,
        }


@dataclass(frozen=True)
class CategorySelection:
    id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CategorySelection | None":
        if not isinstance(payload, dict):
            return None
        category_id = str(payload.get("id") or "").strip()
        name = str(payload.get("name") or "").strip()
        if not category_id and not name:
            return None
        return cls(id=category_id or name, name=name or category_id)


@dataclass(frozen=True)
class Evidence:
    text: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Evidence | None":
        text = str(payload.get("text") or payload.get("quote") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if not text and not reason:
            return None
        return cls(text=text, reason=reason)


@dataclass(frozen=True)
class ClassificationResult:
    mode: ClassificationMode
    paper_id: str | None
    primary_category: CategorySelection | None = None
    secondary_categories: list[CategorySelection] = field(default_factory=list)
    paper_type: str | None = None
    research_methods: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    application_areas: list[str] = field(default_factory=list)
    data_types: list[str] = field(default_factory=list)
    generated_keywords: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    needs_human_review: bool = True
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "paper_id": self.paper_id,
            "primary_category": self.primary_category.to_dict() if self.primary_category else None,
            "secondary_categories": [item.to_dict() for item in self.secondary_categories],
            "paper_type": self.paper_type,
            "research_methods": self.research_methods,
            "domains": self.domains,
            "application_areas": self.application_areas,
            "data_types": self.data_types,
            "generated_keywords": self.generated_keywords,
            "confidence": round(self.confidence, 3),
            "evidence": [item.to_dict() for item in self.evidence],
            "needs_human_review": self.needs_human_review,
            "review_reasons": self.review_reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], fallback_mode: ClassificationMode, fallback_paper_id: str | None = None) -> "ClassificationResult":
        raw_mode = str(payload.get("mode") or fallback_mode).strip().lower()
        mode: ClassificationMode = "custom" if raw_mode == "custom" else "general"
        secondary_raw = payload.get("secondary_categories") or payload.get("secondaryCategories") or []
        evidence_raw = payload.get("evidence") or []
        return cls(
            mode=mode,
            paper_id=str(payload.get("paper_id") or payload.get("paperId") or fallback_paper_id or "").strip() or None,
            primary_category=CategorySelection.from_dict(payload.get("primary_category") or payload.get("primaryCategory")),
            secondary_categories=[
                item
                for item in (CategorySelection.from_dict(row) for row in secondary_raw if isinstance(row, dict))
                if item is not None
            ],
            paper_type=str(payload.get("paper_type") or payload.get("paperType") or "").strip() or None,
            research_methods=_string_list(payload.get("research_methods") or payload.get("researchMethods")),
            domains=_string_list(payload.get("domains")),
            application_areas=_string_list(payload.get("application_areas") or payload.get("applicationAreas")),
            data_types=_string_list(payload.get("data_types") or payload.get("dataTypes")),
            generated_keywords=_string_list(payload.get("generated_keywords") or payload.get("generatedKeywords")),
            confidence=_bounded_float(payload.get("confidence"), 0.0, 1.0, 0.0),
            evidence=[item for item in (Evidence.from_dict(row) for row in evidence_raw if isinstance(row, dict)) if item is not None],
            needs_human_review=bool(payload.get("needs_human_review", payload.get("needsHumanReview", True))),
            review_reasons=_string_list(payload.get("review_reasons") or payload.get("reviewReasons")),
        )


@dataclass(frozen=True)
class PaperClassificationJobResult:
    paper_id: str | None
    title: str
    status: Literal["success", "failed"]
    result: ClassificationResult | None = None
    error: str | None = None
    prompt: str | None = None

    def to_dict(self, include_prompt: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "paper_id": self.paper_id,
            "title": self.title,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
        }
        if include_prompt:
            payload["prompt"] = self.prompt
        return payload


@dataclass(frozen=True)
class BatchClassificationResult:
    intent: ClassificationIntent
    items: list[PaperClassificationJobResult]
    include_prompts: bool = False

    def to_dict(self) -> dict[str, Any]:
        success_count = sum(1 for item in self.items if item.status == "success")
        failed_count = len(self.items) - success_count
        intent_payload = self.intent.to_dict()
        intent_payload["paper_count"] = len(self.items)
        return {
            "intent": intent_payload,
            "summary": {
                "total": len(self.items),
                "success_count": success_count,
                "failed_count": failed_count,
            },
            "items": [item.to_dict(include_prompt=self.include_prompts) for item in self.items],
        }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _bounded_float(value: Any, lower: float, upper: float, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return min(max(number, lower), upper)
