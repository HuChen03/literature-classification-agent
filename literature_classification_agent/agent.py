from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .schema import (
    BatchClassificationResult,
    CategorySelection,
    ClassificationInput,
    ClassificationIntent,
    ClassificationResult,
    Evidence,
    LiteraturePaper,
    Taxonomy,
    TaxonomyCategory,
)
from .taxonomy import (
    DEFAULT_APPLICATION_AREAS,
    DEFAULT_DATA_TYPES,
    DEFAULT_DOMAINS,
    DEFAULT_PAPER_TYPES,
    DEFAULT_RESEARCH_METHODS,
)
from .batch import BatchRunner
from .loader import PaperLoader
from .router import IntentRouter


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")


@dataclass(frozen=True)
class _CategoryScore:
    category: TaxonomyCategory
    score: float
    evidence: list[Evidence]
    exclusion_hits: list[str]


class LiteratureClassificationAgent:
    """Single, standalone literature classification agent.

    The agent has two modes:
    - custom: strict classification against user-provided taxonomy only.
    - general: default open classification over built-in literature dimensions.
    """

    def classify(self, payload: ClassificationInput | dict) -> ClassificationResult:
        request = ClassificationInput.from_dict(payload) if isinstance(payload, dict) else payload
        if request.mode == "custom":
            if request.taxonomy is None:
                raise ValueError("custom mode requires taxonomy")
            return self._classify_custom(request.paper, request.taxonomy)
        return self._classify_general(request.paper)

    def run(self, payload: ClassificationIntent | dict, include_prompts: bool = False) -> BatchClassificationResult:
        intent = IntentRouter().route(payload) if isinstance(payload, dict) else payload
        if intent.needs_clarification:
            return BatchClassificationResult(intent=intent, items=[], include_prompts=include_prompts)
        papers = PaperLoader().load(intent)
        return BatchRunner(self).run(intent, papers, include_prompts=include_prompts)

    def _classify_custom(self, paper: LiteraturePaper, taxonomy: Taxonomy) -> ClassificationResult:
        scores = [self._score_category(paper, category) for category in taxonomy.categories]
        scores.sort(key=lambda item: item.score, reverse=True)
        best = scores[0] if scores else None
        second = scores[1] if len(scores) > 1 else None

        primary: CategorySelection | None = None
        secondary: list[CategorySelection] = []
        evidence: list[Evidence] = []
        review_reasons: list[str] = []

        if best and best.score > 0 and not best.exclusion_hits:
            primary = CategorySelection(id=best.category.id, name=best.category.name)
            evidence = best.evidence[:3]
        elif taxonomy.rules.allow_unknown:
            review_reasons.append("no_taxonomy_category_matched")
        else:
            review_reasons.append("no_allowed_category_matched")

        if primary and taxonomy.rules.allow_multiple_secondary_categories:
            for item in scores[1:]:
                if item.score <= 0 or item.exclusion_hits:
                    continue
                if item.score >= max(1.0, best.score * 0.55):
                    secondary.append(CategorySelection(id=item.category.id, name=item.category.name))
                if len(secondary) >= 3:
                    break

        confidence = self._custom_confidence(best.score if best else 0.0, second.score if second else 0.0, len(evidence))
        if confidence < taxonomy.rules.min_confidence_for_auto_accept:
            review_reasons.append("low_confidence")
        if best and best.exclusion_hits:
            review_reasons.append("best_category_hit_exclusion_terms")
        if best and second and second.score > 0 and abs(best.score - second.score) <= max(1.0, best.score * 0.20):
            review_reasons.append("ambiguous_between_top_categories")
        if len(self._combined_text(paper)) < 80:
            review_reasons.append("paper_text_too_short")
        if not evidence:
            review_reasons.append("missing_evidence")

        return ClassificationResult(
            mode="custom",
            paper_id=paper.paper_id,
            primary_category=primary,
            secondary_categories=secondary,
            confidence=confidence,
            evidence=evidence,
            needs_human_review=bool(review_reasons),
            review_reasons=_dedupe(review_reasons),
        )

    def _classify_general(self, paper: LiteraturePaper) -> ClassificationResult:
        text = self._combined_text(paper)
        paper_type, type_evidence = self._best_dimension(DEFAULT_PAPER_TYPES, text, "论文类型")
        methods, method_evidence = self._multi_dimension(DEFAULT_RESEARCH_METHODS, text, "研究方法")
        domains, domain_evidence = self._multi_dimension(DEFAULT_DOMAINS, text, "研究领域")
        data_types, data_evidence = self._multi_dimension(DEFAULT_DATA_TYPES, text, "数据类型")
        applications, app_evidence = self._multi_dimension(DEFAULT_APPLICATION_AREAS, text, "应用方向")

        evidence = (type_evidence + method_evidence + domain_evidence + data_evidence + app_evidence)[:5]
        review_reasons: list[str] = []
        if not paper_type:
            review_reasons.append("paper_type_unclear")
        if not domains:
            review_reasons.append("domain_unclear")
        if len(text) < 80:
            review_reasons.append("paper_text_too_short")
        if len(evidence) < 2:
            review_reasons.append("weak_evidence")

        confidence = min(0.92, 0.45 + 0.08 * len(evidence) + (0.10 if paper.abstract else 0.0) + (0.05 if paper.keywords else 0.0))
        if confidence < 0.70:
            review_reasons.append("low_confidence")

        return ClassificationResult(
            mode="general",
            paper_id=paper.paper_id,
            paper_type=paper_type,
            research_methods=methods,
            domains=domains,
            application_areas=applications,
            data_types=data_types,
            generated_keywords=self._generate_keywords(paper),
            confidence=confidence,
            evidence=evidence,
            needs_human_review=bool(review_reasons),
            review_reasons=_dedupe(review_reasons),
        )

    def _score_category(self, paper: LiteraturePaper, category: TaxonomyCategory) -> _CategoryScore:
        text = self._combined_text(paper)
        normalized_text = _normalize(text)
        positive_terms = self._category_terms(category)
        exclusion_terms = self._term_list(category.exclusions)

        score = 0.0
        evidence: list[Evidence] = []
        for term in positive_terms:
            hits = _count_term(normalized_text, term)
            if hits <= 0:
                continue
            weight = 3.0 if term in _normalize(category.name) else 1.0
            if term in [_normalize(item) for item in category.keywords]:
                weight += 1.0
            score += min(hits, 3) * weight
            sentence = self._find_sentence(text, term)
            if sentence:
                evidence.append(Evidence(text=sentence, reason=f"命中分类“{category.name}”的判定词：{term}"))

        exclusion_hits = [term for term in exclusion_terms if _count_term(normalized_text, term) > 0]
        score -= len(exclusion_hits) * 2.5
        return _CategoryScore(category=category, score=max(0.0, score), evidence=_dedupe_evidence(evidence), exclusion_hits=exclusion_hits)

    def _category_terms(self, category: TaxonomyCategory) -> list[str]:
        terms = self._term_list([category.name, category.definition, *category.examples, *category.keywords])
        return terms[:80]

    def _term_list(self, values: list[str]) -> list[str]:
        terms: list[str] = []
        for value in values:
            normalized = _normalize(value)
            if not normalized:
                continue
            terms.append(normalized)
            terms.extend(_tokenize(normalized))
        return _dedupe([item for item in terms if len(item) >= 2])

    def _best_dimension(self, dimension: dict[str, list[str]], text: str, reason_prefix: str) -> tuple[str | None, list[Evidence]]:
        ranked = self._rank_dimension(dimension, text, reason_prefix)
        if not ranked or ranked[0][1] <= 0:
            return None, []
        return ranked[0][0], ranked[0][2][:1]

    def _multi_dimension(self, dimension: dict[str, list[str]], text: str, reason_prefix: str) -> tuple[list[str], list[Evidence]]:
        ranked = self._rank_dimension(dimension, text, reason_prefix)
        selected = [(label, evidence) for label, score, evidence in ranked if score > 0][:4]
        return [label for label, _ in selected], [item for _, evidence in selected for item in evidence[:1]]

    def _rank_dimension(self, dimension: dict[str, list[str]], text: str, reason_prefix: str) -> list[tuple[str, float, list[Evidence]]]:
        normalized_text = _normalize(text)
        rows: list[tuple[str, float, list[Evidence]]] = []
        for label, terms in dimension.items():
            score = 0.0
            evidence: list[Evidence] = []
            for raw_term in terms:
                term = _normalize(raw_term)
                hits = _count_term(normalized_text, term)
                if hits <= 0:
                    continue
                score += min(hits, 3)
                sentence = self._find_sentence(text, term)
                if sentence:
                    evidence.append(Evidence(text=sentence, reason=f"{reason_prefix}判断为“{label}”：命中 {raw_term}"))
            rows.append((label, score, _dedupe_evidence(evidence)))
        return sorted(rows, key=lambda item: item[1], reverse=True)

    def _combined_text(self, paper: LiteraturePaper) -> str:
        return "\n".join([paper.title, paper.abstract, " ".join(paper.keywords), paper.text]).strip()

    def _find_sentence(self, text: str, normalized_term: str) -> str:
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            cleaned = sentence.strip()
            if cleaned and normalized_term in _normalize(cleaned):
                return cleaned[:360]
        return text.strip()[:360]

    def _generate_keywords(self, paper: LiteraturePaper) -> list[str]:
        base = [item for item in paper.keywords if item.strip()]
        tokens = _tokenize(_normalize(f"{paper.title} {paper.abstract} {paper.text[:2000]}"))
        stop = {"the", "and", "for", "with", "from", "this", "that", "study", "paper", "using", "based", "研究", "方法", "论文", "基于"}
        counts = Counter(token for token in tokens if token not in stop and len(token) >= 2)
        generated = [token for token, _ in counts.most_common(12)]
        return _dedupe(base + generated)[:12]

    def _custom_confidence(self, best_score: float, second_score: float, evidence_count: int) -> float:
        if best_score <= 0:
            return 0.0
        separation = (best_score - second_score) / max(best_score, 1.0)
        base = 0.52 + min(best_score, 10.0) * 0.035 + max(0.0, separation) * 0.25 + min(evidence_count, 3) * 0.04
        return min(0.95, max(0.0, base))


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(value: str) -> list[str]:
    return _WORD_RE.findall(value)


def _count_term(text: str, term: str) -> int:
    if not term:
        return 0
    if re.fullmatch(r"[a-z0-9_-]+", term):
        pattern = re.compile(rf"(?<![a-z0-9_-]){re.escape(term)}(?![a-z0-9_-])")
        return len(pattern.findall(text))
    return text.count(term)


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _dedupe_evidence(values: list[Evidence]) -> list[Evidence]:
    output: list[Evidence] = []
    seen: set[str] = set()
    for item in values:
        key = item.text
        if key and key not in seen:
            output.append(item)
            seen.add(key)
    return output
