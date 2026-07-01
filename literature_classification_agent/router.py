from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import ClassificationIntent, LiteraturePaper, OutputFormat, SourceType, Taxonomy, TaxonomyCategory, TaxonomyRules


_PATH_RE = re.compile(r"(?P<path>(?:\.{1,2}|/|~)[^\s，。；;]+|[A-Za-z0-9_.\-/]+?\.(?:jsonl?|csv|txt|md))")


class IntentRouter:
    def route(self, payload: str | dict[str, Any]) -> ClassificationIntent:
        if isinstance(payload, str):
            payload = {"request": payload}
        user_text = str(payload.get("request") or payload.get("user_request") or payload.get("userRequest") or "").strip()
        taxonomy = self._resolve_taxonomy(payload, user_text)
        mode = self._resolve_mode(payload, taxonomy, user_text)
        source_type, source_path, papers = self._resolve_source(payload, user_text)
        output_format = self._resolve_output_format(payload)
        max_workers = self._resolve_max_workers(payload.get("max_workers", payload.get("maxWorkers", 4)))

        questions: list[str] = []
        if source_type != "inline" and not source_path:
            questions.append("请提供要分类的论文文件或目录路径。")
        if mode == "custom" and taxonomy is None:
            questions.append("请提供自定义分类标准、关键词列表或 taxonomy.categories。")
        if source_type == "inline" and not papers:
            questions.append("请提供单篇 paper 对象或 papers 列表。")

        return ClassificationIntent(
            mode=mode,
            source_type=source_type,
            source_path=source_path,
            papers=papers,
            taxonomy=taxonomy,
            user_instruction=user_text or str(payload.get("user_instruction") or payload.get("userInstruction") or "").strip(),
            output_format=output_format,
            max_workers=max_workers,
            needs_clarification=bool(questions),
            clarification_questions=questions,
        )

    def _resolve_mode(self, payload: dict[str, Any], taxonomy: Taxonomy | None, user_text: str):
        raw = str(payload.get("mode") or "").strip().lower()
        if raw in {"custom", "general"}:
            return raw  # type: ignore[return-value]
        text = user_text.lower()
        custom_hints = ["按", "关键词", "给定", "自定义", "分类标准", "taxonomy", "keyword", "keywords", "categories"]
        return "custom" if taxonomy or any(hint in text for hint in custom_hints) else "general"

    def _resolve_taxonomy(self, payload: dict[str, Any], user_text: str) -> Taxonomy | None:
        taxonomy_raw = payload.get("taxonomy")
        if isinstance(taxonomy_raw, dict):
            return Taxonomy.from_dict(taxonomy_raw)

        categories_raw = payload.get("categories")
        if isinstance(categories_raw, list) and categories_raw:
            return Taxonomy.from_dict({"categories": categories_raw, "rules": payload.get("rules") or {}})

        keywords = self._extract_keywords(payload, user_text)
        if keywords:
            categories = [
                TaxonomyCategory(
                    id=_safe_id(keyword),
                    name=keyword,
                    definition=f"文献主题或内容与关键词 {keyword} 明确相关。",
                    keywords=[keyword],
                )
                for keyword in keywords
            ]
            return Taxonomy(categories=categories, rules=TaxonomyRules(allow_multiple_secondary_categories=True))
        return None

    def _extract_keywords(self, payload: dict[str, Any], user_text: str) -> list[str]:
        raw = payload.get("keywords") or payload.get("classification_keywords") or payload.get("classificationKeywords")
        if isinstance(raw, list):
            return _dedupe([str(item).strip() for item in raw if str(item).strip()])
        if isinstance(raw, str):
            return _split_keywords(raw)

        match = re.search(r"(?:关键词|keywords?)[:：]\s*(?P<value>.+)", user_text, flags=re.IGNORECASE)
        if match:
            return _split_keywords(match.group("value"))
        return []

    def _resolve_source(self, payload: dict[str, Any], user_text: str) -> tuple[SourceType, str | None, list[LiteraturePaper]]:
        papers_raw = payload.get("papers")
        if isinstance(papers_raw, list):
            return "inline", None, [LiteraturePaper.from_dict(item) for item in papers_raw if isinstance(item, dict)]

        paper_raw = payload.get("paper")
        if isinstance(paper_raw, dict):
            return "inline", None, [LiteraturePaper.from_dict(paper_raw)]

        source_path = str(payload.get("source_path") or payload.get("sourcePath") or payload.get("path") or "").strip()
        if not source_path and user_text:
            match = _PATH_RE.search(user_text)
            source_path = match.group("path") if match else ""
        if source_path:
            path = Path(source_path).expanduser()
            source_type: SourceType = "directory" if path.is_dir() or not path.suffix else "file"
            return source_type, source_path, []
        return "inline", None, []

    def _resolve_output_format(self, payload: dict[str, Any]) -> OutputFormat:
        raw = str(payload.get("output_format") or payload.get("outputFormat") or "json").strip().lower()
        return raw if raw in {"json", "jsonl", "csv"} else "json"  # type: ignore[return-value]

    def _resolve_max_workers(self, value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 4
        return min(max(parsed, 1), 32)


def _split_keywords(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    parts = re.split(r"[,，;；、\n]+", text)
    return _dedupe([part.strip().strip("。.!?") for part in parts if part.strip()])


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return safe or "category"


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
