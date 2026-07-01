from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schema import ClassificationIntent, LiteraturePaper


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md"}


class PaperLoader:
    def load(self, intent: ClassificationIntent) -> list[LiteraturePaper]:
        if intent.source_type == "inline":
            return list(intent.papers)
        if not intent.source_path:
            raise ValueError("source_path is required")
        path = Path(intent.source_path).expanduser()
        if intent.source_type == "directory":
            return self._load_directory(path)
        return self._load_file(path)

    def _load_directory(self, path: Path) -> list[LiteraturePaper]:
        if not path.exists() or not path.is_dir():
            raise ValueError(f"directory not found: {path}")
        papers: list[LiteraturePaper] = []
        for file_path in sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES):
            papers.extend(self._load_file(file_path))
        if not papers:
            raise ValueError(f"no supported paper files found under: {path}")
        return papers

    def _load_file(self, path: Path) -> list[LiteraturePaper]:
        if not path.exists() or not path.is_file():
            raise ValueError(f"file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._load_json(path)
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        if suffix == ".csv":
            return self._load_csv(path)
        if suffix in {".txt", ".md"}:
            return [self._load_text(path)]
        raise ValueError(f"unsupported file type: {path.suffix}")

    def _load_json(self, path: Path) -> list[LiteraturePaper]:
        payload = json.loads(path.read_text(encoding="utf8"))
        if isinstance(payload, list):
            return [LiteraturePaper.from_dict(item) for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            raise ValueError(f"JSON file must contain object or array: {path}")
        if isinstance(payload.get("papers"), list):
            return [LiteraturePaper.from_dict(item) for item in payload["papers"] if isinstance(item, dict)]
        if isinstance(payload.get("paper"), dict):
            return [LiteraturePaper.from_dict(payload["paper"])]
        return [LiteraturePaper.from_dict(payload)]

    def _load_jsonl(self, path: Path) -> list[LiteraturePaper]:
        papers: list[LiteraturePaper] = []
        for line_no, line in enumerate(path.read_text(encoding="utf8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_no} must be an object: {path}")
            papers.append(LiteraturePaper.from_dict(payload.get("paper") if isinstance(payload.get("paper"), dict) else payload))
        return papers

    def _load_csv(self, path: Path) -> list[LiteraturePaper]:
        with path.open("r", encoding="utf8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        papers: list[LiteraturePaper] = []
        for index, row in enumerate(rows, start=1):
            payload: dict[str, Any] = {
                "paper_id": row.get("paper_id") or row.get("paperId") or row.get("id") or f"{path.stem}-{index}",
                "title": row.get("title") or "",
                "abstract": row.get("abstract") or "",
                "text": row.get("text") or row.get("content") or "",
                "keywords": _split_keywords(row.get("keywords") or ""),
                "metadata": {"source_path": str(path), "row": index},
            }
            papers.append(LiteraturePaper.from_dict(payload))
        return papers

    def _load_text(self, path: Path) -> LiteraturePaper:
        content = path.read_text(encoding="utf8")
        title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and len(lines[0]) <= 180:
            title = lines[0].lstrip("#").strip() or title
        return LiteraturePaper(
            paper_id=path.stem,
            title=title,
            text=content,
            metadata={"source_path": str(path)},
        )


def _split_keywords(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.replace("，", ",").replace("；", ",").replace(";", ",").split(",") if item.strip()]
