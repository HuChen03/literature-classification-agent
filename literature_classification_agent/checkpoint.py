from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .schema import ClassificationResult, PaperClassificationJobResult


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def load_successes(self) -> dict[str, PaperClassificationJobResult]:
        if not self.path.exists():
            return {}
        output: dict[str, PaperClassificationJobResult] = {}
        for line in self.path.read_text(encoding="utf8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            item = self._item_from_dict(payload)
            if item.status == "success":
                output[self._key(item.paper_id, item.title)] = item
        return output

    def append(self, item: PaperClassificationJobResult, include_prompt: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf8") as handle:
                handle.write(json.dumps(item.to_dict(include_prompt=include_prompt), ensure_ascii=False) + "\n")

    def _item_from_dict(self, payload: dict) -> PaperClassificationJobResult:
        result_payload = payload.get("result")
        return PaperClassificationJobResult(
            paper_id=payload.get("paper_id"),
            title=str(payload.get("title") or ""),
            status="success" if payload.get("status") == "success" else "failed",
            result=ClassificationResult.from_dict(result_payload, fallback_mode="general") if isinstance(result_payload, dict) else None,
            error=payload.get("error"),
            prompt=payload.get("prompt"),
        )

    def _key(self, paper_id: str | None, title: str) -> str:
        return paper_id or title


def checkpoint_key(paper_id: str | None, title: str) -> str:
    return paper_id or title
