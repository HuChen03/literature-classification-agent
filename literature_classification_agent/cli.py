from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .agent import LiteratureClassificationAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify literature with a standalone single agent.")
    parser.add_argument("input", help="Path to a JSON input file. Use '-' to read stdin.")
    parser.add_argument("--single", action="store_true", help="Use legacy single-paper classify mode.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--include-prompts", action="store_true", help="Include constructed prompts in output.")
    parser.add_argument("--output", help="Optional output file path. Defaults to stdout.")
    args = parser.parse_args(argv)

    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf8")
    payload = _parse_input(raw)
    agent = LiteratureClassificationAgent()
    if args.single:
        output = agent.classify(payload).to_dict()
        text = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)
    else:
        batch = agent.run(payload, include_prompts=args.include_prompts)
        output = batch.to_dict()
        output_format = output["intent"]["output_format"]
        if output_format == "jsonl":
            text = _to_jsonl(output, include_prompts=args.include_prompts)
        elif output_format == "csv":
            text = _to_csv(output)
        else:
            text = json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf8")
    else:
        print(text)
    return 0


def _to_jsonl(output: dict[str, Any], include_prompts: bool) -> str:
    rows = []
    for item in output["items"]:
        rows.append(json.dumps(item if include_prompts else {key: value for key, value in item.items() if key != "prompt"}, ensure_ascii=False))
    return "\n".join(rows)


def _parse_input(raw: str) -> dict[str, Any] | str:
    text = raw.strip()
    if not text:
        raise ValueError("input is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(parsed, dict):
        raise ValueError("JSON input must be an object")
    return parsed


def _to_csv(output: dict[str, Any]) -> str:
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "paper_id",
            "title",
            "status",
            "mode",
            "primary_category",
            "paper_type",
            "confidence",
            "needs_human_review",
            "error",
        ],
    )
    writer.writeheader()
    for item in output["items"]:
        result = item.get("result") or {}
        primary = result.get("primary_category") or {}
        writer.writerow(
            {
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "status": item.get("status"),
                "mode": result.get("mode"),
                "primary_category": primary.get("name") if isinstance(primary, dict) else "",
                "paper_type": result.get("paper_type"),
                "confidence": result.get("confidence"),
                "needs_human_review": result.get("needs_human_review"),
                "error": item.get("error"),
            }
        )
    return handle.getvalue().strip()


if __name__ == "__main__":
    raise SystemExit(main())
