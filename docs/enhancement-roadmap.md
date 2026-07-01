# Agent Enhancement Roadmap

## 1. LLM Robustness

- Add retry with exponential backoff for transient HTTP failures.
- Add per-paper timeout and cancellation at the batch runner level.
- Add JSON repair for minor model formatting issues before failing a paper.
- Record raw model output behind an opt-in debug flag for audit.

## 2. Input Coverage

- Add PDF ingestion as a separate loader stage, not inside the classifier.
- Add DOI/arXiv URL resolution into normalized paper objects.
- Add recursive directory manifests so large batches can be resumed.
- Add output checkpointing to avoid rerunning completed papers.

## 3. Classification Quality

- Add a held-out golden set and report accuracy / macro F1 for known taxonomies.
- Add ambiguity handling for close categories with explicit review reasons.
- Add category descriptions and negative examples to prompt construction.
- Add a strict validator that rejects custom-mode categories outside taxonomy.

## 4. Operations

- Add structured logging with paper id, latency, model, retries, and status.
- Add rate limiting for provider quotas.
- Add cost estimation before large batch runs.
- Add a dry-run mode that only resolves intent, loads papers, and renders prompts.

## 5. Agent Interface

- Expose a small HTTP API for upstream agents.
- Add a streaming progress callback for UI or orchestration systems.
- Add machine-readable clarification responses when the input is incomplete.
- Add result exporters for JSONL, CSV, and review spreadsheets.
