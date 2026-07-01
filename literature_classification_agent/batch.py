from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .checkpoint import CheckpointStore, checkpoint_key
from .prompt_builder import PromptBuilder
from .schema import BatchClassificationResult, ClassificationInput, ClassificationIntent, LiteraturePaper, PaperClassificationJobResult


class BatchRunner:
    def __init__(self, classifier, prompt_builder: PromptBuilder | None = None) -> None:
        self._classifier = classifier
        self._prompt_builder = prompt_builder or PromptBuilder()

    def run(
        self,
        intent: ClassificationIntent,
        papers: list[LiteraturePaper],
        include_prompts: bool = False,
        checkpoint_path: str | None = None,
        resume: bool = False,
    ) -> BatchClassificationResult:
        if not papers:
            return BatchClassificationResult(intent=intent, items=[], include_prompts=include_prompts)

        checkpoint = CheckpointStore(checkpoint_path) if checkpoint_path else None
        resumed = checkpoint.load_successes() if checkpoint and resume else {}
        workers = min(max(intent.max_workers, 1), len(papers))
        items: list[PaperClassificationJobResult | None] = [None] * len(papers)
        pending: list[tuple[int, LiteraturePaper]] = []
        for index, paper in enumerate(papers):
            cached = resumed.get(checkpoint_key(paper.paper_id, paper.title))
            if cached:
                items[index] = cached
            else:
                pending.append((index, paper))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_one, intent, paper, include_prompts): index
                for index, paper in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    items[index] = future.result()
                except Exception as error:  # defensive; _run_one catches expected failures
                    paper = papers[index]
                    items[index] = PaperClassificationJobResult(
                        paper_id=paper.paper_id,
                        title=paper.title,
                        status="failed",
                        error=str(error),
                    )
                if checkpoint and items[index] is not None:
                    checkpoint.append(items[index], include_prompt=include_prompts)

        return BatchClassificationResult(
            intent=intent,
            items=[item for item in items if item is not None],
            include_prompts=include_prompts,
        )

    def _run_one(self, intent: ClassificationIntent, paper: LiteraturePaper, include_prompt: bool) -> PaperClassificationJobResult:
        prompt = self._prompt_builder.build(intent, paper)
        try:
            result = self._classifier.classify(
                ClassificationInput(
                    paper=paper,
                    mode=intent.mode,
                    taxonomy=intent.taxonomy,
                    user_instruction=intent.user_instruction,
                ),
                prompt=prompt,
            )
            return PaperClassificationJobResult(
                paper_id=paper.paper_id,
                title=paper.title,
                status="success",
                result=result,
                prompt=prompt if include_prompt else None,
            )
        except Exception as error:
            return PaperClassificationJobResult(
                paper_id=paper.paper_id,
                title=paper.title,
                status="failed",
                error=str(error),
                prompt=prompt if include_prompt else None,
            )
