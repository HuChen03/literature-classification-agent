from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ) -> BatchClassificationResult:
        if not papers:
            return BatchClassificationResult(intent=intent, items=[], include_prompts=include_prompts)

        workers = min(max(intent.max_workers, 1), len(papers))
        items: list[PaperClassificationJobResult | None] = [None] * len(papers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_one, intent, paper, include_prompts): index
                for index, paper in enumerate(papers)
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
