from __future__ import annotations

from .batch import BatchRunner
from .loader import PaperLoader
from .llm import LlmClassifier
from .prompt_builder import PromptBuilder
from .router import IntentRouter
from .schema import BatchClassificationResult, ClassificationInput, ClassificationIntent, ClassificationResult


class LiteratureClassificationAgent:
    """Standalone literature classification agent backed by an LLM."""

    def __init__(self, classifier: LlmClassifier | None = None, prompt_builder: PromptBuilder | None = None) -> None:
        self._classifier = classifier or LlmClassifier()
        self._prompt_builder = prompt_builder or PromptBuilder()

    def classify(self, payload: ClassificationInput | dict, prompt: str | None = None) -> ClassificationResult:
        request = ClassificationInput.from_dict(payload) if isinstance(payload, dict) else payload
        final_prompt = prompt or self._prompt_builder.build(
            ClassificationIntent(
                mode=request.mode or ("custom" if request.taxonomy else "general"),
                source_type="inline",
                papers=[request.paper],
                taxonomy=request.taxonomy,
                user_instruction=request.user_instruction,
            ),
            request.paper,
        )
        return self._classifier.classify(request, prompt=final_prompt)

    def run(
        self,
        payload: ClassificationIntent | dict | str,
        include_prompts: bool = False,
        checkpoint_path: str | None = None,
        resume: bool = False,
    ) -> BatchClassificationResult:
        intent = payload if isinstance(payload, ClassificationIntent) else IntentRouter().route(payload)
        if intent.needs_clarification:
            return BatchClassificationResult(intent=intent, items=[], include_prompts=include_prompts)
        papers = PaperLoader().load(intent)
        return BatchRunner(self._classifier, prompt_builder=self._prompt_builder).run(
            intent,
            papers,
            include_prompts=include_prompts,
            checkpoint_path=checkpoint_path,
            resume=resume,
        )
