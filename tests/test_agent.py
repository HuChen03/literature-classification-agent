import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from literature_classification_agent import IntentRouter, LiteratureClassificationAgent, LlmClassifier, OpenAICompatibleClient, PaperLoader, PromptBuilder
from literature_classification_agent.cli import _parse_input
from literature_classification_agent.llm import parse_json_object
from literature_classification_agent.schema import ClassificationInput, LiteraturePaper, Taxonomy, TaxonomyCategory


class FakeClient:
    def complete_json(self, prompt):
        lower = prompt.lower()
        if "taxonomy=" in lower and ("cosmological simulation" in lower or "halo" in lower):
            return {
                "mode": "custom",
                "paper_id": "a1",
                "primary_category": {"id": "cosmological-simulation", "name": "cosmological simulation"},
                "secondary_categories": [],
                "confidence": 0.91,
                "evidence": [{"text": "Cosmological Simulation of Dark Matter Halos", "reason": "semantic match"}],
                "needs_human_review": False,
                "review_reasons": [],
            }
        if "taxonomy=" in lower and ("natural language processing" in lower or "nlp" in lower):
            return {
                "mode": "custom",
                "paper_id": "n1",
                "primary_category": {"id": "natural-language-processing", "name": "natural language processing"},
                "secondary_categories": [],
                "confidence": 0.9,
                "evidence": [{"text": "Natural Language Processing", "reason": "semantic match"}],
                "needs_human_review": False,
                "review_reasons": [],
            }
        return {
            "mode": "general",
            "paper_type": "综述",
            "research_methods": ["深度学习"],
            "domains": ["计算机科学"],
            "application_areas": ["自然语言处理"],
            "data_types": ["文本"],
            "generated_keywords": ["transformer", "survey"],
            "confidence": 0.88,
            "evidence": [{"text": "Transformer Survey", "reason": "general classification"}],
            "needs_human_review": False,
            "review_reasons": [],
        }


def build_agent() -> LiteratureClassificationAgent:
    return LiteratureClassificationAgent(classifier=LlmClassifier(client=FakeClient()))


class LiteratureClassificationAgentTest(unittest.TestCase):
    def test_custom_mode_uses_llm_and_only_user_taxonomy(self):
        result = build_agent().classify(
            {
                "mode": "custom",
                "paper": {
                    "paper_id": "a1",
                    "title": "Cosmological Simulation of Dark Matter Halos",
                    "abstract": "This simulation studies halo formation in cosmology.",
                },
                "taxonomy": {
                    "categories": [
                        {"id": "cosmological-simulation", "name": "cosmological simulation"},
                        {"id": "natural-language-processing", "name": "natural language processing"},
                    ]
                },
            }
        ).to_dict()

        self.assertEqual(result["mode"], "custom")
        self.assertEqual(result["primary_category"]["id"], "cosmological-simulation")
        self.assertTrue(result["evidence"])

    def test_general_mode_uses_llm_dimensions(self):
        result = build_agent().classify(
            {
                "paper": {
                    "paper_id": "p2",
                    "title": "Transformer Models for Natural Language Processing: A Survey",
                    "abstract": "This survey reviews transformer architectures and language models.",
                    "keywords": ["transformer", "survey", "NLP"],
                }
            }
        ).to_dict()

        self.assertEqual(result["mode"], "general")
        self.assertEqual(result["paper_type"], "综述")
        self.assertIn("计算机科学", result["domains"])

    def test_router_detects_keyword_custom_mode_and_path(self):
        intent = IntentRouter().route(
            {
                "request": "请按给定关键词分类 ./papers.jsonl。关键词: 天文模拟, 自然语言处理",
                "keywords": ["天文模拟", "自然语言处理"],
            }
        )

        self.assertEqual(intent.mode, "custom")
        self.assertEqual(intent.source_type, "file")
        self.assertEqual(intent.source_path, "./papers.jsonl")
        self.assertIsNotNone(intent.taxonomy)
        self.assertEqual([item.name for item in intent.taxonomy.categories], ["天文模拟", "自然语言处理"])

    def test_router_accepts_natural_language_string(self):
        intent = IntentRouter().route("请按给定关键词分类 ./papers.jsonl。关键词: 天文模拟, 自然语言处理")

        self.assertEqual(intent.mode, "custom")
        self.assertEqual(intent.source_type, "file")
        self.assertEqual(intent.source_path, "./papers.jsonl")
        self.assertIsNotNone(intent.taxonomy)

    def test_loader_and_batch_runner_classify_jsonl_with_threads(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "papers.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"paper_id":"a1","title":"Cosmological Simulation of Dark Matter Halos","abstract":"This simulation studies halo formation in cosmology.","keywords":["halo"]}',
                        '{"paper_id":"n1","title":"Transformer Models for Natural Language Processing","abstract":"A language model benchmark for NLP tasks.","keywords":["NLP"]}',
                    ]
                ),
                encoding="utf8",
            )
            payload = {
                "source_path": str(path),
                "keywords": ["cosmological simulation", "natural language processing"],
                "max_workers": 2,
            }

            batch = build_agent().run(payload, include_prompts=True).to_dict()

            self.assertEqual(batch["summary"]["total"], 2)
            self.assertEqual(batch["summary"]["success_count"], 2)
            self.assertIn("prompt", batch["items"][0])
            self.assertEqual(batch["items"][0]["result"]["primary_category"]["name"], "cosmological simulation")

    def test_agent_run_accepts_natural_language_user_input(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "papers.jsonl"
            path.write_text(
                '{"paper_id":"a1","title":"Cosmological Simulation of Dark Matter Halos","abstract":"This simulation studies halo formation in cosmology."}',
                encoding="utf8",
            )

            batch = build_agent().run(f"请按给定关键词分类 {path}。关键词: cosmological simulation").to_dict()

            self.assertEqual(batch["intent"]["mode"], "custom")
            self.assertEqual(batch["summary"]["success_count"], 1)
            self.assertEqual(batch["items"][0]["result"]["primary_category"]["name"], "cosmological simulation")

    def test_prompt_builder_creates_different_prompts_by_mode(self):
        general_intent = IntentRouter().route(
            {
                "paper": {
                    "title": "Transformer Survey",
                    "abstract": "A survey of transformer language models.",
                }
            }
        )
        custom_intent = IntentRouter().route(
            {
                "paper": {
                    "title": "Transformer Survey",
                    "abstract": "A survey of transformer language models.",
                },
                "keywords": ["NLP"],
            }
        )
        paper = PaperLoader().load(general_intent)[0]

        general_prompt = PromptBuilder().build(general_intent, paper)
        custom_prompt = PromptBuilder().build(custom_intent, paper)

        self.assertIn("普通分类", general_prompt)
        self.assertIn("只能使用用户给定", custom_prompt)

    def test_cli_input_parser_supports_json_and_natural_language(self):
        self.assertIsInstance(_parse_input('{"paper":{"title":"A","abstract":"B"}}'), dict)
        self.assertEqual(_parse_input("请分类 examples/papers.jsonl"), "请分类 examples/papers.jsonl")

    def test_llm_classifier_validates_out_of_taxonomy_categories(self):
        class OutOfTaxonomyClient:
            def complete_json(self, prompt):
                return {
                    "mode": "custom",
                    "paper_id": "p1",
                    "primary_category": {"id": "allowed", "name": "Allowed"},
                    "secondary_categories": [{"id": "outside", "name": "Outside"}],
                    "confidence": 0.9,
                    "evidence": [{"text": "evidence sentence", "reason": "semantic match"}],
                    "needs_human_review": False,
                    "review_reasons": [],
                }

        classifier = LlmClassifier(client=OutOfTaxonomyClient())
        result = classifier.classify(
            ClassificationInput(
                paper=LiteraturePaper(title="A", paper_id="p1", abstract="evidence sentence"),
                mode="custom",
                taxonomy=Taxonomy(categories=[TaxonomyCategory(id="allowed", name="Allowed", keywords=["evidence"])]),
            ),
            prompt="classify",
        )

        self.assertEqual(result.primary_category.id, "allowed")
        self.assertEqual(result.secondary_categories, [])
        self.assertIn("llm_secondary_category_outside_taxonomy", result.review_reasons)

    def test_json_repair_accepts_fenced_json(self):
        parsed = parse_json_object(
            """
```json
{"mode":"general","confidence":0.8}
```
"""
        )

        self.assertEqual(parsed["mode"], "general")

    def test_llm_client_retries_retryable_failures(self):
        class RetryClient(OpenAICompatibleClient):
            def __init__(self):
                self.api_key = "test"
                self.base_url = "https://example.test/v1"
                self.model = "test"
                self.timeout_s = 1
                self.max_retries = 2
                self.retry_backoff_s = 0
                self.calls = 0

            def _post_once(self, body):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("llm_http_503: unavailable")
                return {"choices": [{"message": {"content": '{"mode":"general","confidence":0.8}'}}]}

        client = RetryClient()
        parsed = client.complete_json("prompt")

        self.assertEqual(parsed["mode"], "general")
        self.assertEqual(client.calls, 2)

    def test_checkpoint_resume_skips_successful_items(self):
        with TemporaryDirectory() as tmpdir:
            papers_path = Path(tmpdir) / "papers.jsonl"
            checkpoint_path = Path(tmpdir) / "checkpoint.jsonl"
            papers_path.write_text(
                '{"paper_id":"a1","title":"Cosmological Simulation of Dark Matter Halos","abstract":"This simulation studies halo formation in cosmology."}',
                encoding="utf8",
            )
            payload = {"source_path": str(papers_path), "keywords": ["cosmological simulation"]}

            first = build_agent().run(payload, checkpoint_path=str(checkpoint_path)).to_dict()
            self.assertEqual(first["summary"]["success_count"], 1)
            self.assertTrue(checkpoint_path.exists())

            class FailingClient:
                def complete_json(self, prompt):
                    raise RuntimeError("should_not_be_called")

            resumed = LiteratureClassificationAgent(classifier=LlmClassifier(client=FailingClient())).run(
                payload,
                checkpoint_path=str(checkpoint_path),
                resume=True,
            ).to_dict()

            self.assertEqual(resumed["summary"]["success_count"], 1)
            self.assertEqual(resumed["items"][0]["result"]["primary_category"]["name"], "cosmological simulation")


if __name__ == "__main__":
    unittest.main()
