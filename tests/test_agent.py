import unittest
import os

from pathlib import Path
from tempfile import TemporaryDirectory

from literature_classification_agent import IntentRouter, LiteratureClassificationAgent, LlmClassifier, PaperLoader, PromptBuilder
from literature_classification_agent.cli import _parse_input
from literature_classification_agent.schema import ClassificationInput, LiteraturePaper, Taxonomy, TaxonomyCategory


class LiteratureClassificationAgentTest(unittest.TestCase):
    def setUp(self):
        os.environ["CLASSIFIER_BACKEND"] = "rules"

    def test_custom_mode_uses_only_user_taxonomy(self):
        payload = {
            "mode": "custom",
            "paper": {
                "paper_id": "p1",
                "title": "Questionnaire-Based Study of Student Engagement",
                "abstract": "This empirical study uses questionnaire data and regression analysis to evaluate engagement factors.",
                "keywords": ["questionnaire", "regression"],
            },
            "taxonomy": {
                "categories": [
                    {
                        "id": "theory",
                        "name": "理论研究",
                        "definition": "提出理论模型或概念框架",
                        "keywords": ["theory", "framework", "理论"],
                    },
                    {
                        "id": "empirical",
                        "name": "实证研究",
                        "definition": "基于数据、实验、问卷、访谈或统计分析进行验证",
                        "keywords": ["empirical", "questionnaire", "regression", "data", "问卷"],
                    },
                ],
                "rules": {
                    "allow_unknown": False,
                    "min_confidence_for_auto_accept": 0.6,
                },
            },
        }

        result = LiteratureClassificationAgent().classify(payload).to_dict()

        self.assertEqual(result["mode"], "custom")
        self.assertEqual(result["primary_category"]["id"], "empirical")
        self.assertEqual(result["secondary_categories"], [])
        self.assertGreaterEqual(result["confidence"], 0.6)
        self.assertTrue(result["evidence"])

    def test_custom_mode_marks_review_when_no_allowed_category_matches(self):
        payload = {
            "mode": "custom",
            "paper": {
                "title": "A Clinical Imaging Dataset for Diagnosis",
                "abstract": "The article describes X-ray images and clinical diagnosis labels.",
            },
            "taxonomy": {
                "categories": [
                    {
                        "id": "theory",
                        "name": "理论研究",
                        "definition": "提出理论模型或概念框架",
                        "keywords": ["theory", "framework", "理论"],
                    }
                ],
                "rules": {"allow_unknown": False},
            },
        }

        result = LiteratureClassificationAgent().classify(payload).to_dict()

        self.assertIsNone(result["primary_category"])
        self.assertTrue(result["needs_human_review"])
        self.assertIn("no_allowed_category_matched", result["review_reasons"])

    def test_general_mode_outputs_default_dimensions(self):
        payload = {
            "paper": {
                "paper_id": "p2",
                "title": "Transformer Models for Natural Language Processing: A Survey",
                "abstract": "This survey reviews transformer architectures, language models, benchmarks, and applications in natural language processing.",
                "keywords": ["transformer", "language model", "survey", "NLP"],
            }
        }

        result = LiteratureClassificationAgent().classify(payload).to_dict()

        self.assertEqual(result["mode"], "general")
        self.assertEqual(result["paper_type"], "综述")
        self.assertIn("自然语言处理", result["application_areas"])
        self.assertIn("计算机科学", result["domains"])
        self.assertTrue(result["generated_keywords"])

    def test_router_detects_keyword_custom_mode_and_path(self):
        payload = {
            "request": "请按给定关键词分类 ./papers.jsonl。关键词: 天文模拟, 自然语言处理",
            "keywords": ["天文模拟", "自然语言处理"],
        }

        intent = IntentRouter().route(payload)

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

            batch = LiteratureClassificationAgent().run(payload, include_prompts=True).to_dict()

            self.assertEqual(batch["summary"]["total"], 2)
            self.assertEqual(batch["summary"]["success_count"], 2)
            self.assertEqual(batch["items"][0]["result"]["mode"], "custom")
            self.assertIn("prompt", batch["items"][0])
            self.assertEqual(batch["items"][0]["result"]["primary_category"]["name"], "cosmological simulation")

    def test_agent_run_accepts_natural_language_user_input(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "papers.jsonl"
            path.write_text(
                '{"paper_id":"a1","title":"Cosmological Simulation of Dark Matter Halos","abstract":"This simulation studies halo formation in cosmology."}',
                encoding="utf8",
            )

            batch = LiteratureClassificationAgent().run(f"请按给定关键词分类 {path}。关键词: cosmological simulation").to_dict()

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

    def test_llm_classifier_parses_and_validates_result(self):
        class FakeClient:
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

        classifier = LlmClassifier(client=FakeClient())
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


if __name__ == "__main__":
    unittest.main()
