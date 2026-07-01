from __future__ import annotations

import json

from .schema import ClassificationIntent, LiteraturePaper


class PromptBuilder:
    def build(self, intent: ClassificationIntent, paper: LiteraturePaper) -> str:
        if intent.mode == "custom":
            return self._build_custom_prompt(intent, paper)
        return self._build_general_prompt(intent, paper)

    def _build_general_prompt(self, intent: ClassificationIntent, paper: LiteraturePaper) -> str:
        return "\n".join(
            [
                "你是文献分类 Agent。",
                "请根据标题、摘要、关键词和正文片段，对文献进行普通分类。",
                "输出严格 JSON，不要 Markdown。",
                self._schema_block(),
                "分类维度包括：paper_type, research_methods, domains, data_types, application_areas。",
                "必须给出 confidence、evidence、needs_human_review。",
                self._paper_block(paper),
                self._instruction_block(intent),
            ]
        ).strip()

    def _build_custom_prompt(self, intent: ClassificationIntent, paper: LiteraturePaper) -> str:
        taxonomy_payload = intent.taxonomy.to_dict() if intent.taxonomy else None
        return "\n".join(
            [
                "你是严格文献分类器。",
                "只能使用用户给定的分类标准或关键词集合。",
                "不得新增类别；如果没有类别匹配，primary_category=null，并设置 needs_human_review=true。",
                "每个分类必须给出 evidence，包括命中关键词或语义依据。",
                "输出严格 JSON，不要 Markdown。",
                self._schema_block(),
                f"taxonomy={json.dumps(taxonomy_payload, ensure_ascii=False)}",
                self._paper_block(paper),
                self._instruction_block(intent),
            ]
        ).strip()

    def _paper_block(self, paper: LiteraturePaper) -> str:
        payload = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "keywords": paper.keywords,
            "text": paper.text[:6000],
            "metadata": paper.metadata,
        }
        return f"paper={json.dumps(payload, ensure_ascii=False)}"

    def _instruction_block(self, intent: ClassificationIntent) -> str:
        return f"user_instruction={intent.user_instruction}" if intent.user_instruction else ""

    def _schema_block(self) -> str:
        return (
            "输出 JSON schema："
            '{"mode":"custom|general","paper_id":"...","primary_category":{"id":"...","name":"..."}或null,'
            '"secondary_categories":[{"id":"...","name":"..."}],"paper_type":"...或null",'
            '"research_methods":[],"domains":[],"application_areas":[],"data_types":[],'
            '"generated_keywords":[],"confidence":0到1,"evidence":[{"text":"原文证据","reason":"分类理由"}],'
            '"needs_human_review":true或false,"review_reasons":[]}'
        )
