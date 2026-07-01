# Literature Classification Agent

独立的单 Agent 文献分类实现，不依赖现有 `SciHarnessSystem` 代码。

## 能力

- 先识别意图：判断普通分类还是给定关键词/自定义 taxonomy 分类。
- 自动解析论文来源：单篇 `paper`、`papers` 列表、文件、目录。
- `custom` 模式：用户提供 taxonomy，Agent 严格只按照该分类标准分类。
- `general` 模式：用户不提供 taxonomy 时，使用内置普通文献分类维度。
- 批量分类：目录或多篇文件会走线程池并发执行。
- 分模式构造 prompt：普通分类 prompt 与自定义严格分类 prompt 分开。
- 默认使用 LLM 分类，输出结构化 JSON：分类结果、置信度、证据、人工审核标记。
- 保留 `CLASSIFIER_BACKEND=rules` 离线规则后端，便于测试和无 API key 场景。

## 输入格式

支持两种输入：

1. 自然语言输入：面向用户直接输入，Agent 会从文本中识别分类模式、关键词和论文路径。
2. JSON 输入：面向上游 Agent 或程序调用，字段稳定，适合自动化批处理。

自然语言示例：

```text
请按给定关键词分类 examples/papers.jsonl。关键词: cosmological simulation, natural language processing
```

JSON 示例：

```json
{
  "mode": "custom",
  "paper": {
    "paper_id": "paper_001",
    "title": "Example title",
    "abstract": "Example abstract",
    "keywords": ["keyword"]
  },
  "taxonomy": {
    "categories": [
      {
        "id": "empirical",
        "name": "实证研究",
        "definition": "基于数据、实验、问卷、访谈等进行验证的文献",
        "keywords": ["experiment", "survey", "问卷", "实验"]
      }
    ],
    "rules": {
      "single_primary_category": true,
      "allow_multiple_secondary_categories": false,
      "allow_unknown": false,
      "min_confidence_for_auto_accept": 0.7
    }
  }
}
```

不传 `taxonomy` 时自动进入 `general` 模式。

也可以传一个任务式 JSON 请求，让 Agent 自己识别路径和关键词：

```json
{
  "request": "请按给定关键词分类 examples/papers.jsonl。关键词: 天文模拟, 自然语言处理",
  "source_path": "examples/papers.jsonl",
  "keywords": ["天文模拟", "自然语言处理"],
  "max_workers": 4
}
```

## 运行

默认后端是 LLM。先复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
CLASSIFIER_BACKEND=llm
OPENAI_API_KEY=你的 API key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_TIMEOUT_S=60
```

然后运行：

```bash
cd literature_classification_agent
python3 -m literature_classification_agent.cli examples/custom_input.json --pretty
python3 -m literature_classification_agent.cli examples/general_input.json --pretty
python3 -m literature_classification_agent.cli examples/batch_keyword_request.json --pretty
python3 -m literature_classification_agent.cli examples/natural_language_request.txt --pretty
```

没有 LLM API key 时，可以临时使用规则后端：

```bash
CLASSIFIER_BACKEND=rules python3 -m literature_classification_agent.cli examples/batch_keyword_request.json --pretty
```

也可以从 stdin 读取：

```bash
python3 -m literature_classification_agent.cli - --pretty < examples/general_input.json
```

旧的单篇直接分类输出可使用：

```bash
python3 -m literature_classification_agent.cli examples/general_input.json --single --pretty
```

## 测试

```bash
cd literature_classification_agent
python3 -m unittest discover -s tests
```
