from __future__ import annotations

DEFAULT_PAPER_TYPES: dict[str, list[str]] = {
    "综述": ["survey", "review", "systematic review", "meta-analysis", "综述", "系统综述", "元分析"],
    "理论研究": ["theory", "theoretical", "framework", "conceptual", "模型框架", "理论", "机制解释"],
    "实证研究": ["empirical", "experiment", "survey data", "questionnaire", "interview", "regression", "实证", "实验", "问卷", "访谈"],
    "方法论文": ["method", "algorithm", "approach", "model", "benchmark", "算法", "方法", "模型", "基准"],
    "系统设计": ["system", "architecture", "prototype", "platform", "workflow", "系统", "架构", "平台", "原型"],
    "案例研究": ["case study", "case", "案例", "个案"],
}

DEFAULT_RESEARCH_METHODS: dict[str, list[str]] = {
    "实验": ["experiment", "experimental", "ablation", "controlled trial", "实验", "消融"],
    "问卷": ["questionnaire", "survey data", "问卷", "量表"],
    "访谈": ["interview", "访谈", "半结构化"],
    "统计建模": ["regression", "statistical", "bayesian", "causal", "统计", "回归", "贝叶斯", "因果"],
    "仿真": ["simulation", "simulate", "仿真", "模拟"],
    "深度学习": ["deep learning", "neural", "transformer", "cnn", "gnn", "深度学习", "神经网络", "大模型"],
    "定性分析": ["qualitative", "thematic analysis", "grounded theory", "定性", "主题分析"],
}

DEFAULT_DOMAINS: dict[str, list[str]] = {
    "计算机科学": ["computer science", "machine learning", "nlp", "vision", "algorithm", "软件", "计算机", "机器学习", "自然语言处理"],
    "教育": ["education", "student", "learning", "teaching", "classroom", "教育", "学生", "学习"],
    "医学": ["medicine", "clinical", "patient", "diagnosis", "healthcare", "医学", "临床", "患者", "诊断"],
    "管理学": ["management", "organization", "firm", "strategy", "管理", "组织", "企业"],
    "社会科学": ["social", "policy", "behavior", "社会", "政策", "行为"],
    "工程": ["engineering", "manufacturing", "robot", "control", "工程", "制造", "控制"],
    "经济金融": ["economics", "finance", "market", "risk", "经济", "金融", "市场", "风险"],
    "天文学": ["astronomy", "cosmology", "galaxy", "simulation", "telescope", "天文", "宇宙学", "星系", "模拟"],
}

DEFAULT_DATA_TYPES: dict[str, list[str]] = {
    "文本": ["text", "corpus", "document", "language", "文本", "语料"],
    "图像": ["image", "vision", "x-ray", "mri", "图像", "影像"],
    "表格数据": ["tabular", "record", "database", "表格", "结构化数据"],
    "传感器数据": ["sensor", "signal", "iot", "传感器", "信号"],
    "问卷数据": ["survey data", "questionnaire", "问卷"],
    "访谈文本": ["interview transcript", "interview", "访谈"],
    "仿真数据": ["simulation data", "simulated", "mock catalog", "仿真数据", "模拟数据"],
}

DEFAULT_APPLICATION_AREAS: dict[str, list[str]] = {
    "推荐系统": ["recommendation", "recommender", "推荐"],
    "自然语言处理": ["natural language processing", "nlp", "language model", "translation", "自然语言处理", "语言模型", "翻译"],
    "医疗诊断": ["diagnosis", "clinical decision", "disease", "医疗诊断", "疾病"],
    "教学分析": ["learning analytics", "student performance", "教学分析", "学习分析"],
    "风险预测": ["risk prediction", "forecasting risk", "风险预测"],
    "天文模拟": ["cosmological simulation", "n-body", "galaxy formation", "halo", "天文模拟", "宇宙学模拟", "暗物质晕"],
}
