# doc-qa-cli-lite

零第三方依赖的命令行文档问答：**文档切分 + top-k 检索**。无 LLM 时输出检索片段，有 key 时生成答案。

## 功能简介

- 把长文档按字符大小切分，并保留可配置的重叠（overlap）；
- 对每个片段建 TF-IDF 索引，查询时用余弦相似度召回 top-k；
- **没有 API key**：直接打印命中的片段，完成"从文档里找信息"的闭环；
- **有 `OPENAI_API_KEY`**：自动把 top-k 片段作为上下文，调用 OpenAI 兼容接口生成答案；调用失败会优雅降级回片段展示。

## 快速开始

```bash
# 用内置样例文档提问（无需任何 key）
python3 cli.py "余弦相似度取值范围是多少"

# 调整召回数量与切分参数
python3 cli.py "怎么做文档问答" --k 3 --size 80 --overlap 15

# 用自己的文档
python3 cli.py "你的问题" --file doc.txt
```

## 无 API key 如何运行

本项目核心是检索，**完全不需要 API key**。设不设置 `OPENAI_API_KEY` 都能跑：

- 不设置：只输出 top-k 检索片段；
- 设置 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`、`OPENAI_MODEL`）：额外生成答案。

## 目录说明

```
doc-qa-cli-lite/
├── docqa.py             # 核心库：chunk_text / TfidfRetriever / answer_with_llm
├── cli.py               # 命令行入口
├── tests/test_docqa.py  # unittest 测试
└── README.md
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT License，Copyright (c) 2026 ljiang9
