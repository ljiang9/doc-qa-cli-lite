#!/usr/bin/env python3
"""cli.py — doc-qa-cli-lite 命令行入口。

用法：
    python3 cli.py "你的问题" [--k 3] [--size 120] [--overlap 20] [--file doc.txt]

无 API key：只输出检索到的 top-k 片段。
有 OPENAI_API_KEY（可选 OPENAI_BASE_URL / OPENAI_MODEL）：额外生成答案。
"""
from __future__ import annotations

import argparse
import os
import sys

from docqa import (
    SAMPLE_DOC,
    TfidfRetriever,
    answer_with_llm,
    chunk_text,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="文档检索问答（切分 + top-k 检索）")
    p.add_argument("question", nargs="?", help="要问的问题")
    p.add_argument("--k", type=int, default=3, help="召回 top-k 片段，默认 3")
    p.add_argument("--size", type=int, default=120, help="每块字符大小，默认 120")
    p.add_argument("--overlap", type=int, default=20, help="块重叠字符数，默认 20")
    p.add_argument("--file", help="从文本文件读入文档，否则用内置样例")
    args = p.parse_args(argv)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            doc = f.read()
    else:
        doc = SAMPLE_DOC

    chunks = chunk_text(doc, chunk_size=args.size, overlap=args.overlap)
    print(f"[切分] 共 {len(chunks)} 个片段（size={args.size}, overlap={args.overlap}）")

    if not args.question:
        print("未提供问题。示例：python3 cli.py \"余弦相似度取值范围是多少\"")
        return 0

    retriever = TfidfRetriever(chunks)
    hits = retriever.search(args.question, k=args.k)
    contexts = [chunks[i] for i, _ in hits]

    print(f"\n问题：{args.question}")
    print(f"召回 top-{len(hits)} 片段：")
    for rank, (idx, score) in enumerate(hits, 1):
        print(f"  {rank}. [chunk#{idx} score={score:.4f}] {chunks[idx]}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print("\n[LLM] 检测到 OPENAI_API_KEY，正在生成答案...")
        try:
            ans = answer_with_llm(
                args.question, contexts, api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            )
            print(f"答案：{ans}")
        except Exception as e:  # 网络/密钥问题不阻断本地检索
            print(f"[LLM] 调用失败（已降级为仅展示片段）：{e}")
    else:
        print("\n[提示] 未设置 OPENAI_API_KEY，已跳过答案生成（以上为检索片段）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
