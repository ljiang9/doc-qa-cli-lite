"""docqa.py — 文档检索问答（无 LLM 也可用）。

流程：文档切分(chunking) -> 对 chunk 建 TF-IDF 索引 -> 检索 top-k 片段。
有 OPENAI_API_KEY 时，可把检索片段作为上下文交给 OpenAI 兼容接口生成答案；
没有 key 时直接打印命中的片段，核心链路完整可用。
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """按字符长度切分，带重叠。中文按字符计长，不破坏整词边界的硬要求。"""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")
    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


class TfidfRetriever:
    """对 chunk 列表建 TF-IDF 索引并做余弦 top-k。"""

    def __init__(self, chunks: Sequence[str]) -> None:
        self.chunks = list(chunks)
        self.tf: List[Counter] = []
        self.df: Counter = Counter()
        for c in self.chunks:
            counts = Counter(tokenize(c))
            self.tf.append(counts)
            for term in counts:
                self.df[term] += 1

    def _vec(self, counts: Counter) -> Dict[str, float]:
        total = len(self.chunks)
        if not counts:
            return {}
        mx = max(counts.values())
        out = {}
        for term, c in counts.items():
            idf = math.log((1 + total) / (1 + self.df.get(term, 0))) + 1.0
            out[term] = (c / mx) * idf
        return out

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        q = self._vec(Counter(tokenize(query)))
        if not q:
            return []
        scored = [(i, self._cos(q, self._vec(c))) for i, c in enumerate(self.tf)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def answer_with_llm(question: str, contexts: Sequence[str], api_key: str,
                    base_url: str = "https://api.openai.com/v1",
                    model: str = "gpt-3.5-turbo",
                    timeout: int = 30) -> str:
    """用 OpenAI 兼容接口生成答案。仅在传入 api_key 时调用。"""
    ctx = "\n\n".join(f"[片段{i+1}] {c}" for i, c in enumerate(contexts))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "仅根据给定的文档片段回答问题，不知道就说不知道。"},
            {"role": "user", "content": f"文档片段：\n{ctx}\n\n问题：{question}"},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                  "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


# 内置样例文档（无 key 即可跑）。
SAMPLE_DOC = (
    "余弦相似度衡量两个向量夹角的余弦值，对文本长度不敏感，取值在 -1 到 1 之间。"
    "TF-IDF 由词频和逆文档频率相乘构成，用来衡量一个词对文档的重要程度。"
    "向量检索把文本映射到高维空间，再通过相似度召回与查询最相关的片段。"
    "在做文档问答时，通常先把长文档切分成固定大小的片段，并保留一定重叠以保持上下文连贯。"
    "召回 top-k 片段后，再交给大模型生成最终答案；没有大模型时，直接展示片段也能定位信息。"
)
