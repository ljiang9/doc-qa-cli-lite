import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docqa import (  # noqa: E402
    SAMPLE_DOC,
    TfidfRetriever,
    chunk_text,
    tokenize,
)


class TestChunkText(unittest.TestCase):
    def test_basic(self):
        text = "一二三四五六七八九十" * 5  # 50 字
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        self.assertTrue(len(chunks) > 1)
        # 首尾块长度不超过 chunk_size
        for c in chunks:
            self.assertLessEqual(len(c), 20)

    def test_empty(self):
        self.assertEqual(chunk_text(""), [])

    def test_overlap_validation(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=0)

    def test_short_text_single_chunk(self):
        chunks = chunk_text("短文本", chunk_size=100)
        self.assertEqual(len(chunks), 1)


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_text(SAMPLE_DOC, chunk_size=60, overlap=10)
        self.ret = TfidfRetriever(self.chunks)

    def test_search_returns_ordered(self):
        hits = self.ret.search("余弦相似度取值范围", k=3)
        self.assertLessEqual(len(hits), 3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_relevant_chunk_found(self):
        hits = self.ret.search("切分文档 重叠 上下文", k=2)
        joined = " ".join(self.chunks[i] for i, _ in hits)
        self.assertIn("重叠", joined)

    def test_empty_query(self):
        self.assertEqual(self.ret.search("", k=3), [])


class TestTokenize(unittest.TestCase):
    def test_mixed(self):
        self.assertIn("tf", tokenize("TF-IDF"))


if __name__ == "__main__":
    unittest.main()
