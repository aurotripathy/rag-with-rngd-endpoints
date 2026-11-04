"""RAG pipeline against the Furiosa RNGD OpenAI-compatible endpoint.

Embed with Qwen3-Embedding-8B, keep the nearest passages, rerank with
Qwen3-Reranker-8B, then answer with a chat model. Same stages as
simple-rag-demo-with-rngd-models, with an in-memory index in place of pgvector.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
CACHE_PATH = ROOT / "data" / "embeddings.json"

# Passages from the RNGD RAG demo notebook.
CORPUS = [
    "PostgreSQL is an open-source relational database.",
    "RAG combines retrieval and generation for better answers.",
    "pgvector adds vector search capabilities to Postgres.",
    "PostgreSQL scales well vertically (up to ~10-100 TB) by increasing CPU/RAM, but it hits bottlenecks at massive scale.",
    "The scalability of PostgreSQL with pgvector is generally effective for datasets up to tens of millions of vectors.",
    "quote - I tested pgvector on 5 million 512-dimensional vectors, and the performance was good enough with 1 sec similarity search.",
    "pgvector supports exact nearest-neighbor search as well as approximate indexes (IVFFlat and HNSW) that trade a little recall for much faster queries.",
    "RAG reduces hallucinations by grounding the model's answer in retrieved passages, so the quality of the retrieval step bounds the quality of the final answer.",
    "pgvector compares vectors using L2 distance, inner product, or cosine distance, and the index must be built with the same operator class the query uses.",
]

RETRIEVE_K = 4


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    embed_model: str
    rerank_model: str
    chat_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        api_key = os.environ.get("FURIOSA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "FURIOSA_API_KEY is missing. Copy .env.example to .env and set the key."
            )
        return cls(
            api_key=api_key,
            base_url=os.environ.get(
                "FURIOSA_BASE_URL", "https://endpoint.access.furiosa.dev/v1"
            ).rstrip("/"),
            embed_model=os.environ.get("EMBED_MODEL", "furiosa-ai/Qwen3-Embedding-8B"),
            rerank_model=os.environ.get("RERANK_MODEL", "furiosa-ai/Qwen3-Reranker-8B"),
            chat_model=os.environ.get("CHAT_MODEL", "furiosa-ai/gpt-oss-120b"),
        )


@dataclass
class Hit:
    text: str
    score: float


@dataclass
class Answer:
    question: str
    retrieved: list[Hit]
    reranked: list[Hit]
    answer: str
    chat_model: str
    embed_model: str
    rerank_model: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class Pipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=120,
        )
        self._docs: list[tuple[str, list[float]]] | None = None

    def ensure_index(self) -> int:
        if self._docs is None:
            self._docs = self._load_or_embed()
        return len(self._docs)

    def _load_or_embed(self) -> list[tuple[str, list[float]]]:
        cached = self._read_cache()
        if cached is not None:
            return cached
        response = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=CORPUS,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        docs = [(text, list(item.embedding)) for text, item in zip(CORPUS, ordered)]
        self._write_cache(docs)
        return docs

    def _read_cache(self) -> list[tuple[str, list[float]]] | None:
        if not CACHE_PATH.exists():
            return None
        payload = json.loads(CACHE_PATH.read_text())
        if payload.get("model") != self.settings.embed_model:
            return None
        if payload.get("texts") != CORPUS:
            return None
        vectors = payload.get("vectors") or []
        if len(vectors) != len(CORPUS):
            return None
        return list(zip(CORPUS, vectors))

    def _write_cache(self, docs: list[tuple[str, list[float]]]) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(
                {
                    "model": self.settings.embed_model,
                    "texts": [text for text, _ in docs],
                    "vectors": [vector for _, vector in docs],
                }
            )
        )

    def _embed_query(self, question: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=[question],
        )
        return list(response.data[0].embedding)

    def _retrieve(self, question: str) -> list[Hit]:
        self.ensure_index()
        assert self._docs is not None
        query = self._embed_query(question)
        scored = [Hit(text, _cosine(query, vector)) for text, vector in self._docs]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:RETRIEVE_K]

    def _rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        payload = {
            "model": self.settings.rerank_model,
            "query": question,
            "documents": [hit.text for hit in hits],
            "top_n": len(hits),
        }
        request = urllib.request.Request(
            f"{self.settings.base_url}/rerank",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Rerank failed ({exc.code}): {detail}") from exc
        ranked: list[Hit] = []
        for item in body.get("results") or []:
            document = item.get("document")
            if isinstance(document, dict):
                text = document.get("text") or ""
            else:
                text = str(document or "")
            ranked.append(Hit(text, float(item.get("relevance_score") or 0.0)))
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked

    def _generate(self, question: str, context_hits: list[Hit]) -> str:
        context = "\n".join(hit.text for hit in context_hits)
        prompt = (
            "Answer using only the context. Prefer a short paraphrase. "
            "Never mention operators, functions, indexes, or steps unless those exact words "
            "appear in the context. If the context is not relevant, say \"I don't know\".\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n"
        )
        kwargs: dict = {
            "model": self.settings.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
        }
        if "Qwen3" in self.settings.chat_model and "VL" not in self.settings.chat_model:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return (message.content or "").strip()

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty.")
        retrieved = self._retrieve(question)
        reranked = self._rerank(question, retrieved)
        answer = self._generate(question, reranked)
        return Answer(
            question=question,
            retrieved=retrieved,
            reranked=reranked,
            answer=answer,
            chat_model=self.settings.chat_model,
            embed_model=self.settings.embed_model,
            rerank_model=self.settings.rerank_model,
        )


def main() -> None:
    import sys

    question = " ".join(sys.argv[1:]).strip() or "How do I search vectors in Postgres?"
    result = Pipeline().ask(question)
    print(f"question: {result.question}\n")
    print("retrieved:")
    for hit in result.retrieved:
        print(f"  {hit.score:.4f}  {hit.text}")
    print("\nreranked:")
    for hit in result.reranked:
        print(f"  {hit.score:.4f}  {hit.text}")
    print(f"\nanswer:\n{result.answer}")


if __name__ == "__main__":
    main()
