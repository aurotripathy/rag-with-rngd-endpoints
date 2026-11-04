# RAG with RNGD endpoints

A small retrieval-augmented generation demo that runs entirely against the
Furiosa RNGD OpenAI-compatible endpoint:

1. **Embed** the corpus and the question with `Qwen3-Embedding-8B`.
2. **Retrieve** the top 4 passages by cosine similarity (in-memory index).
3. **Rerank** those passages with `Qwen3-Reranker-8B`.
4. **Generate** an answer with a chat model (`gpt-oss-120b` by default), using
   only the reranked passages as context.

The corpus is a handful of short statements about PostgreSQL, pgvector, and
RAG, defined in `CORPUS` in `rag.py`. Embeddings are cached in
`data/embeddings.json` and recomputed automatically if the corpus or the
embedding model changes.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and set your API key.

## `.env`

`rag.py` reads `.env` from the project root at startup. Variables already set
in the environment take precedence over the file. `.env` is gitignored.

| Variable           | Required | Default                                     | Purpose                                  |
| ------------------ | -------- | ------------------------------------------- | ---------------------------------------- |
| `FURIOSA_API_KEY`  | yes      | —                                           | Bearer token for the RNGD endpoint       |
| `FURIOSA_BASE_URL` | no       | `https://endpoint.access.furiosa.dev/v1`    | OpenAI-compatible base URL               |
| `EMBED_MODEL`      | no       | `furiosa-ai/Qwen3-Embedding-8B`             | Model for `/embeddings`                  |
| `RERANK_MODEL`     | no       | `furiosa-ai/Qwen3-Reranker-8B`              | Model for `/rerank`                      |
| `CHAT_MODEL`       | no       | `furiosa-ai/gpt-oss-120b`                   | Model for `/chat/completions`            |

The web UI additionally honors:

| Variable   | Default     | Purpose                 |
| ---------- | ----------- | ----------------------- |
| `RAG_HOST` | `127.0.0.1` | Bind address for `app.py` |
| `RAG_PORT` | `8765`      | Port for `app.py`         |

Example `.env`:

```dotenv
FURIOSA_API_KEY=sk-...
FURIOSA_BASE_URL=https://endpoint.access.furiosa.dev/v1
EMBED_MODEL=furiosa-ai/Qwen3-Embedding-8B
RERANK_MODEL=furiosa-ai/Qwen3-Reranker-8B
CHAT_MODEL=furiosa-ai/gpt-oss-120b
```

## Usage

### Command line

```bash
python3 rag.py "How do I search vectors in Postgres?"
```

With no argument it uses that same question. Output shows the retrieved
passages with cosine scores, the reranked passages with relevance scores, and
the final answer.

### Web UI

```bash
python3 app.py
# Open http://127.0.0.1:8765
```

Routes:

- `/` — form with example questions and the full corpus.
- `/ask?q=...` — runs the pipeline and renders answer, retrieved, and reranked
  passages.
- `/health` — JSON liveness check; also builds the embedding index if needed
  and reports the document count.

Change the bind address or port with `RAG_HOST` / `RAG_PORT`, e.g.
`RAG_PORT=9000 python3 app.py`.

### From Python

```python
from rag import Pipeline

result = Pipeline().ask("What is RAG?")
print(result.answer)
for hit in result.reranked:
    print(hit.score, hit.text)
```

## Project layout

```
rag.py             pipeline: settings, embedding cache, retrieve/rerank/generate
app.py             minimal stdlib HTTP server wrapping the pipeline
requirements.txt   openai>=1.40
.env.example       template for .env
data/              embeddings cache (gitignored, created on first run)
```
