# SimpleEmbeddingMock

A tiny **mock** of the OpenAI embeddings HTTP API. Pure Python standard library — no `pip install`, no `numpy`, no third-party dependencies. Runs as a single `python3 server.py`.

> **This is a mock, not a real embedding model.** It is meant for local development, integration tests, CI, offline demos, and anywhere you need an OpenAI-shaped embeddings endpoint without paying for tokens, hitting the network, or downloading a multi-GB model. The vectors it returns are computed by simple, deterministic, well-known hashing algorithms — they are *not* learned semantic embeddings. Don't use them for production retrieval.

## What it does

Exposes two OpenAI-compatible HTTP endpoints (with and without the `/v1` prefix):

- `GET  /v1/models` &nbsp;/&nbsp; `GET  /models` — list available mock models
- `POST /v1/embeddings` &nbsp;/&nbsp; `POST /embeddings` — compute embeddings

Request and response shapes match the OpenAI API closely enough that the official `openai` Python SDK can talk to it by setting `base_url=http://localhost:8080/v1`.

Supported request fields:
- `model` (required) — one of the mock model ids below
- `input` (required) — a string or a list of strings
- `encoding_format` — `"float"` (default) or `"base64"` (little-endian float32)

Errors come back in the OpenAI envelope: `{"error": {"message": ..., "type": ..., "code": ...}}` with appropriate HTTP status codes.

## Mock models

All three are real, well-known algorithms — implemented from scratch in stdlib. All produce L2-normalized vectors so cosine similarity reduces to a dot product. All hashing uses `hashlib.blake2b` so output is **deterministic across processes and machines** (unlike Python's randomized `hash()`).

| Model id | Dim | Algorithm |
|---|---|---|
| `hash-bow-256` | 256 | Word-level hashing trick (Weinberger et al. 2009 — same idea as scikit-learn's `HashingVectorizer`) |
| `hash-ngram-512` | 512 | FastText-flavored character n-gram hashing (n ∈ {3,4,5}) — captures sub-word overlap, so `running` and `run` get non-zero similarity |
| `random-proj-128` | 128 | Random projection (Johnson–Lindenstrauss) of bag-of-words via a deterministic ±1 Rademacher matrix computed on demand |

Because there is no training, no corpus, and no learned semantics, similarities reflect surface form (shared tokens, shared n-grams) — not meaning. That's intentional: the goal is a fast, predictable, dependency-free stand-in for real embedding services.

## Running

Requires Python 3.x — nothing else.

```bash
python3 server.py                  # listens on 127.0.0.1:8080
python3 server.py --port 9000      # custom port
PORT=9000 python3 server.py        # or via env var
python3 server.py --host 0.0.0.0   # expose on all interfaces
```

## Examples

List models:
```bash
curl -s http://localhost:8080/v1/models
```

Single embedding:
```bash
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"hash-bow-256","input":"hello world"}'
```

Batch embedding:
```bash
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"hash-ngram-512","input":["hello","world","hello world"]}'
```

Base64 encoding (little-endian float32):
```bash
curl -s -X POST http://localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model":"hash-bow-256","input":"hi","encoding_format":"base64"}'
```

Using the official OpenAI Python SDK against the mock:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-used")
resp = client.embeddings.create(model="hash-ngram-512", input="hello world")
print(len(resp.data[0].embedding))  # 512
```

## Project layout

- `server.py` — HTTP server, routing, JSON parse/serialize, error envelopes
- `embeddings.py` — algorithms registry (`MODELS`) and shared helpers

## Limitations (read before using)

- **Not real embeddings.** Vectors are derived from token/n-gram hashing, not from a trained model. Use real services or models (OpenAI, sentence-transformers, etc.) for anything that depends on semantic similarity.
- **Mock-grade server.** Single-process `ThreadingHTTPServer`, no auth, no rate limiting, no TLS — bind to `127.0.0.1` (the default).
- **English-leaning tokenizer.** Tokenization is `[a-z0-9]+` lowercased; non-ASCII text is effectively dropped at the word level (the n-gram model still partially handles it via byte-level encoding inside Python strings, but don't rely on it).
- **No streaming, no async, no tool-call shape** — only the embeddings surface is mocked.
