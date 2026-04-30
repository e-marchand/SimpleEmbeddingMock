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
| `perf-zero-8` | 8 | **Perf** — returns a fixed 8-dim zero vector. Tiniest possible payload, for measuring pure HTTP/JSON round-trip overhead with zero compute. |
| `perf-zero-1536` | 1536 | **Perf** — fixed zero vector matching OpenAI `text-embedding-3-small` dim. Zero compute, zero values (so the JSON is mostly `"0.0"` separators). |
| `perf-fixed-3072` | 3072 | **Perf** — fixed *non-zero* constant vector matching OpenAI `text-embedding-3-large` dim. Each float serializes to ~20 chars, so response size is realistic — use this to benchmark bandwidth/serialization, not just round-trip. |

Because there is no training, no corpus, and no learned semantics, similarities reflect surface form (shared tokens, shared n-grams) — not meaning. That's intentional: the goal is a fast, predictable, dependency-free stand-in for real embedding services.

### Perf-test models (zero compute on the server side)

The three `perf-*` models exist purely so you can isolate **client / network / serialization cost** from server-side embedding cost. They all return a pre-allocated constant vector — no tokenization, no hashing, no math. Pick the dim that matches what you want to measure:

- **`perf-zero-8`** → measures HTTP + JSON parsing round-trip with effectively zero payload. If your client is slow against this, the bottleneck is in the client or the network, not the server or the model.
- **`perf-zero-1536`** → realistic OpenAI-small dim, but each float is `"0.0"` so JSON stays light. Good middle-ground.
- **`perf-fixed-3072`** → realistic OpenAI-large dim *and* realistic float string length. Best model to benchmark throughput when the response payload size matters (batching, streaming clients, base64 vs float comparisons, etc.). For an apples-to-apples bandwidth test, also try `encoding_format: "base64"` — it cuts the response by ~6×.

Quick example: throughput against a 200-request synchronous client on the same host (numbers from a verification run, not a benchmark guarantee):

| Model | Response size | req/s |
|---|---|---|
| `perf-zero-8` | 194 B | ~2030 |
| `perf-zero-1536` | 7.8 KB | ~2220 |
| `perf-fixed-3072` | 67.7 KB | ~520 |
| `hash-ngram-512` (real compute) | 12 KB | ~2210 |

The drop on `perf-fixed-3072` is purely serialization + transfer — your real-world client will see the same shape against any 3072-dim embedding service.

## Running

### Plain Python

Requires Python 3.9+ — nothing else.

```bash
python3 server.py                  # listens on 127.0.0.1:8080
python3 server.py --port 9000      # custom port
PORT=9000 python3 server.py        # or via env var
python3 server.py --host 0.0.0.0   # expose on all interfaces
python3 server.py --debug          # verbose request/response logging
EMBEDMOCK_DEBUG=1 python3 server.py  # same, via env var
```

### Via [uvx](https://docs.astral.sh/uv/) (no clone needed)

The project ships a `pyproject.toml` and registers a `simple-embedding-mock` console script, so [`uvx`](https://docs.astral.sh/uv/guides/tools/) can run it in an isolated, ephemeral virtualenv with zero install steps:

```bash
# from a local checkout
uvx --from . simple-embedding-mock --port 8080

# directly from a Git URL (no clone required)
uvx --from git+https://github.com/e-marchand/SimpleEmbeddingMock simple-embedding-mock

# once published to PyPI
uvx simple-embedding-mock
```

To pull in optional plugins in the same one-liner, use the [extras](https://peps.python.org/pep-0508/#extras) declared in `pyproject.toml`:

```bash
# real semantic embeddings via sentence-transformers
uvx --from '.[sentence-transformers]' simple-embedding-mock

# sklearn HashingVectorizer
uvx --from '.[sklearn]' simple-embedding-mock

# everything
uvx --from '.[all]' simple-embedding-mock --debug
```

Available extras: `sentence-transformers`, `sklearn`, `all`.

> If you don't have uv yet: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).

### Debug logging

`--debug` (or `EMBEDMOCK_DEBUG=1`) prints, per request, a numbered block on stderr with the method/path/headers, the raw request body, and the response status, latency, and body preview (truncated at 2 KB):

```
[debug #3] >>> POST /v1/embeddings from 127.0.0.1
  Host: 127.0.0.1:8080
  content-type: application/json
  Content-Length: 46
[debug #3]     request body (46 bytes): {"model":"hash-bow-256","input":"hello debug"}
[debug #3] embedding model=hash-bow-256 inputs=1 encoding=float
[debug #3] <<< 200 (0.2 ms, 1467 bytes)
  body: {"object": "list", "data": [...], "usage": {...}}
```

Without the flag, only the standard one-line access log is emitted (`127.0.0.1 - - [date] "POST /v1/embeddings HTTP/1.1" 200 -`).

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

## Optional plugins (more models, only if the lib is installed)

The core stays stdlib-only, but if extra libraries are present on your machine, additional models light up automatically. Plugins live in `plugins/`, each one detects its dependency at import time and silently no-ops if the lib is missing. On startup the server logs which plugins were loaded vs skipped:

```
listening on 127.0.0.1:8080
models (3): hash-bow-256, hash-ngram-512, random-proj-128
plugin skipped: plugins.sentence_transformers_plugin (library not installed or no models)
plugin skipped: plugins.sklearn_plugin (library not installed or no models)
```

Built-in plugins:

| Plugin | Lib to install | Adds | Notes |
|---|---|---|---|
| `sentence_transformers_plugin` | `pip install sentence-transformers` | Real semantic embeddings, e.g. `all-MiniLM-L6-v2` (384 dims) | Models load lazily on first embed. Configure with `EMBEDMOCK_ST_MODELS=name1,name2`. Default: `all-MiniLM-L6-v2`. First call downloads weights (network required). |
| `sklearn_plugin` | `pip install scikit-learn` | `sklearn-hashing-1024` | Sklearn's own `HashingVectorizer` — useful as a parity reference for the core `hash-bow-256`. |

Once a plugin's lib is installed, restart the server — its models appear in `GET /v1/models` with their own `owned_by` field (`sentence-transformers`, `scikit-learn`, …) so clients can tell mock-grade from real.

### Adding your own plugin

Drop a new file `plugins/myplugin.py` with this shape:

```python
from importlib.util import find_spec

def register():
    if find_spec("mylib") is None:
        return {}
    import mylib

    def embed(text: str) -> list[float]:
        return mylib.embed(text)  # must return a list[float]

    return {
        "my-model-id": {"embed": embed, "owned_by": "mylib"},
    }
```

Then add `"plugins.myplugin"` to `_PLUGIN_MODULES` in `registry.py`. That's the entire contract — no class hierarchy, no config, no entrypoints.

## Project layout

- `server.py` — HTTP server, routing, JSON parse/serialize, error envelopes
- `embeddings.py` — core stdlib algorithms (always available)
- `registry.py` — merges core models with detected plugins into a single `MODELS` dict
- `plugins/` — optional plugins, each gated on an `import` check
- `pyproject.toml` — packaging metadata; registers the `simple-embedding-mock` console script and declares plugin extras for `uvx` / `pip install`

## Limitations (read before using)

- **Not real embeddings (core models).** The three built-in models are derived from token/n-gram hashing, not from a trained model. For real semantic similarity, install the `sentence-transformers` plugin or use a real embedding service.
- **Mock-grade server.** Single-process `ThreadingHTTPServer`, no auth, no rate limiting, no TLS — bind to `127.0.0.1` (the default).
- **English-leaning tokenizer.** Tokenization is `[a-z0-9]+` lowercased; non-ASCII text is effectively dropped at the word level (the n-gram model still partially handles it via byte-level encoding inside Python strings, but don't rely on it).
- **No streaming, no async, no tool-call shape** — only the embeddings surface is mocked.
