"""OpenAI-compatible embedding mock server — pure stdlib."""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, List, Tuple

from embeddings import MODELS, MODEL_DIMS, count_tokens

CREATED_TS = 1700000000
OWNED_BY = "simple-embedding-mock"


def _models_payload() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": CREATED_TS, "owned_by": OWNED_BY}
            for name in MODELS
        ],
    }


def _err(message: str, type_: str, code: str | None = None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def _to_base64_f32(vec: List[float]) -> str:
    buf = struct.pack(f"<{len(vec)}f", *vec)
    return base64.b64encode(buf).decode("ascii")


def _normalize_path(path: str) -> str:
    # Strip query string and optional /v1 prefix.
    p = path.split("?", 1)[0]
    if p.startswith("/v1/"):
        p = p[3:]  # leaves "/models" / "/embeddings"
    elif p == "/v1":
        p = "/"
    return p


class Handler(BaseHTTPRequestHandler):
    server_version = "SimpleEmbeddingMock/1.0"

    # Keep the default access log but route it through stderr cleanly.
    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    # --- response helpers ---------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, type_: str, code: str | None = None) -> None:
        self._send_json(status, _err(message, type_, code))

    # --- routing ------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = _normalize_path(self.path)
        if path == "/models":
            self._send_json(HTTPStatus.OK, _models_payload())
        elif path == "/":
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": OWNED_BY})
        elif path == "/embeddings":
            self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Use POST for /embeddings", "invalid_request_error")
        else:
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {self.path}", "invalid_request_error")

    def do_POST(self) -> None:  # noqa: N802
        path = _normalize_path(self.path)
        if path != "/embeddings":
            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {self.path}", "invalid_request_error")
            return

        body, err = self._read_json_body()
        if err is not None:
            self._send_error(HTTPStatus.BAD_REQUEST, err, "invalid_request_error")
            return

        model = body.get("model")
        if not isinstance(model, str) or not model:
            self._send_error(HTTPStatus.BAD_REQUEST, "Missing required field: 'model'", "invalid_request_error")
            return
        if model not in MODELS:
            self._send_error(
                HTTPStatus.NOT_FOUND,
                f"Model '{model}' not found. Available: {sorted(MODELS)}",
                "invalid_request_error",
                code="model_not_found",
            )
            return

        raw_input = body.get("input")
        inputs, err = _coerce_inputs(raw_input)
        if err is not None:
            self._send_error(HTTPStatus.BAD_REQUEST, err, "invalid_request_error")
            return

        encoding_format = body.get("encoding_format", "float")
        if encoding_format not in ("float", "base64"):
            self._send_error(
                HTTPStatus.BAD_REQUEST,
                "encoding_format must be 'float' or 'base64'",
                "invalid_request_error",
            )
            return

        embed = MODELS[model]
        data = []
        total_tokens = 0
        for i, text in enumerate(inputs):
            vec = embed(text)
            total_tokens += count_tokens(text)
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": _to_base64_f32(vec) if encoding_format == "base64" else vec,
                }
            )

        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": data,
                "model": model,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            },
        )

    # --- body parsing -------------------------------------------------------

    def _read_json_body(self) -> Tuple[dict, str | None]:
        length_hdr = self.headers.get("Content-Length")
        if length_hdr is None:
            return {}, "Missing Content-Length header"
        try:
            length = int(length_hdr)
        except ValueError:
            return {}, "Invalid Content-Length"
        if length < 0 or length > 10 * 1024 * 1024:
            return {}, "Request body too large"
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}, "Empty request body"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {}, f"Invalid JSON body: {exc}"
        if not isinstance(parsed, dict):
            return {}, "Request body must be a JSON object"
        return parsed, None


def _coerce_inputs(raw: Any) -> Tuple[List[str], str | None]:
    if raw is None:
        return [], "Missing required field: 'input'"
    if isinstance(raw, str):
        if raw == "":
            return [], "'input' must not be empty"
        return [raw], None
    if isinstance(raw, list):
        if not raw:
            return [], "'input' list must not be empty"
        out: List[str] = []
        for i, item in enumerate(raw):
            if not isinstance(item, str):
                return [], f"'input[{i}]' must be a string"
            out.append(item)
        return out, None
    return [], "'input' must be a string or list of strings"


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple OpenAI-compatible embedding mock server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
        help="Port to listen on (default: 8080 or $PORT)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    sys.stderr.write(f"listening on {args.host}:{args.port}\n")
    sys.stderr.write(f"models: {', '.join(MODELS)}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("shutting down\n")
        server.server_close()


if __name__ == "__main__":
    main()
