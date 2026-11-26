"""End-to-end tests for github.com/xulang-org/packages/{anthropic,rag}."""
import http.server
import json
import os
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
import unittest

PACKAGES = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("XU_ROOT", PACKAGES.parent / "xu")).expanduser().resolve()
XU = Path(os.environ.get("XU", str(REPO / "bin/xu"))).resolve()


def xu_env(work):
    env = dict(
        os.environ,
        XUPATH=str(PACKAGES),
        XU_CACHE=str(work / "cache"),
        XU_ROOT=str(REPO),
        XU_STDLIB_ROOT=str(REPO / "src"),
    )
    for key in ("XUOS", "XUARCH", "XU_HEAP_STATS"):
        env.pop(key, None)
    return env


def build_xu(work, name, src_text):
    proj = work / name
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "main.xu").write_text(src_text)
    (proj / "xu.xun").write_text(
        f'module: {name}\nversion: !ver 0.1.0\nrequire:\n  "github.com/xulang-org/packages": file:{PACKAGES}\n'
    )
    exe = proj / name
    res = subprocess.run(
        [str(XU), "build", str(proj), "-o", str(exe)],
        cwd=REPO,
        env=xu_env(work),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise AssertionError(res.stdout + res.stderr)
    if os.uname().sysname == "Darwin":
        subprocess.run(["codesign", "--force", "--sign", "-", str(exe)], check=True, capture_output=True)
    return exe


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


RETRY_LOCK = threading.Lock()
RETRY_COUNTS = {}


def fake_embed(text):
    t = text.lower()
    v0 = 1.0 if ("matrix" in t or "gpu" in t or "metal" in t) else 0.1
    v1 = 1.0 if ("moe" in t or "expert" in t or "router" in t) else 0.1
    v2 = 1.0 if ("mcp" in t or "stdio" in t or "tool" in t) else 0.1
    v3 = 1.0 if ("rag" in t or "vector" in t or "cosine" in t) else 0.1
    return [v0, v1, v2, v3]


class MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(ln).decode())
        if self.path == "/v1/embeddings":
            if self.headers.get("Authorization") != "Bearer rag-key":
                self._send_json(401, {"error": {"message": "bad auth"}})
                return
            inputs = req.get("input", [])
            data = [{"object": "embedding", "index": i, "embedding": fake_embed(s)} for i, s in enumerate(inputs)]
            self._send_json(200, {
                "object": "list",
                "model": req["model"],
                "data": data,
                "usage": {"prompt_tokens": len(inputs) * 2, "completion_tokens": 0, "total_tokens": len(inputs) * 2},
            })
            return

        if self.path == "/v1/messages":
            if self.headers.get("x-api-key") != "ant-secret":
                self._send_json(401, {"error": {"type": "authentication_error", "message": "invalid x-api-key"}})
                return
            assert self.headers.get("anthropic-version") == "2023-06-01", dict(self.headers)

            rtag = self.headers.get("X-Retry-Test")
            if rtag:
                with RETRY_LOCK:
                    cnt = RETRY_COUNTS.get(rtag, 0) + 1
                    RETRY_COUNTS[rtag] = cnt
                if cnt == 1:
                    self._send_json(529, {"error": {"type": "overloaded_error", "message": "overloaded"}})
                    return

            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"type": "message_start", "message": {"id": "msg_s1", "model": req["model"], "usage": {"input_tokens": 11, "cache_read_input_tokens": 3}}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan step 1. "}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello "}},
                    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Claude!"}},
                    {"type": "content_block_stop", "index": 1},
                    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}},
                    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"q\":"}},
                    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "\"xu\"}"}},
                    {"type": "content_block_stop", "index": 2},
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
                    {"type": "message_stop"},
                ]
                for ev in events:
                    self.wfile.write(("event: " + ev["type"] + "\ndata: " + json.dumps(ev) + "\n\n").encode())
                    self.wfile.flush()
                return

            msgs = req["messages"]
            if msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list) and msgs[-1]["content"][0].get("type") == "tool_result":
                tr = msgs[-1]["content"][0]
                assert tr["tool_use_id"] == "toolu_sync_1" and tr["content"] == "42"
                self._send_json(200, {
                    "id": "msg_sync_2",
                    "model": req["model"],
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "Result confirmed: 42"}],
                    "usage": {"input_tokens": 15, "output_tokens": 6},
                })
                return

            self._send_json(200, {
                "id": "msg_sync_1",
                "model": req["model"],
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "thinking", "thinking": "Need calc"},
                    {"type": "text", "text": "Calling calc"},
                    {"type": "tool_use", "id": "toolu_sync_1", "name": "calc", "input": {"x": 6, "y": 7}},
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            })
            return
        self._send_json(404, {"error": {"message": "not found"}})


class TestAnthropicAndRAG(unittest.TestCase):
    def test_anthropic_and_rag_end_to_end(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}"
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                store_path = work / "rag_store.json"
                src = f"""@main
#errors
#math
#strings
anthropic #"github.com/xulang-org/packages/anthropic"
openai #"github.com/xulang-org/packages/openai"
rag #"github.com/xulang-org/packages/rag"

fn on_ev(ev anthropic.StreamEvent) bool {{
    if ev.event_type == "text_delta" || ev.event_type == "thinking_delta" {{
        << true
    }}
    << true
}}

fn run() int {{
    // 1. Anthropic sync + 529 retry + tool_use + tool_result turn
    $ac := anthropic.new_client_with_base("{url}", "ant-secret")
    ac.set_max_retries(2)
    ac.add_header("X-Retry-Test", "ant-529")
    $m1 := [anthropic.message("user", "Multiply 6*7")]
    $r1 := anthropic.messages_request("claude-3-7-sonnet", ~m1, 512)
    r1.system = "Be concise"
    r1.thinking_budget_tokens = 256
    r1.tools.push(anthropic.tool("calc", "calc", "{{\\"type\\":\\"object\\"}}"))
    resp1 := ac.messages(&r1) >>>
    if resp1.id != "msg_sync_1" || resp1.stop_reason != "tool_use" || resp1.thinking != "Need calc" {{ << 1 }}
    if resp1.tool_uses.len != 1 || resp1.tool_uses[0].name != "calc" {{ << 2 }}

    $m2 := [anthropic.message("user", "Multiply 6*7"), anthropic.tool_result_message(resp1.tool_uses[0].id, "42", false)]
    $ac2 := anthropic.new_client_with_base("{url}", "ant-secret")
    r2 := anthropic.messages_request("claude-3-7-sonnet", ~m2, 256)
    resp2 := ac2.messages(&r2) >>>
    if resp2.content != "Result confirmed: 42" || resp2.usage.output_tokens != 6 {{ << 3 }}

    // 2. Anthropic SSE stream with thinking_delta + chunked tool_use input_json_delta
    $ch := chan anthropic.StreamEvent{{cap: 16}}
    r3 := anthropic.messages_request("claude-3-7-sonnet", [anthropic.message("user", "Hi")], 256)
    sresp := ac2.messages_stream_chan(&r3, ch) >>>
    if sresp.content != "Hello Claude!" || sresp.thinking != "Plan step 1. " {{ << 4 }}
    if sresp.tool_uses.len != 1 || sresp.tool_uses[0].input_json != "{{\\"q\\":\\"xu\\"}}" {{ << 5 }}
    if sresp.usage.input_tokens != 11 || sresp.usage.output_tokens != 9 || sresp.usage.cache_read_input_tokens != 3 {{ << 6 }}
    _ := ac2.messages_stream(&r3, on_ev) >>>

    // 3. RAG chunk_text + #tensor cosine search + filter + save_json/load_json + OpenAI index/retrieve
    parts := rag.chunk_text("abcdefghij", 4, 1)
    if parts.len != 3 || parts[0] != "abcd" || parts[1] != "defg" || parts[2] != "ghij" {{ << 7 }}

    $oc := openai.new_client("{url}", "rag-key")
    $store := rag.new_store(4)
    _ := rag.index_document($&store, &oc, "emb-1", "doc-gpu", "metal.md", "Metal GPU matrix kernel speedup", 128, 16, "{{\\"topic\\":\\"gpu\\"}}") >>>
    _ := rag.index_document($&store, &oc, "emb-1", "doc-moe", "moe.md", "MoE expert router topk dispatch", 128, 16, "{{\\"topic\\":\\"moe\\"}}") >>>
    _ := rag.index_document($&store, &oc, "emb-1", "doc-rag", "rag.md", "Vector cosine retrieval via RAG", 128, 16, "{{\\"topic\\":\\"rag\\"}}") >>>
    if store.len() != 3 {{ << 8 }}

    hits := rag.retrieve(&store, &oc, "emb-1", "MoE router expert selection", 2, "") >>>
    if hits.len != 2 || hits[0].chunk.doc_id != "doc-moe" || hits[0].score < 0.9 {{ << 9 }}
    fhits := rag.retrieve(&store, &oc, "emb-1", "MoE router expert selection", 2, "doc-gpu") >>>
    if fhits.len != 1 || fhits[0].chunk.doc_id != "doc-gpu" {{ << 10 }}

    _ := store.save_json("{store_path}") >>>
    reloaded := rag.load_json("{store_path}") >>>
    if reloaded.len() != 3 || reloaded.dim != 4 {{ << 11 }}
    rhits := reloaded.search(&[0.1, 0.1, 0.1, 1.0], 1, "") >>>
    if rhits.len != 1 || rhits[0].chunk.doc_id != "doc-rag" {{ << 12 }}
    ctx := rag.format_context(&rhits)
    if !strings.contains(&ctx, "rag.md") || !strings.contains(&ctx, "Vector cosine") {{ << 13 }}

    println("ANTHROPIC_AND_RAG_OK")
    << 0
}}

fn main() int {{
    << run() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
                exe = build_xu(work, "ant_rag_test", src)
                res = subprocess.run([str(exe)], env=xu_env(work), text=True, capture_output=True, timeout=30)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertIn("ANTHROPIC_AND_RAG_OK", res.stdout)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
