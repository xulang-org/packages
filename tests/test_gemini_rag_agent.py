"""End-to-end tests for github.com/xulang-org/packages/{gemini,rag,agent} including live llm_engine /v1/messages + HTTP MCP."""
import http.server
import json
import os
from pathlib import Path
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
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
RETRY_SEEN = {}


class MockAIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
        return json.loads(raw)

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        req = self._read_json()

        # --- Google Gemini endpoints ---
        if self.path == "/v1beta/models/gemini-2.5-pro:generateContent":
            if self.headers.get("x-goog-api-key") != "gem-secret":
                self._send_json(401, {"error": {"message": "bad gemini key"}})
                return
            rkey = self.headers.get("X-Retry-Key", "")
            if rkey:
                with RETRY_LOCK:
                    c = RETRY_SEEN.get(rkey, 0) + 1
                    RETRY_SEEN[rkey] = c
                if c == 1:
                    self._send_json(429, {"error": {"message": "rate limited"}})
                    return
            # Verify multimodal + tools + schema were encoded
            parts = req["contents"][0]["parts"]
            has_img = any("inlineData" in p and p["inlineData"]["mimeType"] == "image/png" for p in parts)
            has_tool = bool(req.get("tools", [{}])[0].get("functionDeclarations"))
            has_sch = "responseSchema" in req.get("generationConfig", {})
            assert has_img and has_tool and has_sch, req
            self._send_json(200, {
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {"thought": True, "text": "pondering"},
                            {"text": "gem_ok"},
                            {"functionCall": {"name": "lookup", "args": {"id": 42}}},
                        ],
                    },
                }],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 6,
                    "totalTokenCount": 21,
                    "thoughtsTokenCount": 3,
                },
            })
            return

        if self.path == "/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"candidates": [{"content": {"parts": [{"thought": True, "text": "th1"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]},
                {
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Gemini!"}]}}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9, "thoughtsTokenCount": 1},
                },
            ]
            for ch in chunks:
                self.wfile.write(("data: " + json.dumps(ch) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            return

        if self.path == "/v1beta/models/text-embedding-004:embedContent":
            self._send_json(200, {"embedding": {"values": [0.25, 0.5, -0.75]}})
            return

        # --- OpenAI-compatible mock for streaming + parallel tools Agent test ---
        if self.path == "/v1/chat/completions":
            msgs = req.get("messages", [])
            has_tool_res = any(m.get("role") == "tool" for m in msgs)
            stream = bool(req.get("stream", False))
            if not stream:
                self._send_json(200, {
                    "id": "cmpl-1",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if not has_tool_res:
                ev1 = {"id": "s1", "choices": [{"index": 0, "delta": {"reasoning_content": "plan"}, "finish_reason": None}]}
                ev2 = {"id": "s1", "choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": 0, "id": "c0", "type": "function", "function": {"name": "tool_a", "arguments": "{\"v\":1}"}},
                    {"index": 1, "id": "c1", "type": "function", "function": {"name": "tool_b", "arguments": "{\"v\":2}"}},
                ]}, "finish_reason": None}]}
                ev3 = {"id": "s1", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}
                for ev in (ev1, ev2, ev3):
                    self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                    self.wfile.flush()
            else:
                ev1 = {"id": "s2", "choices": [{"index": 0, "delta": {"content": "FINAL:"}, "finish_reason": None}]}
                ev2 = {"id": "s2", "choices": [{"index": 0, "delta": {"content": "AB"}, "finish_reason": None}]}
                ev3 = {"id": "s2", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}}
                for ev in (ev1, ev2, ev3):
                    self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                    self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self.send_response(404)
        self.end_headers()


def pack_gguf_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_live_anthropic_mcp_gguf(path):
    dim = 64
    hidden = 128
    n_heads = 4
    n_kv = 4
    n_layers = 1
    vocab = 32
    max_seq = 128

    def f32(x):
        return struct.unpack("<f", struct.pack("<f", float(x)))[0]

    pieces = [f"tok_{i}" for i in range(vocab)]
    pieces[0] = "<|endoftext|>"
    pieces[1] = "<|im_start|>"
    pieces[2] = "<|im_end|>"
    pieces[3] = "user"
    pieces[4] = "assistant"
    pieces[5] = "system"
    pieces[6] = "<tool_response>"
    pieces[7] = "\n"
    pieces[8] = "</tool_response>"
    pieces[10] = "<think>"
    pieces[11] = "calc"
    pieces[12] = "</think>"
    pieces[13] = "<tool_call>"
    pieces[14] = "{\"name\":\"mul\",\"arguments\":{\"x\":6,\"y\":7}}"
    pieces[15] = "</tool_call>"
    pieces[16] = "42"
    pieces[17] = "RESULT_42"

    emb = [f32(0.0)] * (vocab * dim)
    head = [f32(0.0)] * (vocab * dim)
    for t in range(vocab):
        emb[t * dim] = f32(1.0)
    chain = [(7, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 2)]
    for col_idx, (src_t, dst_t) in enumerate(chain, start=1):
        emb[src_t * dim + col_idx] = f32(1.0)
        head[dst_t * dim + col_idx] = f32(8.0)

    # Attention signal on token 16 ("42"): dim 40 shifts hidden state on turn 2 -> RESULT_42 (tok 17) -> <|im_end|> (tok 2)
    emb[16 * dim + 40] = f32(1.0)
    head[17 * dim + 40] = f32(20.0)
    head[10 * dim + 40] = f32(-20.0)
    emb[17 * dim + 41] = f32(1.0)
    head[2 * dim + 41] = f32(80.0)
    head[17 * dim + 41] = f32(-50.0)

    wv = [f32(0.0)] * (dim * dim)
    wo = [f32(0.0)] * (dim * dim)
    wv[40 * dim + 40] = f32(1.0)
    wo[40 * dim + 40] = f32(30.0)

    kv_entries = [
        ("general.architecture", 8, pack_gguf_str("qwen2")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("qwen2.embedding_length", 4, struct.pack("<I", dim)),
        ("qwen2.feed_forward_length", 4, struct.pack("<I", hidden)),
        ("qwen2.block_count", 4, struct.pack("<I", n_layers)),
        ("qwen2.attention.head_count", 4, struct.pack("<I", n_heads)),
        ("qwen2.attention.head_count_kv", 4, struct.pack("<I", n_kv)),
        ("qwen2.context_length", 4, struct.pack("<I", max_seq)),
        ("qwen2.rope.freq_base", 6, struct.pack("<f", 10000.0)),
        ("qwen2.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-5)),
        ("tokenizer.ggml.model", 8, pack_gguf_str("llama")),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, vocab) + b"".join(pack_gguf_str(p) for p in pieces)),
    ]

    tensor_specs = [
        ("token_embd.weight", [vocab, dim], emb),
        ("blk.0.attn_norm.weight", [dim], [f32(1.0)] * dim),
        ("blk.0.attn_q.weight", [dim, dim], [f32(0.0)] * (dim * dim)),
        ("blk.0.attn_k.weight", [dim, dim], [f32(0.0)] * (dim * dim)),
        ("blk.0.attn_v.weight", [dim, dim], wv),
        ("blk.0.attn_output.weight", [dim, dim], wo),
        ("blk.0.ffn_norm.weight", [dim], [f32(1.0)] * dim),
        ("blk.0.ffn_gate.weight", [hidden, dim], [f32(0.0)] * (hidden * dim)),
        ("blk.0.ffn_up.weight", [hidden, dim], [f32(0.0)] * (hidden * dim)),
        ("blk.0.ffn_down.weight", [dim, hidden], [f32(0.0)] * (dim * hidden)),
        ("output_norm.weight", [dim], [f32(1.0)] * dim),
        ("output.weight", [vocab, dim], head),
    ]

    hdr = bytearray(b"GGUF")
    hdr.extend(struct.pack("<IQQ", 3, len(tensor_specs), len(kv_entries)))
    for k, vt, vb in kv_entries:
        hdr.extend(pack_gguf_str(k))
        hdr.extend(struct.pack("<I", vt))
        hdr.extend(vb)
    cur = 0
    payload = bytearray()
    for name, dims, vals in tensor_specs:
        cur = (cur + 31) & ~31
        while len(payload) < cur:
            payload.append(0)
        hdr.extend(pack_gguf_str(name))
        hdr.extend(struct.pack("<I", len(dims)))
        for d in reversed(dims):
            hdr.extend(struct.pack("<Q", d))
        raw = struct.pack(f"<{len(vals)}f", *vals)
        hdr.extend(struct.pack("<IQ", 0, cur))
        payload.extend(raw)
        cur += len(raw)
    while len(hdr) % 32 != 0:
        hdr.append(0)
    path.write_bytes(bytes(hdr) + bytes(payload))


class TestGeminiRagAgent(unittest.TestCase):
    def test_gemini_and_rag_and_agent_streaming_parallel(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockAIHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}"
            xu_src = f"""@main
#errors
#strings
#time
gemini #github.com/xulang-org/packages/gemini
rag #github.com/xulang-org/packages/rag
agent #github.com/xulang-org/packages/agent
openai #github.com/xulang-org/packages/openai

fn slow_a(args &string) string {{
    time.sleep(140 * time.millisecond())
    << "A:" + (*args)
}}

fn slow_b(args &string) string {{
    time.sleep(140 * time.millisecond())
    << "B:" + (*args)
}}

fn on_stream_chunk(sc gemini.StreamChunk) bool {{
    if sc.thought_delta.len > 0 || sc.text_delta.len > 0 || sc.done {{
        println("GEM_SC:" + sc.thought_delta + "|" + sc.text_delta + "|" + sc.finish_reason)
    }}
    << true
}}

fn on_agent_ev(ev agent.AgentEvent) bool {{
    println("AG_EV:" + ev.kind + ":" + str(ev.turn) + ":" + ev.text_delta + ":" + ev.reasoning_delta + ":" + ev.tool_name + ":" + ev.tool_output)
    << true
}}

fn run_all() int {{
    // 1. Gemini: retry 429, inline_data, tools, schema, thoughts, stream + embed
    $gc := gemini.new_client_with_base("{url}", "gem-secret")
    gc.set_max_retries(2)
    gc.add_header("X-Retry-Key", "g-r1")
    $parts := [gemini.text_part("describe"), gemini.inline_data_part("image/png", "AQID")]
    $greq := gemini.generate_request("gemini-2.5-pro", [gemini.content("user", ~parts)])
    greq.system_instruction = "sys"
    greq.tools = [gemini.function_decl("lookup", "lookup id", "{{\\"type\\":\\"object\\"}}")]
    greq.response_mime_type = "application/json"
    greq.response_schema_json = "{{\\"type\\":\\"object\\"}}"
    greq.thinking_budget = 128
    gresp := gc.generate(&greq) >>>
    if gresp.thoughts != "pondering" || gresp.text != "gem_ok" || gresp.function_calls.len != 1 || gresp.function_calls[0].name != "lookup" {{
        << 10
    }}
    if gresp.usage.thoughts_token_count != 3 || gresp.usage.total_token_count != 21 {{ << 11 }}

    $gsreq := gemini.generate_request("gemini-2.5-pro", [gemini.user_text("stream")])
    gs_cb := gc.generate_stream(&gsreq, on_stream_chunk) >>>
    if gs_cb.thoughts != "th1" || gs_cb.text != "Hello Gemini!" || gs_cb.finish_reason != "STOP" {{ << 12 }}

    ch := chan gemini.StreamChunk{{cap: 16}}
    gs_ch := gc.generate_stream_chan(&gsreq, ch) >>>
    if gs_ch.text != "Hello Gemini!" {{ << 13 }}

    gemb := gc.embed("text-embedding-004", "hello") >>>
    if gemb.len != 3 {{ << 14 }}

    // 2. RAG v1.7: chunk_markdown + BM25 + Hybrid RRF
    md := "# Guide\\nIntro text.\\n## GPU Metal\\n### MoE Kernels\\nUse moe_gemv_q4_0_f32 for fast top-k expert execution.\\n## Vector Search\\nCosine topk works great."
    mchunks := rag.chunk_markdown("g1", "guide.md", md, 80, 10)
    if mchunks.len < 3 {{ << 20 }}
    if !strings.has_prefix(&mchunks[1].text, "[Guide > GPU Metal > MoE Kernels]") {{ << 21 }}

    $store := rag.new_store(4)
    _ := store.add_chunk("cA", "d1", "metal fused moe_gemv_q4_0_f32 kernel", "a.md", "{{}}", &[0.1, 0.1, 0.1, 0.1]) >>>
    _ := store.add_chunk("cB", "d1", "generic hardware accelerator note", "b.md", "{{}}", &[0.9, 0.9, 0.1, 0.1]) >>>
    _ := store.add_chunk("cC", "d1", "xu metal gpu moe_gemv_q4_0_f32 YaRN guide", "c.md", "{{}}", &[0.95, 0.95, 0.1, 0.1]) >>>

    bm := store.search_bm25("moe_gemv_q4_0_f32", 3)
    if bm.len != 2 || bm[0].chunk.id == "cB" {{ << 22 }}

    hy := store.search_hybrid("moe_gemv_q4_0_f32", &[1.0, 1.0, 0.0, 0.0], 2, 60)
    if hy.len != 2 || hy[0].chunk.id != "cC" {{ << 23 }}
    ctx := store.retrieve_hybrid("moe_gemv_q4_0_f32", &[1.0, 1.0, 0.0, 0.0], 2, 60)
    if ctx.len == 0 {{ << 24 }}

    // 3. Agent v1.7: run_stream + Session.send_stream + parallel_tools
    oc := openai.new_client("{url}", "ok")
    $ag := agent.new_agent(oc, "xu-7b")
    ag.set_parallel_tools(true)
    ag.add_tool("tool_a", "a", "{{\\"type\\":\\"object\\"}}", slow_a)
    ag.add_tool("tool_b", "b", "{{\\"type\\":\\"object\\"}}", slow_b)

    t0 := time.now()
    $sess := agent.new_session(ag)
    res := sess.send_stream("go", on_agent_ev) >>>
    elapsed_ms := time.since(t0) / time.millisecond()
    if res.content != "FINAL:AB" || res.reasoning_content != "plan" {{ << 30 }}
    if res.steps.len != 2 || res.steps[0].tool_name != "tool_a" || res.steps[1].tool_name != "tool_b" {{ << 31 }}
    println("PARALLEL_MS:" + str(elapsed_ms))
    if elapsed_ms >= 260 {{ << 32 }}
    << 0
}}

fn main() int {{
    << run_all() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
            with tempfile.TemporaryDirectory() as td:
                exe = build_xu(Path(td), "drv_gem_rag_agent", xu_src)
                res = subprocess.run([str(exe)], env=xu_env(Path(td)), capture_output=True, text=True, timeout=30)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                lines = res.stdout.splitlines()
                self.assertTrue(any(l == "AG_EV:reasoning_delta:1::plan::" for l in lines), lines)
                self.assertTrue(any(l.startswith("AG_EV:tool_start:1:::tool_a:") for l in lines), lines)
                self.assertTrue(any(l.startswith("AG_EV:tool_end:1:::tool_b:B:") for l in lines), lines)
                self.assertTrue(any(l == "AG_EV:text_delta:2:FINAL::::" for l in lines), lines)
                self.assertTrue(any(l.startswith("AG_EV:done:2:FINAL:AB:") for l in lines), lines)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_live_anthropic_agent_mcp_llm_engine(self):
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            env = xu_env(work)
            llm_exe = work / "llm_engine"
            res = subprocess.run(
                [str(XU), "build", str(REPO / "examples/llm_engine/main.xu"), "-o", str(llm_exe)],
                cwd=REPO, env=env, capture_output=True, text=True,
            )
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            if os.uname().sysname == "Darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(llm_exe)], check=True, capture_output=True)

            gguf_path = work / "live_anth_mcp.gguf"
            write_live_anthropic_mcp_gguf(gguf_path)

            srv_proc = subprocess.Popen(
                [str(llm_exe), "serve", str(gguf_path), "127.0.0.1:0", "--api-key", "live-key"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                base_url = None
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    line = srv_proc.stdout.readline()
                    if line and line.strip().startswith("serving:http://"):
                        base_url = line.strip().split("serving:", 1)[1]
                        break
                self.assertIsNotNone(base_url, "llm_engine server failed to start")

                mcp_src = """@main
#errors
#net/http
mcp #github.com/xulang-org/packages/mcp
#encoding/json

fn handle_mul(args &string) mcp.ToolResult {
    d := json.parse(args)
    $x := 0
    $y := 0
    if d.ok() && d.is_dict(d.root()) {
        xn := d.get(d.root(), "x") >>> (_) { << mcp.error_result("bad x") }
        yn := d.get(d.root(), "y") >>> (_) { << mcp.error_result("bad y") }
        x = d.as_int(xn) >>> (_) { << mcp.error_result("bad x int") }
        y = d.as_int(yn) >>> (_) { << mcp.error_result("bad y int") }
    }
    << mcp.text_result(str(x * y))
}

fn run_mcp() int {
    $msrv := mcp.new_server("calc-mcp", "1.0.0")
    msrv.add_tool("mul", "multiply x and y", "{\\"type\\":\\"object\\"}", handle_mul)
    $hsrv := http.new_server()
    _ := hsrv.listen("127.0.0.1:0") >>>
    println("mcp:http://" + hsrv.addr() + "/mcp")
    _ := msrv.serve_http_server($&hsrv) >>>
    _ := hsrv.close()
    << 0
}

fn main() int {
    << run_mcp() >>> (_) { << 1 }
}
"""
                mcp_exe = build_xu(work, "mcp_srv", mcp_src)
                mcp_proc = subprocess.Popen(
                    [str(mcp_exe)], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                try:
                    mcp_url = None
                    deadline2 = time.monotonic() + 10.0
                    while time.monotonic() < deadline2:
                        mline = mcp_proc.stdout.readline()
                        if mline and mline.strip().startswith("mcp:http://"):
                            mcp_url = mline.strip().split("mcp:", 1)[1]
                            break
                    self.assertIsNotNone(mcp_url, "mcp server failed to start")

                    xu_src = f"""@main
#errors
anthropic #github.com/xulang-org/packages/anthropic
mcp #github.com/xulang-org/packages/mcp
agent #github.com/xulang-org/packages/agent

fn on_ev(ev agent.AgentEvent) bool {{
    println("LIVE_EV:" + ev.kind + ":" + str(ev.turn) + ":" + ev.text_delta + ":" + ev.reasoning_delta + ":" + ev.tool_name + ":" + ev.tool_output)
    << true
}}

fn run_live() int {{
    $mc := mcp.new_http_client("{mcp_url}")
    _ := mc.initialize() >>>

    ac := anthropic.new_client_with_base("{base_url}", "live-key")
    $aa := agent.new_anthropic_agent(ac, "xu-local")
    aa.set_max_tokens(16)
    aa.set_max_turns(4)

    res := aa.run_stream_with_mcp("call mul", $&mc, on_ev) >>>
    println("LIVE_RES:" + res.content + "|" + res.reasoning_content + "|" + str(res.turns) + "|" + str(res.steps.len))
    _ := mc.close_session()
    if res.steps.len != 1 || res.steps[0].tool_name != "mul" || res.steps[0].output != "42" {{ << 20 }}
    if res.content != "RESULT_42" || res.reasoning_content != "calc" {{ << 21 }}
    << 0
}}

fn main() int {{
    << run_live() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
                    drv = build_xu(work, "drv_live_anth_mcp", xu_src)
                    out = subprocess.run([str(drv)], env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
                    self.assertIn("LIVE_RES:RESULT_42|calc|2|1", out.stdout, out.stdout)
                    self.assertIn("LIVE_EV:tool_end:1:::mul:42", out.stdout, out.stdout)
                finally:
                    mcp_proc.terminate()
                    mcp_proc.wait(timeout=5)
            finally:
                srv_proc.terminate()
                try:
                    srv_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    srv_proc.kill()
                    srv_proc.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
