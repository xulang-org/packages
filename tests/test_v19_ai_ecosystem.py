"""End-to-end tests for v1.9.0 AI ecosystem upgrades:
- openai.Client.clone_client & structured ResponsesRequest.input_json helpers
- anthropic.Client.clone_client, multimodal image_base64/image_url blocks, prompt cache_control / cache_system
- gemini.Client.batch_embed
- mcp dynamic resource template handlers (resources/read matching) & logging/setLevel (HTTP + Stdio)
- agent set_tool_hook guardrail across Agent/AnthropicAgent/GeminiAgent/ResponsesAgent + ResponsesAgent & ResponsesSession
- rag delete_doc, search_by_meta, search_hybrid_by_meta, rerank_hits
- live llm_engine serve /v1/responses + ResponsesAgent + MCP integration
"""
import http.server
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import subprocess
import sys
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


class MockV19Handler(http.server.BaseHTTPRequestHandler):
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

    def _send_sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for ev_name, payload in events:
            chunk = f"event: {ev_name}\ndata: {json.dumps(payload)}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    def do_POST(self):
        req = self._read_json()

        if self.path == "/v1/responses":
            inp = req.get("input")
            prev_id = req.get("previous_response_id", "")
            if isinstance(inp, list):
                outputs = [it for it in inp if it.get("type") == "function_call_output"]
                if outputs and prev_id:
                    joined = "|".join(f"{it.get('call_id')}:{it.get('output')}" for it in outputs)
                    self._send_json(200, {
                        "id": "resp_after_tool",
                        "model": req.get("model", "gpt-5"),
                        "status": "completed",
                        "output_text": f"after={joined} prev={prev_id}",
                        "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
                    })
                    return
                txt = f"list-in len={len(inp)}"
                self._send_json(200, {
                    "id": "resp_list",
                    "model": req.get("model", "gpt-5"),
                    "status": "completed",
                    "output_text": txt,
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                })
                return

            if req.get("stream"):
                self._send_sse([
                    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Stream "}),
                    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "OK"}),
                    ("response.completed", {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_stream_done",
                            "model": req.get("model", "gpt-5"),
                            "status": "completed",
                            "output_text": "Stream OK",
                            "usage": {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13},
                        },
                    }),
                ])
                return

            if isinstance(inp, str) and "GUARD" in inp:
                self._send_json(200, {
                    "id": "resp_guard_1",
                    "model": req.get("model", "gpt-5"),
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_rm",
                            "call_id": "call_rm",
                            "name": "rm_rf",
                            "arguments": '{"path":"/"}',
                        },
                        {
                            "type": "function_call",
                            "id": "fc_add",
                            "call_id": "call_add",
                            "name": "add",
                            "arguments": '{"a":4,"b":5}',
                        },
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 6, "total_tokens": 14},
                })
                return

            self._send_json(200, {
                "id": "resp_plain_2" if prev_id else "resp_plain_1",
                "model": req.get("model", "gpt-5"),
                "status": "completed",
                "output_text": f"prev={prev_id or 'none'} inp={inp}",
                "usage": {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
            })
            return

        if self.path == "/v1/chat/completions":
            msgs = req.get("messages", [])
            if any(m.get("role") == "tool" for m in msgs):
                tmsg = [m.get("content", "") for m in msgs if m.get("role") == "tool"]
                self._send_json(200, {
                    "id": "c_2",
                    "model": req.get("model", "gpt"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "|".join(tmsg)}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                })
                return
            self._send_json(200, {
                "id": "c_1",
                "model": req.get("model", "gpt"),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "tc1", "type": "function", "function": {"name": "rm_rf", "arguments": "{}"}},
                            {"id": "tc2", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            })
            return

        if self.path == "/v1/messages":
            sys_v = req.get("system")
            m0 = req.get("messages", [{}])[0]
            blocks = m0.get("content", [])
            has_cache_sys = isinstance(sys_v, list) and sys_v[0].get("cache_control") == {"type": "ephemeral"}
            has_img_b64 = any(b.get("type") == "image" and b.get("source", {}).get("type") == "base64" and b.get("cache_control") == {"type": "ephemeral"} for b in blocks if isinstance(b, dict))
            has_img_url = any(b.get("type") == "image" and b.get("source", {}).get("type") == "url" for b in blocks if isinstance(b, dict))
            self._send_json(200, {
                "id": "msg_mm",
                "model": req.get("model", "claude"),
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": f"cs={int(has_cache_sys)} b64={int(has_img_b64)} url={int(has_img_url)}"}],
                "usage": {"input_tokens": 15, "output_tokens": 5, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 5},
            })
            return

        if self.path.endswith(":batchEmbedContents"):
            reqs = req.get("requests", [])
            embs = [{"values": [float(i + 1), float(len(r["content"]["parts"][0]["text"]))]} for i, r in enumerate(reqs)]
            self._send_json(200, {"embeddings": embs})
            return

        self._send_json(404, {"error": {"message": "not found"}})


class TestV19Ecosystem(unittest.TestCase):
    def test_v19_clients_mcp_agents_and_rag(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockV19Handler)
        port = server.server_address[1]
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            mcp_srv_src = """@main
mcp #github.com/xulang-org/packages/mcp
#encoding/json
#os

fn handle_tmpl(uri &string, vars_json &string) string {
    doc := json.parse(vars_json)
    cat_n := doc.get(doc.root(), "cat") >>> (_) { << "" }
    id_n := doc.get(doc.root(), "id") >>> (_) { << "" }
    cat := doc.as_string(cat_n) >>> (_) { << "" }
    id := doc.as_string(id_n) >>> (_) { << "" }
    << (*uri) + "|cat=" + cat + "|id=" + id
}

fn echo_tool(args_json &string) mcp.ToolResult {
    << mcp.text_result("mcp:" + (*args_json))
}

fn run_srv() int {
    $srv := mcp.new_server("v19-mcp", "1.9.0")
    srv.add_tool("mcp_ping", "Ping tool", "{\\"type\\":\\"object\\"}", echo_tool)
    srv.add_resource_template_handler("docs://{cat}/{id}", "Docs", "Dynamic docs", "text/markdown", handle_tmpl)
    args := os.args()
    if args.len >= 3 && args[1] == "--http" {
        << srv.serve_http(args[2]) >>>
    }
    << srv.serve_stdio()
}

fn main() int {
    << run_srv() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
"""
            src = f"""@main
openai #github.com/xulang-org/packages/openai
anthropic #github.com/xulang-org/packages/anthropic
gemini #github.com/xulang-org/packages/gemini
mcp #github.com/xulang-org/packages/mcp
agent #github.com/xulang-org/packages/agent
rag #github.com/xulang-org/packages/rag
#os
#strconv

fn dangerous_guard(name &string, args &string) string {{
    if *name == "rm_rf" {{ << "forbidden tool" }}
    << ""
}}

fn add_tool(args_json &string) string {{
    << "9"
}}

fn rm_tool(args_json &string) string {{
    << "SHOULD_NOT_RUN"
}}

fn on_stream(ev agent.AgentEvent) bool {{
    if ev.kind == "text_delta" {{ println("RESP_DELTA:" + ev.text_delta) }}
    << true
}}

fn run_test(base string, mcp_url string, mcp_bin string) int {{
    // 1. OpenAI clone_client + ResponsesRequest.input_json
    oai := openai.new_client(base + "", "oa-key")
    oai2 := oai.clone_client()
    $rr := openai.new_responses_request("gpt-5", "")
    items := [openai.responses_input_text_item("user", "structured"), openai.responses_function_call_output_item("c1", "ok")]
    rr.input_json = openai.responses_input_array(&items)
    r_struct := oai2.responses(&rr) >>>
    println("OAI_STRUCT:" + r_struct.output_text)

    // 2. Anthropic clone_client + image_base64/image_url + cache_control + cache_system
    ant := anthropic.new_client_with_base(base + "", "ant-key")
    ant2 := ant.clone_client()
    $img1 := anthropic.image_base64_block("image/png", "AAAA")
    img1.cache_control = "ephemeral"
    img2 := anthropic.image_url_block("https://example.com/a.png")
    txtb := anthropic.text_block("describe")
    $mreq := anthropic.messages_request("claude-4", [anthropic.message_blocks("user", [txtb, ~img1, img2])], 128)
    mreq.system = "cached system prompt"
    mreq.cache_system = true
    mresp := ant2.messages(&mreq) >>>
    println("ANT_MM:" + mresp.content + ":" + str(mresp.usage.cache_read_input_tokens))

    // 3. Gemini batch_embed
    gem := gemini.new_client_with_base(base + "", "gem-key")
    bvecs := gem.batch_embed("text-embedding-004", &["hi", "hello"]) >>>
    println("GEM_BE:" + str(bvecs.len) + ":" + str(int(bvecs[1][0])) + "," + str(int(bvecs[1][1])))

    // 4. MCP resource template handler + logging/setLevel (Stdio + HTTP)
    $sc := mcp.connect_stdio(mcp_bin, []string{{}}) >>>
    _ := sc.initialize() >>>
    _ := sc.set_logging_level("debug") >>>
    sr_txt := sc.read_resource("docs://api/42") >>>
    println("MCP_STDIO_TMPL:" + sr_txt)
    _ := sc.close()

    $hc := mcp.new_http_client(mcp_url)
    _ := hc.initialize() >>>
    _ := hc.set_logging_level("warning") >>>
    hr_txt := hc.read_resource("docs://guide/intro") >>>
    println("MCP_HTTP_TMPL:" + hr_txt)

    // 5. Agent + ResponsesAgent + ResponsesSession + set_tool_hook guardrail
    $ag := agent.new_agent(oai.clone_client(), "gpt-5")
    ag.add_tool("rm_rf", "danger", "{{\\"type\\":\\"object\\"}}", rm_tool)
    ag.add_tool("add", "add", "{{\\"type\\":\\"object\\"}}", add_tool)
    ag.set_tool_hook(dangerous_guard)
    ar := ag.run("check guard") >>>
    println("AG_GUARD:" + ar.content)

    $resp_ag := agent.new_responses_agent(oai.clone_client(), "gpt-5")
    resp_ag.add_tool("rm_rf", "danger", "{{\\"type\\":\\"object\\"}}", rm_tool)
    resp_ag.add_tool("add", "add", "{{\\"type\\":\\"object\\"}}", add_tool)
    resp_ag.set_tool_hook(dangerous_guard)
    rr2 := resp_ag.run_with_mcp("GUARD check", $&hc) >>>
    println("RESP_AG:" + rr2.content)

    $rses := agent.new_responses_session(agent.new_responses_agent(oai.clone_client(), "gpt-5"))
    sr1 := rses.send("hello1") >>>
    sr2 := rses.send("hello2") >>>
    println("RESP_SES:" + rses.last_response_id() + ":" + sr1.content + "|" + sr2.content)

    $rag_s := agent.new_responses_agent(oai.clone_client(), "gpt-5")
    sr_s := rag_s.run_stream("hello stream", on_stream) >>>
    println("RESP_STREAM:" + sr_s.content)
    _ := hc.close_session() >>>

    // 6. RAG delete_doc, search_by_meta, search_hybrid_by_meta, rerank_hits
    $store := rag.new_store(2)
    _ := store.add_chunk("c1", "d1", "alpha metal kernel", "a.md", "{{\\"env\\":\\"prod\\"}}", &[1.0, 0.0]) >>>
    _ := store.add_chunk("c2", "d1", "alpha staging note", "a.md", "{{\\"env\\":\\"staging\\"}}", &[0.95, 0.2]) >>>
    _ := store.add_chunk("c3", "d2", "beta prod metal", "b.md", "{{\\"env\\":\\"prod\\"}}", &[0.7, 0.7]) >>>
    _ := store.add_chunk("c4", "d3", "temporary doc", "c.md", "{{\\"env\\":\\"prod\\"}}", &[1.0, 0.0]) >>>
    deleted := store.delete_doc("d3")
    hm := store.search_by_meta(&[1.0, 0.0], 5, "", "env", "prod") >>>
    hyb := store.search_hybrid_by_meta("metal", &[1.0, 0.0], 5, 60, "", "env", "prod") >>>
    reranked := rag.rerank_hits(&hm, &[0.2, 0.9], 1)
    println("RAG:" + str(deleted) + ":" + str(store.len()) + ":" + str(hm.len) + ":" + str(hyb.len) + ":" + reranked[0].chunk.id)
    << 0
}}

fn main() int {{
    args := os.args()
    << run_test(args[1], args[2], args[3]) >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                mcp_exe = build_xu(work, "mcp_v19", mcp_srv_src)
                cli_exe = build_xu(work, "cli_v19", src)
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    mport = s.getsockname()[1]
                mproc = subprocess.Popen(
                    [str(mcp_exe), "--http", f"127.0.0.1:{mport}"],
                    env=xu_env(work),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    time.sleep(0.2)
                    res = subprocess.run(
                        [str(cli_exe), f"http://127.0.0.1:{port}", f"http://127.0.0.1:{mport}", str(mcp_exe)],
                        env=xu_env(work),
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                finally:
                    mproc.terminate()
                    mproc.wait(timeout=5)
            kv = dict(line.split(":", 1) for line in res.stdout.splitlines() if ":" in line)
            self.assertEqual(kv.get("OAI_STRUCT"), "list-in len=2")
            self.assertEqual(kv.get("ANT_MM"), "cs=1 b64=1 url=1:5")
            self.assertEqual(kv.get("GEM_BE"), "2:2,5")
            self.assertEqual(kv.get("MCP_STDIO_TMPL"), "docs://api/42|cat=api|id=42")
            self.assertEqual(kv.get("MCP_HTTP_TMPL"), "docs://guide/intro|cat=guide|id=intro")
            self.assertEqual(kv.get("AG_GUARD"), "blocked: forbidden tool|9")
            self.assertEqual(kv.get("RESP_AG"), "after=call_rm:blocked: forbidden tool|call_add:9 prev=resp_guard_1")
            self.assertEqual(kv.get("RESP_SES"), "resp_plain_2:prev=none inp=hello1|prev=resp_plain_1 inp=hello2")
            self.assertEqual(kv.get("RESP_STREAM"), "Stream OK")
            self.assertEqual(kv.get("RAG"), "1:3:2:2:c3")
        finally:
            server.shutdown()
            server.server_close()

    def test_v19_live_llm_engine_responses_and_responses_agent(self):
        # End-to-end live integration: llm_engine serve /v1/responses + packages/agent ResponsesAgent & ResponsesSession
        sys.path.insert(0, str(REPO / "scripts"))
        import test_llm_engine as tle

        llm_src = REPO / "examples/llm_engine/main.xu"
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            llm_exe = work / "llm_engine_v19"
            env = xu_env(work)
            res = subprocess.run(
                [str(XU), "build", str(llm_src), "-o", str(llm_exe)],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                raise AssertionError(res.stdout + res.stderr)
            if os.uname().sysname == "Darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(llm_exe)], check=True, capture_output=True)

            tensors = tle.generate_weight_dict(tle.CFG)
            vocab = tle.CFG["vocab_size"]
            dim = tle.CFG["dim"]
            pieces = [f"tok_{i}" for i in range(vocab)]
            pieces[0] = "<|endoftext|>"
            pieces[1] = "<|im_start|>"
            pieces[2] = "<|im_end|>"
            pieces[3] = "user"
            pieces[4] = "assistant"
            pieces[5] = "system"
            pieces[7] = "\n"
            pieces[9] = "hi"
            pieces[10] = "HelloResponses!"
            zt = {k: (list(s), [tle.f32(0.0)] * len(v)) for k, (s, v) in tensors.items()}
            zt["norm.weight"] = ([dim], [tle.f32(1.0)] * dim)
            emb = [tle.f32(0.0)] * (vocab * dim)
            head = [tle.f32(0.0)] * (vocab * dim)
            for tid in range(vocab):
                emb[tid * dim] = tle.f32(1.0)
            for col_idx, (src_tok, dst_tok) in enumerate([(7, 10), (10, 2)], start=1):
                emb[src_tok * dim + col_idx] = tle.f32(1.0)
                head[dst_tok * dim + 0] = tle.f32(2.0)
                head[dst_tok * dim + col_idx] = tle.f32(5.0)
            zt["embed_tokens.weight"] = ([vocab, dim], emb)
            zt["lm_head.weight"] = ([vocab, dim], head)
            gguf = work / "live_resp.gguf"
            tle.write_gguf_q4_0(gguf, tle.CFG, zt, vocab_pieces=pieces)

            srv = subprocess.Popen(
                [str(llm_exe), "serve", str(gguf), "127.0.0.1:0"],
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                url = None
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    ln = srv.stdout.readline()
                    if ln and ln.strip().startswith("serving:http://"):
                        url = ln.strip().split("serving:", 1)[1]
                        break
                self.assertIsNotNone(url)

                cli_src = """@main
openai #github.com/xulang-org/packages/openai
agent #github.com/xulang-org/packages/agent
#os

fn run_live(url string) int {
    cli := openai.new_client(url, "")
    $ra := agent.new_responses_agent(cli, "local-q4")
    ra.set_max_output_tokens(4)
    $ses := agent.new_responses_session(ra)
    r1 := ses.send("hi") >>>
    rid1 := ses.last_response_id()
    r2 := ses.send("hi") >>>
    rid2 := ses.last_response_id()
    println("LIVE_RESP:" + r1.content + "|" + r2.content + "|" + str(rid1 != rid2) + "|" + str(r2.prompt_tokens > r1.prompt_tokens))
    << 0
}

fn main() int {
    args := os.args()
    << run_live(args[1]) >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
"""
                cli_exe = build_xu(work, "live_cli_v19", cli_src)
                out = subprocess.run([str(cli_exe), url], env=env, capture_output=True, text=True, check=True).stdout
            finally:
                srv.terminate()
                srv.wait(timeout=5)
            kv = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
            self.assertEqual(kv.get("LIVE_RESP"), "HelloResponses!|HelloResponses!|true|true")


if __name__ == "__main__":
    unittest.main()
