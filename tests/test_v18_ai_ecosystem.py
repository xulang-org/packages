"""End-to-end tests for v1.8.0 AI ecosystem upgrades:
- openai /v1/responses (sync + SSE stream + ResponsesToolCall + ResponsesStreamEvent)
- mcp ToolAnnotations, ResourceTemplateInfo (resources/templates/list), completion/complete (HTTP + Stdio)
- gemini.Client.clone_client
- agent GeminiAgent, GeminiSession, AnthropicSession, set_max_total_tokens
- rag search_bm25_filtered, search_hybrid_filtered, search_mmr
"""
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


class MockV18Handler(http.server.BaseHTTPRequestHandler):
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
            chunk = ""
            if ev_name:
                chunk += f"event: {ev_name}\n"
            chunk += f"data: {json.dumps(payload) if isinstance(payload, dict) else payload}\n\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    def do_POST(self):
        req = self._read_json()

        # OpenAI /v1/responses
        if self.path == "/v1/responses":
            if self.headers.get("Authorization") != "Bearer oa-key":
                self._send_json(401, {"error": {"message": "bad openai key"}})
                return
            if req.get("stream"):
                self._send_sse([
                    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Hello "}),
                    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Responses!"}),
                    ("response.output_item.done", {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": "fc_s1",
                            "call_id": "call_s1",
                            "name": "lookup",
                            "arguments": '{"q":"xu"}'
                        }
                    }),
                    ("response.completed", {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_stream_1",
                            "model": req.get("model", "gpt-5"),
                            "status": "completed",
                            "output_text": "Hello Responses!",
                            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
                        }
                    }),
                ])
                return
            prev_id = req.get("previous_response_id", "")
            self._send_json(200, {
                "id": "resp_sync_2" if prev_id else "resp_sync_1",
                "model": req.get("model", "gpt-5"),
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "weather",
                        "arguments": '{"city":"Tokyo"}'
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f"Prev={prev_id or 'none'} Input={req.get('input', '')}"}
                        ]
                    }
                ],
                "usage": {"input_tokens": 14, "output_tokens": 9, "total_tokens": 23}
            })
            return

        # OpenAI /v1/chat/completions (for Agent budget check)
        if self.path == "/v1/chat/completions":
            self._send_json(200, {
                "id": "chat_1",
                "model": req.get("model", "gpt-5"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "heavy"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 40, "completion_tokens": 25, "total_tokens": 65}
            })
            return

        # Anthropic /v1/messages (for AnthropicSession + budget check)
        if self.path == "/v1/messages":
            msgs = req.get("messages", [])
            last_text = ""
            if msgs and isinstance(msgs[-1].get("content"), str):
                last_text = msgs[-1]["content"]
            if "BUDGET" in last_text:
                self._send_json(200, {
                    "id": "msg_b",
                    "model": req.get("model", "claude"),
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "too expensive"}],
                    "usage": {"input_tokens": 45, "output_tokens": 20}
                })
                return
            self._send_json(200, {
                "id": f"msg_{len(msgs)}",
                "model": req.get("model", "claude"),
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": f"anth-turns={len(msgs)} last={last_text}"}],
                "usage": {"input_tokens": 10, "output_tokens": 5}
            })
            return

        # Gemini generateContent / streamGenerateContent
        if self.path.startswith("/v1beta/models/gemini-2.5-pro:"):
            if self.headers.get("x-goog-api-key") != "gem-v18" or self.headers.get("X-Custom-Hdr") != "ok":
                self._send_json(401, {"error": {"message": "missing headers"}})
                return
            contents = req.get("contents", [])
            has_tool_resp = any(
                any("functionResponse" in p for p in c.get("parts", []))
                for c in contents
            )
            first_text = ""
            for c in contents:
                for p in c.get("parts", []):
                    if "text" in p:
                        first_text = p["text"]
            is_stream = self.path.endswith(":streamGenerateContent?alt=sse")

            if "BUDGET" in first_text:
                self._send_json(200, {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": "over budget"}]},
                        "finishReason": "STOP"
                    }],
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 30, "totalTokenCount": 80}
                })
                return

            if is_stream:
                if not has_tool_resp and "TOOL" in first_text:
                    self._send_sse([
                        ("", {
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"thought": True, "text": "planning tool"},
                                        {"functionCall": {"name": "add_nums", "args": {"a": 6, "b": 7}}}
                                    ]
                                },
                                "finishReason": "STOP"
                            }],
                            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8, "totalTokenCount": 20}
                        })
                    ])
                    return
                tool_out = ""
                if has_tool_resp:
                    for c in contents:
                        for p in c.get("parts", []):
                            if "functionResponse" in p:
                                tool_out = p["functionResponse"]["response"].get("result", "")
                self._send_sse([
                    ("", {
                        "candidates": [{"content": {"role": "model", "parts": [{"thought": True, "text": "checking "}]} }],
                    }),
                    ("", {
                        "candidates": [{
                            "content": {"role": "model", "parts": [{"text": f"gem-ans={tool_out}"}]},
                            "finishReason": "STOP"
                        }],
                        "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 6, "totalTokenCount": 21}
                    })
                ])
                return

            # Non-streaming Gemini session response
            self._send_json(200, {
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": f"gem-turns={len(contents)} last={first_text}"}]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12}
            })
            return

        self._send_json(404, {"error": {"message": f"unknown path {self.path}"}})


class TestV18AIEcosystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), MockV18Handler)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_v18_responses_gemini_agent_sessions_budget_and_rag_mmr(self):
        base = f"http://127.0.0.1:{self.port}"
        src = f"""@main
openai #github.com/xulang-org/packages/openai
anthropic #github.com/xulang-org/packages/anthropic
gemini #github.com/xulang-org/packages/gemini
agent #github.com/xulang-org/packages/agent
rag #github.com/xulang-org/packages/rag
#encoding/json

fn json_int_val(doc &json.Doc, r int, key string) int {{
    if !doc.has(r, key) {{ << 0 }}
    n := doc.get(r, key) >>> (_) {{ << 0 }}
    << doc.as_int(n) >>> (_) {{ << 0 }}
}}

fn add_handler(args_json &string) string {{
    doc := json.parse(args_json)
    if doc.ok() && doc.is_dict(doc.root()) {{
        r := doc.root()
        << str(json_int_val(&doc, r, "a") + json_int_val(&doc, r, "b"))
    }}
    << "0"
}}

fn on_stream_ev(ev openai.ResponsesStreamEvent) bool {{
    if ev.text_delta.len > 0 {{
        println("RTD:" + ev.text_delta)
    }}
    << true
}}

fn on_gem_ev(ev agent.AgentEvent) bool {{
    if ev.kind == "reasoning_delta" {{ println("GRD:" + ev.reasoning_delta) }}
    if ev.kind == "tool_end" {{ println("GTE:" + ev.tool_name + "=" + ev.tool_output) }}
    if ev.kind == "text_delta" {{ println("GTD:" + ev.text_delta) }}
    << true
}}

fn catch_oa_err(a &agent.Agent) string {{
    _ := a.run("BUDGET") >>> (e) {{ << e.msg() }}
    << ""
}}

fn catch_aa_err(a &agent.AnthropicAgent) string {{
    _ := a.run("BUDGET") >>> (e) {{ << e.msg() }}
    << ""
}}

fn catch_ga_err(a &agent.GeminiAgent) string {{
    _ := a.run("BUDGET") >>> (e) {{ << e.msg() }}
    << ""
}}

fn run_eco() int {{
    // 1. OpenAI Responses API (sync + previous_response_id + SSE)
    $oc := openai.new_client("{base}", "oa-key")
    $rr1 := openai.new_responses_request("gpt-5", "Ping")
    rr1.instructions = "Be concise"
    rr1.tools.push(openai.tool("weather", "Weather", "{{\\"type\\":\\"object\\"}}"))
    res1 := oc.responses(&rr1) >>>
    println("R1_ID:" + res1.id)
    println("R1_ST:" + res1.status)
    println("R1_TXT:" + res1.output_text)
    println("R1_TC:" + str(res1.tool_calls.len) + ":" + res1.tool_calls[0].name + ":" + res1.tool_calls[0].call_id)
    println("R1_TOK:" + str(res1.usage.total_tokens))

    $rr2 := openai.new_responses_request("gpt-5", "Follow")
    rr2.previous_response_id = res1.id + ""
    res2 := oc.responses(&rr2) >>>
    println("R2_TXT:" + res2.output_text)

    $rrs := openai.new_responses_request("gpt-5", "Stream")
    res_s := oc.responses_stream(&rrs, on_stream_ev) >>>
    println("RS_OUT:" + res_s.id + ":" + res_s.output_text + ":" + str(res_s.tool_calls.len) + ":" + str(res_s.usage.total_tokens))

    // 2. Gemini clone_client + GeminiAgent streaming tool call + GeminiSession
    $gc := gemini.new_client_with_base("{base}", "gem-v18")
    gc.add_header("X-Custom-Hdr", "ok")
    gc_copy := gc.clone_client()
    $ga := agent.new_gemini_agent(~gc_copy, "gemini-2.5-pro")
    ga.set_system_prompt("Use tools.")
    ga.add_tool("add_nums", "Add two ints", "{{\\"type\\":\\"object\\"}}", add_handler)
    gr := ga.run_stream("Please run TOOL now", on_gem_ev) >>>
    println("GA_FIN:" + gr.content + "|turns=" + str(gr.turns) + "|steps=" + str(gr.steps.len) + "|tok=" + str(gr.total_tokens))

    $gs := agent.new_gemini_session(agent.new_gemini_agent(gc.clone_client(), "gemini-2.5-pro"))
    gs1 := gs.send("first gem msg") >>>
    gs2 := gs.send("second gem msg") >>>
    println("GS:" + gs1.content + "|" + gs2.content + "|hlen=" + str(gs.history().len))
    gs.clear()
    println("GS_CLR:" + str(gs.history().len))

    // 3. AnthropicSession
    ac := anthropic.new_client_with_base("{base}", "anth-key")
    $asess := agent.new_anthropic_session(agent.new_anthropic_agent(ac, "claude-4"))
    ar1 := asess.send("hello anth") >>>
    ar2 := asess.send("next anth") >>>
    println("AS:" + ar1.content + "|" + ar2.content + "|hlen=" + str(asess.history().len))
    asess.clear()
    println("AS_CLR:" + str(asess.history().len))

    // 4. Token budgets across Agent / AnthropicAgent / GeminiAgent
    $oa_bud := agent.new_agent(openai.new_client("{base}", "oa-key"), "gpt-5")
    oa_bud.set_max_total_tokens(50)
    oa_err := catch_oa_err(&oa_bud)
    println("BUD_OA:" + oa_err)

    $aa_bud := agent.new_anthropic_agent(anthropic.new_client_with_base("{base}", "anth-key"), "claude-4")
    aa_bud.set_max_total_tokens(50)
    aa_err := catch_aa_err(&aa_bud)
    println("BUD_AA:" + aa_err)

    $ga_bud := agent.new_gemini_agent(gc.clone_client(), "gemini-2.5-pro")
    ga_bud.set_max_total_tokens(50)
    ga_err := catch_ga_err(&ga_bud)
    println("BUD_GA:" + ga_err)

    // 5. RAG v1.8.0: search_bm25_filtered, search_hybrid_filtered, search_mmr
    $st := rag.new_store(3)
    _ := st.add_chunk("c1a", "d1", "xu language metal tensor kernel", "s1", "{{}}", &[1.0, 0.05, 0.0]) >>>
    _ := st.add_chunk("c1b", "d1", "xu language tensor matrix compute", "s1", "{{}}", &[0.99, 0.06, 0.0]) >>>
    _ := st.add_chunk("c2",  "d2", "xu metal agent mmr diversity search", "s2", "{{}}", &[0.75, -0.66, 0.0]) >>>
    bf := st.search_bm25_filtered("xu metal", 5, "d2")
    println("BM25F:" + str(bf.len) + ":" + bf[0].chunk.id)
    qemb := [1.0, 0.0, 0.0]
    hf := st.search_hybrid_filtered("xu metal", &qemb, 5, 60, "d2")
    println("HYBF:" + str(hf.len) + ":" + hf[0].chunk.id)
    mmr_hits := st.search_mmr(&qemb, 3, 0.5, "")
    println("MMR:" + mmr_hits[0].chunk.id + "," + mmr_hits[1].chunk.id + "," + mmr_hits[2].chunk.id)
    mmr_f := st.search_mmr(&qemb, 3, 0.5, "d1")
    println("MMRF:" + str(mmr_f.len) + ":" + mmr_f[0].chunk.id + "," + mmr_f[1].chunk.id)

    << 0
}}

fn main() int {{
    << run_eco() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            exe = build_xu(work, "v18_eco", src)
            res = subprocess.run([str(exe)], env=xu_env(work), capture_output=True, text=True, check=True)
        out = res.stdout
        lines = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
        self.assertEqual(lines.get("R1_ID"), "resp_sync_1")
        self.assertEqual(lines.get("R1_ST"), "completed")
        self.assertEqual(lines.get("R1_TXT"), "Prev=none Input=Ping")
        self.assertEqual(lines.get("R1_TC"), "1:weather:call_1")
        self.assertEqual(lines.get("R1_TOK"), "23")
        self.assertEqual(lines.get("R2_TXT"), "Prev=resp_sync_1 Input=Follow")
        self.assertIn("RTD:Hello ", out)
        self.assertIn("RTD:Responses!", out)
        self.assertEqual(lines.get("RS_OUT"), "resp_stream_1:Hello Responses!:1:18")
        self.assertIn("GRD:planning tool", out)
        self.assertIn("GTE:add_nums=13", out)
        self.assertIn("GTD:gem-ans=13", out)
        self.assertEqual(lines.get("GA_FIN"), "gem-ans=13|turns=2|steps=1|tok=41")
        self.assertEqual(lines.get("GS"), "gem-turns=1 last=first gem msg|gem-turns=3 last=second gem msg|hlen=4")
        self.assertEqual(lines.get("GS_CLR"), "0")
        self.assertEqual(lines.get("AS"), "anth-turns=1 last=hello anth|anth-turns=3 last=next anth|hlen=4")
        self.assertEqual(lines.get("AS_CLR"), "0")
        self.assertIn("token budget exceeded", lines.get("BUD_OA", ""))
        self.assertIn("token budget exceeded", lines.get("BUD_AA", ""))
        self.assertIn("token budget exceeded", lines.get("BUD_GA", ""))
        self.assertEqual(lines.get("BM25F"), "1:c2")
        self.assertEqual(lines.get("HYBF"), "1:c2")
        self.assertEqual(lines.get("MMR"), "c1a,c2,c1b")
        self.assertEqual(lines.get("MMRF"), "2:c1a,c1b")

    def test_mcp_annotations_templates_and_completions_http_and_stdio(self):
        srv_src = """@main
mcp #github.com/xulang-org/packages/mcp
#os

fn echo_tool(args &string) mcp.ToolResult {
    << mcp.text_result("ok:" + *args)
}

fn on_complete(ref_json &string, arg_name &string, arg_val &string) []string {
    if *arg_name == "id" {
        << ["alpha", "alpine", *arg_val]
    }
    << ["default"]
}

fn run_srv() int {
    $srv := mcp.new_server("xu-v18-mcp", "1.8.0")
    ann := mcp.ToolAnnotations{
        title: "Readonly Echo",
        read_only_hint: true,
        destructive_hint: false,
        idempotent_hint: true,
        open_world_hint: false,
        has_annotations: true
    }
    srv.add_tool_annotated("echo", "Echo with annotations", "{\\"type\\":\\"object\\"}", ann, echo_tool)
    srv.add_resource_template("db://items/{id}", "ItemTemplate", "Dynamic item lookup", "application/json")
    srv.set_completion_handler(on_complete)
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
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            srv_exe = build_xu(work, "mcp_srv_v18", srv_src)

            # Test StdioClient + HTTP Client together from one Xu runner
            import socket
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                http_port = s.getsockname()[1]
            http_proc = subprocess.Popen(
                [str(srv_exe), "--http", f"127.0.0.1:{http_port}"],
                env=xu_env(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                cli_src = """@main
mcp #github.com/xulang-org/packages/mcp

fn check_client_http(url string) int {
    $hc := mcp.new_http_client(url)
    _ := hc.initialize() >>>
    tools := hc.list_tools() >>>
    println("HTTP_TOOL:" + tools[0].name + ":" + tools[0].annotations.title + ":" +
        str(tools[0].annotations.read_only_hint) + ":" + str(tools[0].annotations.idempotent_hint) + ":" +
        str(tools[0].annotations.has_annotations))
    tmpls := hc.list_resource_templates() >>>
    println("HTTP_TMPL:" + str(tmpls.len) + ":" + tmpls[0].uri_template + ":" + tmpls[0].mime_type)
    comp := hc.complete("ref/resource", "db://items/{id}", "id", "al") >>>
    println("HTTP_COMP:" + comp[0] + "," + comp[1] + "," + comp[2])
    _ := hc.close_session() >>>
    << 0
}

fn check_client_stdio(bin_path string) int {
    $sc := mcp.connect_stdio(bin_path, []string{}) >>>
    _ := sc.initialize() >>>
    tools := sc.list_tools() >>>
    println("STDIO_TOOL:" + tools[0].name + ":" + tools[0].annotations.title + ":" + str(tools[0].annotations.destructive_hint))
    tmpls := sc.list_resource_templates() >>>
    println("STDIO_TMPL:" + str(tmpls.len) + ":" + tmpls[0].name)
    comp := sc.complete("ref/resource", "db://items/{id}", "id", "xy") >>>
    println("STDIO_COMP:" + comp[0] + "," + comp[1] + "," + comp[2])
    _ := sc.close()
    << 0
}

fn run_cli() int {
    _ := check_client_stdio("__SRV_EXE__") >>>
    _ := check_client_http("http://127.0.0.1:__HTTP_PORT__") >>>
    << 0
}

fn main() int {
    << run_cli() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
""".replace("__SRV_EXE__", str(srv_exe)).replace("__HTTP_PORT__", str(http_port))
                import time
                time.sleep(0.2)
                cli_exe = build_xu(work, "mcp_cli_v18", cli_src)
                res = subprocess.run([str(cli_exe)], env=xu_env(work), capture_output=True, text=True, check=True)
            finally:
                http_proc.terminate()
                http_proc.wait(timeout=5)
        kv = dict(line.split(":", 1) for line in res.stdout.splitlines() if ":" in line)
        self.assertEqual(kv.get("STDIO_TOOL"), "echo:Readonly Echo:false")
        self.assertEqual(kv.get("STDIO_TMPL"), "1:ItemTemplate")
        self.assertEqual(kv.get("STDIO_COMP"), "alpha,alpine,xy")
        self.assertEqual(kv.get("HTTP_TOOL"), "echo:Readonly Echo:true:true:true")
        self.assertEqual(kv.get("HTTP_TMPL"), "1:db://items/{id}:application/json")
        self.assertEqual(kv.get("HTTP_COMP"), "alpha,alpine,al")


if __name__ == "__main__":
    unittest.main()
