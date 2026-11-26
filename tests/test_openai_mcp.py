"""End-to-end tests for github.com/xulang-org/packages/{openai,mcp,agent}."""
import http.server
import json
import os
from pathlib import Path
import socketserver
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
RETRY_COUNTS = {}


class MockOpenAIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _check_auth(self):
        if self.headers.get("Authorization") != "Bearer test-secret":
            body = json.dumps({"error": {"message": "invalid key"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False
        return True

    def do_GET(self):
        if not self._check_auth():
            return
        if self.path == "/v1/models":
            body = json.dumps({
                "object": "list",
                "data": [
                    {"id": "xu-7b", "object": "model"},
                    {"id": "xu-embed", "object": "model"},
                ],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        if not self._check_auth():
            return
        cl = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(cl).decode())
        if self.path == "/v1/completions":
            if self.headers.get("X-Verify-Completions") == "1":
                assert req["prompt"] == "Say hi", req
                assert abs(req["temperature"] - 0.2) < 1e-3, req
                assert req["max_tokens"] == 16, req
                assert req["seed"] == 99, req
                assert req["stop"] == ["END"], req
                assert req["echo"] is True, req
                assert req["top_k"] == 5, req
            body = json.dumps({
                "id": "cmpl-1",
                "model": req["model"],
                "choices": [{"index": 0, "text": "completed:" + req["prompt"], "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/embeddings":
            inputs = req["input"]
            data = [{"object": "embedding", "index": i, "embedding": [float(i + 1), 0.5]} for i, _ in enumerate(inputs)]
            body = json.dumps({
                "object": "list",
                "model": req["model"],
                "data": data,
                "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/v1/chat/completions":
            retry_key = self.headers.get("X-Retry-Key")
            if retry_key:
                with RETRY_LOCK:
                    n = RETRY_COUNTS.get(retry_key, 0) + 1
                    RETRY_COUNTS[retry_key] = n
                if n == 1:
                    body = json.dumps({"error": {"message": "rate limited"}}).encode()
                    self.send_response(429)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if n == 2 and not req.get("stream"):
                    body = json.dumps({"error": {"message": "upstream busy"}}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                mode = self.headers.get("X-Stream-Mode", "text")
                if mode == "tools":
                    chunks = [
                        {
                            "id": "chatcmpl-st1",
                            "model": req["model"],
                            "choices": [{"index": 0, "delta": {"reasoning_content": "plan1 "}, "finish_reason": None}],
                        },
                        {
                            "id": "chatcmpl-st1",
                            "model": req["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "reasoning_content": "plan2",
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": "call_s1",
                                        "type": "function",
                                        "function": {"name": "add", "arguments": "{\"a\":2,"},
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        },
                        {
                            "id": "chatcmpl-st1",
                            "model": req["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "function": {"arguments": "\"b\":3}"},
                                    }],
                                },
                                "finish_reason": "tool_calls",
                            }],
                        },
                    ]
                else:
                    chunks = [
                        {"id": "chatcmpl-s1", "model": req["model"], "choices": [{"index": 0, "delta": {"content": "Hello "}, "finish_reason": None}]},
                        {"id": "chatcmpl-s1", "model": req["model"], "choices": [{"index": 0, "delta": {"content": "Xu!"}, "finish_reason": "stop"}]},
                        {"id": "chatcmpl-s1", "model": req["model"], "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
                    ]
                for c in chunks:
                    self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            if self.headers.get("X-Agent-Test") == "1":
                msgs = req["messages"]
                last = msgs[-1]
                tool_names = [t["function"]["name"] for t in req.get("tools", [])]
                exp_choice = self.headers.get("X-Agent-Expect-Choice")
                if exp_choice:
                    assert req.get("tool_choice") == exp_choice, req
                exp_fmt = self.headers.get("X-Agent-Expect-Format")
                if exp_fmt:
                    assert req.get("response_format", {}).get("type") == exp_fmt, req
                if self.headers.get("X-Agent-Session") == "1":
                    user_count = sum(1 for m in msgs if m["role"] == "user")
                    msg = {"role": "assistant", "content": f"TURN{user_count}:total={len(msgs)}"}
                    fr = "stop"
                elif last["role"] != "tool":
                    target_tool = self.headers.get("X-Agent-Tool", "mul")
                    args_json = self.headers.get("X-Agent-Args", '{"a":6,"b":7}')
                    assert target_tool in tool_names, (target_tool, tool_names)
                    msg = {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "calling " + target_tool,
                        "tool_calls": [{
                            "id": "call_agent_1",
                            "type": "function",
                            "function": {"name": target_tool, "arguments": args_json},
                        }],
                    }
                    fr = "tool_calls"
                else:
                    msg = {
                        "role": "assistant",
                        "content": "FINAL:" + last["content"],
                        "reasoning_content": "done",
                    }
                    fr = "stop"
                body = json.dumps({
                    "id": "chatcmpl-agent",
                    "model": req["model"],
                    "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
                    "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.headers.get("X-Verify-Vision") == "1":
                parts = req["messages"][0]["content"]
                assert isinstance(parts, list) and len(parts) == 3, req
                assert parts[0] == {"type": "text", "text": "describe image"}, parts
                assert parts[1] == {"type": "image_url", "image_url": {"url": "https://example.com/a.png", "detail": "high"}}, parts
                assert parts[2] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"}}, parts
                assert req["response_format"]["type"] == "json_schema", req
                assert req["response_format"]["json_schema"]["strict"] is True, req
                assert req["top_k"] == 7, req
                msg = {"role": "assistant", "content": "{\"ok\":true}"}
                fr = "stop"
            elif self.headers.get("X-Verify-Params") == "1":
                assert abs(req["top_p"] - 0.9) < 1e-3, req
                assert abs(req["presence_penalty"] - 0.25) < 1e-3, req
                assert abs(req["frequency_penalty"] - 0.5) < 1e-3, req
                assert req["max_completion_tokens"] == 64, req
                assert req["tool_choice"] == "auto", req
                assert req["response_format"] == {"type": "json_object"}, req
                assert self.headers.get("X-Custom-Org") == "xu-org", dict(self.headers)
                msg = {
                    "role": "assistant",
                    "content": "{\"ans\":5}",
                    "reasoning_content": "thought-step",
                }
                fr = "stop"
            elif bool(req.get("tools")):
                msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
                    }],
                }
                fr = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "Answer is 5"}
                fr = "stop"
            body = json.dumps({
                "id": "chatcmpl-1",
                "model": req["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class MockMCPHTTPHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        cl = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(cl).decode())
        method = req.get("method")
        rid = req.get("id")
        if rid is None and method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        extra_hdrs = {}
        if method == "initialize":
            extra_hdrs["Mcp-Session-Id"] = "sess-123"
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "py-mcp-http", "version": "0.1.0"},
            }
        elif method == "ping":
            assert self.headers.get("Mcp-Session-Id") == "sess-123"
            result = {}
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo msg", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            msg = req["params"]["arguments"]["msg"]
            payload = json.dumps({"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": "echo:" + msg}]}})
            body = ("event: message\ndata: " + payload + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif method == "resources/list":
            result = {"resources": [{"uri": "doc://readme", "name": "readme", "description": "Doc", "mimeType": "text/plain"}]}
        elif method == "resources/read":
            result = {"contents": [{"uri": req["params"]["uri"], "mimeType": "text/plain", "text": "HTTP resource text"}]}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "summarize", "description": "Summarize text", "arguments": [{"name": "topic", "required": True}]}]}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": "Topic: " + req["params"]["arguments"]["topic"]}}]}
        else:
            result = {}

        body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        for k, v in extra_hdrs.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


STDIO_MCP_SRV_SRC = """@main
#encoding/json
#errors
mcp #"github.com/xulang-org/packages/mcp"

fn add_handler(args_json &string) mcp.ToolResult {
    doc := json.parse(args_json)
    if !doc.ok() || !doc.is_dict(doc.root()) {
        << mcp.error_result("invalid args")
    }
    root := doc.root()
    an := doc.get(root, "a") >>> (_) { << mcp.error_result("missing a") }
    bn := doc.get(root, "b") >>> (_) { << mcp.error_result("missing b") }
    av := doc.as_int(an) >>> (_) { << mcp.error_result("bad a") }
    bv := doc.as_int(bn) >>> (_) { << mcp.error_result("bad b") }
    << mcp.text_result(str(av + bv))
}

fn res_handler(uri &string) string {
    << "hello from mcp resource:" + *uri
}

fn prompt_handler(args_json &string) string {
    doc := json.parse(args_json)
    if doc.ok() && doc.has(doc.root(), "name") {
        n := doc.get(doc.root(), "name") >>> (_) { << "world" }
        << "Hello " + doc.as_string(n) >>> (_) { << "world" }
    }
    << "Hello"
}

fn main() int {
    $srv := mcp.new_server("xu-test-mcp", "1.5.0")
    srv.add_tool("add", "Add a and b", "{\\"type\\":\\"object\\"}", add_handler)
    srv.add_resource("memo://hello", "hello", "Greeting resource", "text/plain", res_handler)
    srv.add_prompt("greet", "Greet by name", "[{\\"name\\":\\"name\\"}]", prompt_handler)
    << srv.serve_stdio() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
"""


HTTP_MCP_SRV_SRC = """@main
#encoding/json
#errors
#net/http
mcp #"github.com/xulang-org/packages/mcp"

fn mul_handler(args_json &string) mcp.ToolResult {
    doc := json.parse(args_json)
    if !doc.ok() || !doc.is_dict(doc.root()) {
        << mcp.error_result("invalid args")
    }
    root := doc.root()
    an := doc.get(root, "a") >>> (_) { << mcp.error_result("missing a") }
    bn := doc.get(root, "b") >>> (_) { << mcp.error_result("missing b") }
    av := doc.as_int(an) >>> (_) { << mcp.error_result("bad a") }
    bv := doc.as_int(bn) >>> (_) { << mcp.error_result("bad b") }
    << mcp.text_result(str(av * bv))
}

fn res_handler(uri &string) string {
    << "xu-http-res:" + *uri
}

fn prompt_handler(args_json &string) string {
    << "xu-http-prompt:" + *args_json
}

fn run() int {
    $msrv := mcp.new_server("xu-http-mcp", "1.5.0")
    msrv.add_tool("http_mul", "Multiply ints", "{\\"type\\":\\"object\\"}", mul_handler)
    msrv.add_resource("mem://http", "http", "HTTP res", "text/plain", res_handler)
    msrv.add_prompt("ask", "Ask prompt", "[]", prompt_handler)
    $hsrv := http.new_server()
    _ := hsrv.listen("127.0.0.1:0") >>>
    println("serving:http://" + hsrv.addr() + "/mcp")
    _ := msrv.serve_http_server($&hsrv) >>>
    _ := hsrv.close()
    << 0
}

fn main() int {
    << run() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
"""


class TestOpenAIAndMCP(unittest.TestCase):
    def test_openai_client_end_to_end(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}"
            xu_src = f"""@main
#errors
openai #"github.com/xulang-org/packages/openai"

fn keep_streaming(d openai.StreamDelta) bool {{
    << true
}}

fn run() int {{
    c := openai.new_client("{url}", "test-secret")
    models := c.list_models() >>>
    if models.len != 2 || models[0].id != "xu-7b" || models[1].id != "xu-embed" {{
        << 11
    }}

    msgs1 := [openai.message("user", "Add 2 and 3")]
    $req1 := openai.chat_request("xu-7b", msgs1)
    req1.tools.push(openai.tool("add", "Add two ints", "{{\\"type\\":\\"object\\"}}"))
    resp1 := c.chat(&req1) >>>
    if resp1.finish_reason != "tool_calls" || resp1.tool_calls.len != 1 {{
        << 12
    }}
    if resp1.tool_calls[0].name != "add" || resp1.usage.total_tokens != 13 {{
        << 13
    }}

    msgs2 := [openai.message("user", "hi")]
    $req2 := openai.chat_request("xu-7b", msgs2)
    req2.include_usage = true
    ev_ch := chan openai.StreamDelta{{cap: 16}}
    resp2 := c.chat_stream_chan(&req2, ev_ch) >>>
    resp2_cb := c.chat_stream(&req2, keep_streaming) >>>
    if resp2_cb.content != "Hello Xu!" {{ << 19 }}
    ev_ch <- openai.StreamDelta{{content: "", reasoning_content: "", finish_reason: "", tool_calls: []openai.ToolCall{{}}, has_usage: false, usage: resp2.usage, done: false}}
    $deltas := 0
    $done_seen := false
    for {{
        d := <-ev_ch
        if d.done {{
            done_seen = true
        }} else if d.content.len > 0 {{
            deltas += 1
        }} else if !d.has_usage && d.finish_reason.len == 0 {{
            <-
        }}
    }}
    if !done_seen || deltas != 2 || resp2.content != "Hello Xu!" || resp2.usage.completion_tokens != 2 {{
        << 14
    }}

    ereq := openai.embed_request("xu-embed", ["alpha", "beta"])
    eresp := c.embeddings(&ereq) >>>
    if eresp.embeddings.len != 2 || eresp.embeddings[0].len != 2 || eresp.usage.prompt_tokens != 4 {{
        << 15
    }}

    // v1.5.0: custom headers, retry on 429/503, reasoning_content, extended params
    $rc := openai.new_client("{url}", "test-secret")
    rc.set_max_retries(3)
    rc.add_header("X-Custom-Org", "xu-org")
    rc.add_header("X-Verify-Params", "1")
    rc.add_header("X-Retry-Key", "chat-r1")
    msgs3 := [openai.message("user", "json pls")]
    $req3 := openai.chat_request("xu-7b", msgs3)
    req3.top_p = 0.9
    req3.has_top_p = true
    req3.presence_penalty = 0.25
    req3.has_presence_penalty = true
    req3.frequency_penalty = 0.5
    req3.has_frequency_penalty = true
    req3.max_completion_tokens = 64
    req3.tool_choice = "auto"
    req3.response_format_json = "{{\\"type\\":\\"json_object\\"}}"
    resp3 := rc.chat(&req3) >>>
    if resp3.reasoning_content != "thought-step" || resp3.content != "{{\\"ans\\":5}}" {{
        << 17
    }}

    // v1.5.0: streaming tool_calls index accumulation + reasoning_content + stream retry
    $sc := openai.new_client("{url}", "test-secret")
    sc.set_max_retries(2)
    sc.add_header("X-Stream-Mode", "tools")
    sc.add_header("X-Retry-Key", "stream-r1")
    msgs4 := [openai.message("user", "call add")]
    $req4 := openai.chat_request("xu-7b", msgs4)
    resp4 := sc.chat_stream(&req4, keep_streaming) >>>
    if resp4.reasoning_content != "plan1 plan2" || resp4.finish_reason != "tool_calls" {{
        << 18
    }}
    if resp4.tool_calls.len != 1 || resp4.tool_calls[0].id != "call_s1" || resp4.tool_calls[0].name != "add" || resp4.tool_calls[0].arguments != "{{\\"a\\":2,\\"b\\":3}}" {{
        << 20
    }}

    // v1.6.0: multimodal ContentPart, json_schema_format, extra_body_json, /v1/completions
    $vc := openai.new_client("{url}", "test-secret")
    vc.add_header("X-Verify-Vision", "1")
    vparts := [openai.text_part("describe image"), openai.image_url_part("https://example.com/a.png", "high"), openai.image_base64_part("image/png", "AAAA", "low")]
    vmsgs := [openai.message_parts("user", vparts)]
    $vreq := openai.chat_request("xu-7b", vmsgs)
    vreq.response_format_json = openai.json_schema_format("out", "{{\\"type\\":\\"object\\"}}", true)
    vreq.extra_body_json = "{{\\"top_k\\":7}}"
    vresp := vc.chat(&vreq) >>>
    if vresp.content != "{{\\"ok\\":true}}" {{
        << 21
    }}

    $cc := openai.new_client("{url}", "test-secret")
    cc.add_header("X-Verify-Completions", "1")
    $creq := openai.new_completion_request("xu-7b", "Say hi")
    creq.temperature = 0.2
    creq.has_temperature = true
    creq.max_tokens = 16
    creq.seed = 99
    creq.has_seed = true
    creq.stop = ["END"]
    creq.echo = true
    creq.extra_body_json = "{{\\"top_k\\":5}}"
    cresp := cc.completions(&creq) >>>
    if cresp.text != "completed:Say hi" || cresp.finish_reason != "stop" || cresp.usage.total_tokens != 5 {{
        << 22
    }}

    bad := openai.new_client("{url}", "wrong-key")
    _ := bad.list_models() >>> (err) {{
        println("AUTH_ERR:" + err.msg())
        << 0
    }}
    << 16
}}

fn main() int {{
    rc := run() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
    << rc
}}
"""
            with tempfile.TemporaryDirectory() as td:
                exe = build_xu(Path(td), "openai_test", xu_src)
                res = subprocess.run([str(exe)], env=xu_env(Path(td)), text=True, capture_output=True)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertIn("AUTH_ERR:invalid key", res.stdout)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_mcp_stdio_and_http_roundtrip(self):
        http_srv = ThreadingHTTPServer(("127.0.0.1", 0), MockMCPHTTPHandler)
        port = http_srv.server_address[1]
        t = threading.Thread(target=http_srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}/mcp"
            client_src = f"""@main
#errors
mcp #"github.com/xulang-org/packages/mcp"

fn run() int {{
    $c := mcp.new_http_client("{url}")
    _ := c.initialize() >>>
    if c.session_id != "sess-123" || c.server_name != "py-mcp-http" {{ << 21 }}
    if !c.ping() {{ << 22 }}
    tools := c.list_tools() >>>
    if tools.len != 1 || tools[0].name != "echo" {{ << 23 }}
    tres := c.call_tool("echo", "{{\\"msg\\":\\"xu\\"}}") >>>
    if tres.content.len != 1 || tres.content[0].text != "echo:xu" {{ << 24 }}
    res_list := c.list_resources() >>>
    if res_list.len != 1 || res_list[0].uri != "doc://readme" {{ << 25 }}
    rtext := c.read_resource("doc://readme") >>>
    if rtext != "HTTP resource text" {{ << 26 }}
    plist := c.list_prompts() >>>
    if plist.len != 1 || plist[0].name != "summarize" {{ << 27 }}
    ptext := c.get_prompt("summarize", "{{\\"topic\\":\\"AI\\"}}") >>>
    if ptext != "Topic: AI" {{ << 28 }}
    println("MCP_HTTP_OK")
    << 0
}}

fn main() int {{
    rc := run() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
    << rc
}}
"""
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                srv_exe = build_xu(work, "mcp_stdio_srv", STDIO_MCP_SRV_SRC)
                requests = [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "py", "version": "1"}}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                    {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
                    {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "add", "arguments": {"a": 19, "b": 23}}},
                    {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "missing", "arguments": {}}},
                    {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
                    {"jsonrpc": "2.0", "id": 7, "method": "resources/read", "params": {"uri": "memo://hello"}},
                    {"jsonrpc": "2.0", "id": 8, "method": "prompts/list"},
                    {"jsonrpc": "2.0", "id": 9, "method": "prompts/get", "params": {"name": "greet", "arguments": {"name": "Xu"}}},
                ]
                stdin_text = "".join(json.dumps(r) + "\n" for r in requests)
                stdio_res = subprocess.run([str(srv_exe)], input=stdin_text, env=xu_env(work), text=True, capture_output=True)
                self.assertEqual(stdio_res.returncode, 0, stdio_res.stderr)
                lines = [json.loads(line) for line in stdio_res.stdout.splitlines() if line.strip()]
                by_id = {obj["id"]: obj for obj in lines}
                self.assertEqual(len(lines), 9)
                self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-03-26")
                self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "xu-test-mcp")
                self.assertEqual(by_id[2]["result"], {})
                self.assertEqual(by_id[3]["result"]["tools"][0]["name"], "add")
                self.assertEqual(by_id[4]["result"]["content"][0]["text"], "42")
                self.assertEqual(by_id[5]["error"]["code"], -32602)
                self.assertEqual(by_id[6]["result"]["resources"][0]["uri"], "memo://hello")
                self.assertEqual(by_id[7]["result"]["contents"][0]["text"], "hello from mcp resource:memo://hello")
                self.assertEqual(by_id[8]["result"]["prompts"][0]["name"], "greet")
                self.assertEqual(by_id[9]["result"]["messages"][0]["content"]["text"], "Hello Xu")

                cli_exe = build_xu(work, "mcp_http_cli", client_src)
                cli_res = subprocess.run([str(cli_exe)], env=xu_env(work), text=True, capture_output=True)
                self.assertEqual(cli_res.returncode, 0, cli_res.stdout + cli_res.stderr)
                self.assertIn("MCP_HTTP_OK", cli_res.stdout)
        finally:
            http_srv.shutdown()
            http_srv.server_close()

    def test_mcp_stdio_client_http_server_and_agent(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
        oai_port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        http_mcp_proc = None
        try:
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                stdio_exe = build_xu(work, "mcp_stdio_srv", STDIO_MCP_SRV_SRC)
                http_exe = build_xu(work, "mcp_http_srv", HTTP_MCP_SRV_SRC)

                http_mcp_proc = subprocess.Popen(
                    [str(http_exe)], env=xu_env(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                mcp_http_url = None
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    line = http_mcp_proc.stdout.readline()
                    if line and line.strip().startswith("serving:http://"):
                        mcp_http_url = line.strip().split("serving:", 1)[1]
                        break
                self.assertIsNotNone(mcp_http_url)

                oai_url = f"http://127.0.0.1:{oai_port}"
                driver_src = f"""@main
#encoding/json
#errors
#strings
openai #"github.com/xulang-org/packages/openai"
mcp #"github.com/xulang-org/packages/mcp"
agent #"github.com/xulang-org/packages/agent"

fn local_mul(args_json &string) string {{
    doc := json.parse(args_json)
    if !doc.ok() || !doc.is_dict(doc.root()) {{ << "err" }}
    root := doc.root()
    an := doc.get(root, "a") >>> (_) {{ << "err" }}
    bn := doc.get(root, "b") >>> (_) {{ << "err" }}
    av := doc.as_int(an) >>> (_) {{ << "err" }}
    bv := doc.as_int(bn) >>> (_) {{ << "err" }}
    << str(av * bv)
}}

fn run() int {{
    // 1. Xu MCP StdioClient -> Xu MCP Stdio Server
    $sc := mcp.connect_stdio("{stdio_exe}", []string{{}}) >>>
    _ := sc.initialize() >>>
    if sc.server_name != "xu-test-mcp" || !sc.ping() {{ << 31 }}
    stools := sc.list_tools() >>>
    if stools.len != 1 || stools[0].name != "add" {{ << 32 }}
    stres := sc.call_tool("add", "{{\\"a\\":10,\\"b\\":32}}") >>>
    if stres.content.len != 1 || stres.content[0].text != "42" {{ << 33 }}
    sres := sc.read_resource("memo://hello") >>>
    if sres != "hello from mcp resource:memo://hello" {{ << 34 }}
    sprompt := sc.get_prompt("greet", "{{\\"name\\":\\"Agent\\"}}") >>>
    if sprompt != "Hello Agent" {{ << 35 }}

    // 2. Xu MCP HTTP Client -> Xu MCP HTTP Server (serve_http_server)
    $hc := mcp.new_http_client("{mcp_http_url}")
    _ := hc.initialize() >>>
    if !strings.has_prefix(&hc.session_id, "xu-mcp-") || hc.server_name != "xu-http-mcp" {{ << 36 }}
    if !hc.ping() {{ << 37 }}
    htools := hc.list_tools() >>>
    if htools.len != 1 || htools[0].name != "http_mul" {{ << 38 }}
    htres := hc.call_tool("http_mul", "{{\\"a\\":4,\\"b\\":9}}") >>>
    if htres.content.len != 1 || htres.content[0].text != "36" {{ << 39 }}
    if hc.read_resource("mem://http") >>> != "xu-http-res:mem://http" {{ << 40 }}
    if !strings.has_prefix(&hc.get_prompt("ask", "{{\\"k\\":1}}") >>>, "xu-http-prompt:") {{ << 41 }}

    // 3. Agent local tool loop
    $oc1 := openai.new_client("{oai_url}", "test-secret")
    oc1.add_header("X-Agent-Test", "1")
    oc1.add_header("X-Agent-Tool", "local_mul")
    oc1.add_header("X-Agent-Args", "{{\\"a\\":6,\\"b\\":7}}")
    $a1 := agent.new_agent(oc1, "xu-7b")
    a1.set_system_prompt("Be concise.")
    a1.add_tool("local_mul", "Multiply ints", "{{\\"type\\":\\"object\\"}}", local_mul)
    r1 := a1.run("6*7?") >>>
    if r1.turns != 2 || r1.steps.len != 1 || r1.content != "FINAL:42" || r1.total_tokens != 20 {{
        << 42
    }}

    // 4. Agent + HTTP MCP loop
    $oc2 := openai.new_client("{oai_url}", "test-secret")
    oc2.add_header("X-Agent-Test", "1")
    oc2.add_header("X-Agent-Tool", "http_mul")
    oc2.add_header("X-Agent-Args", "{{\\"a\\":8,\\"b\\":9}}")
    a2 := agent.new_agent(oc2, "xu-7b")
    r2 := a2.run_with_mcp("8*9?", $&hc) >>>
    if r2.turns != 2 || r2.steps.len != 1 || r2.content != "FINAL:72" || r2.steps[0].tool_name != "http_mul" {{
        << 43
    }}

    // 5. Agent + Stdio MCP loop + clean close
    $oc3 := openai.new_client("{oai_url}", "test-secret")
    oc3.add_header("X-Agent-Test", "1")
    oc3.add_header("X-Agent-Tool", "add")
    oc3.add_header("X-Agent-Args", "{{\\"a\\":100,\\"b\\":23}}")
    a3 := agent.new_agent(oc3, "xu-7b")
    r3 := a3.run_with_stdio_mcp("100+23?", $&sc) >>>
    if r3.turns != 2 || r3.steps.len != 1 || r3.content != "FINAL:123" || r3.steps[0].output != "123" {{
        << 44
    }}
    // 6. v1.6.0: MCP custom headers + SSE Accept + DELETE close_session
    hc.add_header("X-MCP-Client", "xu-1.6")
    hc.set_bearer_token("tok-123")
    if !hc.ping() {{ << 45 }}
    _ := hc.close_session() >>>
    if hc.session_id.len != 0 {{ << 46 }}

    // 7. v1.6.0: Agent tool_choice + response_format + error recovery + Session
    $hc_err := mcp.new_http_client("{mcp_http_url}")
    _ := hc_err.initialize() >>>
    $oc4 := openai.new_client("{oai_url}", "test-secret")
    oc4.add_header("X-Agent-Test", "1")
    oc4.add_header("X-Agent-Tool", "http_mul")
    oc4.add_header("X-Agent-Args", "{{\\"a\\":\\"bad\\",\\"b\\":2}}")
    oc4.add_header("X-Agent-Expect-Choice", "required")
    oc4.add_header("X-Agent-Expect-Format", "json_object")
    $a4 := agent.new_agent(oc4, "xu-7b")
    a4.set_tool_choice("required")
    a4.set_response_format("{{\\"type\\":\\"json_object\\"}}")
    r4 := a4.run_with_mcp("bad args", $&hc_err) >>>
    if r4.steps.len != 1 || !r4.steps[0].is_error || r4.content != "FINAL:bad a" {{
        << 47
    }}
    _ := hc_err.close_session() >>>

    $oc5 := openai.new_client("{oai_url}", "test-secret")
    oc5.add_header("X-Agent-Test", "1")
    oc5.add_header("X-Agent-Session", "1")
    $a5 := agent.new_agent(oc5, "xu-7b")
    a5.set_system_prompt("Stay terse.")
    $sess := agent.new_session(a5)
    sr1 := sess.send("first") >>>
    sr2 := sess.send("second") >>>
    if sr1.content != "TURN1:total=2" || sr2.content != "TURN2:total=4" || sess.history().len != 5 {{
        << 48
    }}
    sess.clear()
    if sess.history().len != 1 {{ << 49 }}

    _ := sc.close() >>>
    println("MCP_STDIO_HTTP_AGENT_OK")
    << 0
}}

fn main() int {{
    << run() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
                driver_exe = build_xu(work, "mcp_agent_driver", driver_src)
                res = subprocess.run([str(driver_exe)], env=xu_env(work), text=True, capture_output=True, timeout=30)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertIn("MCP_STDIO_HTTP_AGENT_OK", res.stdout)

                # Verify HTTP MCP server responds with text/event-stream when Accept: text/event-stream is sent
                import urllib.request
                sse_req = urllib.request.Request(
                    mcp_http_url,
                    data=json.dumps({"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}}).encode(),
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    method="POST",
                )
                with urllib.request.urlopen(sse_req, timeout=5) as resp:
                    self.assertTrue(resp.headers.get("Content-Type", "").startswith("text/event-stream"))
                    sse_body = resp.read().decode()
                    self.assertIn("data: ", sse_body)
                    self.assertIn("http_mul", sse_body)
        finally:
            if http_mcp_proc is not None:
                http_mcp_proc.terminate()
                try:
                    http_mcp_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    http_mcp_proc.kill()
                    http_mcp_proc.wait(timeout=3)
            srv.shutdown()
            srv.server_close()

    def test_live_agent_mcp_with_llm_engine(self):
        import sys
        scripts_dir = str(REPO / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import test_llm_engine as tle

        llm_src = REPO / "examples/llm_engine/main.xu"
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            llm_exe = work / "llm_engine"
            bres = subprocess.run(
                [str(XU), "build", str(llm_src), "-o", str(llm_exe)],
                cwd=REPO,
                env=xu_env(work),
                capture_output=True,
                text=True,
            )
            self.assertEqual(bres.returncode, 0, bres.stdout + bres.stderr)
            if os.uname().sysname == "Darwin":
                subprocess.run(["codesign", "--force", "--sign", "-", str(llm_exe)], check=True, capture_output=True)

            # Build a deterministic mini Q4_0 GGUF:
            # Turn 1 (no "42" in context, last token = '\n'): emits <tool_call>{"name":"http_mul","arguments":{"a":6,"b":7}}</tool_call><|im_end|>
            # Turn 2 ("42" in context from <tool_response>, last token = '\n'): attention carries signal on residual dim 32, emits FINAL:42<|im_end|>
            cfg = {
                "dim": 64,
                "hidden_dim": 128,
                "n_layers": 1,
                "n_heads": 4,
                "n_kv_heads": 2,
                "vocab_size": 512,
                "max_seq_len": 128,
                "rope_theta": 10000,
            }
            base_tensors = tle.generate_weight_dict(cfg)
            zero_tensors = {k: (list(s), [tle.f32(0.0)] * len(v)) for k, (s, v) in base_tensors.items()}
            dim = cfg["dim"]
            vocab = cfg["vocab_size"]
            zero_tensors["layers.0.input_layernorm.weight"] = ([dim], [tle.f32(1.0)] * dim)
            zero_tensors["norm.weight"] = ([dim], [tle.f32(1.0)] * dim)

            emb = [tle.f32(0.0)] * (vocab * dim)
            head = [tle.f32(0.0)] * (vocab * dim)
            # Every token activates dim 0 slightly so RMSNorm is well-defined even without "42"
            for tid in range(vocab):
                emb[tid * dim + 0] = tle.f32(1.0)
            # Token identities on block 0 [1..5]
            emb[7 * dim + 1] = tle.f32(1.0)    # '\n' (end of <|im_start|>assistant\n)
            emb[10 * dim + 2] = tle.f32(1.0)   # '<tool_call>'
            emb[11 * dim + 3] = tle.f32(1.0)   # '{"name":"http_mul","arguments":{"a":6,"b":7}}'
            emb[12 * dim + 4] = tle.f32(1.0)   # '</tool_call>'
            emb[13 * dim + 5] = tle.f32(1.0)   # 'FINAL:42'
            # Token 14 ("42") activates block 1 dim 32 -> propagated uniformly via V/O into residual dim 32
            emb[14 * dim + 32] = tle.f32(1.0)

            n_kv = cfg["n_kv_heads"]
            hd = dim // cfg["n_heads"]
            kv_dim = n_kv * hd
            wv = [tle.f32(0.0)] * (kv_dim * dim)
            wv[0 * dim + 32] = tle.f32(7.0)  # kv_head 0 dim 0 reads residual dim 32
            wo = [tle.f32(0.0)] * (dim * dim)
            wo[32 * dim + 0] = tle.f32(7.0)  # writes attn_head 0 dim 0 back to residual dim 32
            zero_tensors["layers.0.self_attn.v_proj.weight"] = ([kv_dim, dim], wv)
            zero_tensors["layers.0.self_attn.o_proj.weight"] = ([dim, dim], wo)

            # Logits: '\n' -> '<tool_call>' (unless dim 32 is active, in which case 'FINAL:42' wins)
            head[10 * dim + 1] = tle.f32(4.0)
            head[10 * dim + 32] = tle.f32(-7.0)
            head[11 * dim + 2] = tle.f32(7.0)
            head[12 * dim + 3] = tle.f32(7.0)
            head[13 * dim + 1] = tle.f32(1.0)
            head[13 * dim + 32] = tle.f32(5.0)
            head[2 * dim + 4] = tle.f32(7.0)
            head[2 * dim + 5] = tle.f32(7.0)
            head[2 * dim + 32] = tle.f32(5.0)

            zero_tensors["embed_tokens.weight"] = ([vocab, dim], emb)
            zero_tensors["lm_head.weight"] = ([vocab, dim], head)

            vocab_pieces = [f"tok_{i}" for i in range(vocab)]
            vocab_pieces[0] = "<|endoftext|>"
            vocab_pieces[1] = "<|im_start|>"
            vocab_pieces[2] = "<|im_end|>"
            vocab_pieces[3] = "user"
            vocab_pieces[4] = "assistant"
            vocab_pieces[5] = "system"
            vocab_pieces[7] = "\n"
            vocab_pieces[10] = "<tool_call>"
            vocab_pieces[11] = "{\"name\":\"http_mul\",\"arguments\":{\"a\":6,\"b\":7}}"
            vocab_pieces[12] = "</tool_call>"
            vocab_pieces[13] = "FINAL:42"
            vocab_pieces[14] = "42"

            gguf_path = work / "live_agent.gguf"
            tle.write_gguf_q4_0(gguf_path, cfg, zero_tensors, vocab_pieces=vocab_pieces)

            http_mcp_srv_exe = build_xu(work, "live_http_mcp_srv", HTTP_MCP_SRV_SRC)
            mcp_proc = subprocess.Popen(
                [str(http_mcp_srv_exe)],
                env=xu_env(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            llm_proc = subprocess.Popen(
                [str(llm_exe), "serve", str(gguf_path), "127.0.0.1:0"],
                env=xu_env(work),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                mcp_url = None
                llm_url = None
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline and (mcp_url is None or llm_url is None):
                    if mcp_url is None:
                        line = mcp_proc.stdout.readline()
                        if line and line.strip().startswith("serving:http://"):
                            mcp_url = line.strip().split("serving:", 1)[1]
                    if llm_url is None:
                        line = llm_proc.stdout.readline()
                        if line and line.strip().startswith("serving:http://"):
                            llm_url = line.strip().split("serving:", 1)[1]
                self.assertIsNotNone(mcp_url)
                self.assertIsNotNone(llm_url)

                live_driver_src = f"""@main
#errors
openai #"github.com/xulang-org/packages/openai"
mcp #"github.com/xulang-org/packages/mcp"
agent #"github.com/xulang-org/packages/agent"

fn run() int {{
    $hc := mcp.new_http_client("{mcp_url}")
    _ := hc.initialize() >>>
    oc := openai.new_client("{llm_url}", "")
    a := agent.new_agent(oc, "mini-qwen")
    r := a.run_with_mcp("6*7?", $&hc) >>>
    if r.turns != 2 || r.steps.len != 1 || r.steps[0].tool_name != "http_mul" || r.steps[0].output != "42" || r.content != "FINAL:42" {{
        eprintln("unexpected live result: turns=" + str(r.turns) + " steps=" + str(r.steps.len) + " content=" + r.content)
        << 61
    }}
    _ := hc.close_session() >>>
    println("LIVE_LLM_AGENT_MCP_OK")
    << 0
}}

fn main() int {{
    << run() >>> (err) {{
        eprintln(err.msg())
        << 1
    }}
}}
"""
                live_exe = build_xu(work, "live_agent_driver", live_driver_src)
                lres = subprocess.run([str(live_exe)], env=xu_env(work), text=True, capture_output=True, timeout=30)
                self.assertEqual(lres.returncode, 0, lres.stdout + lres.stderr)
                self.assertIn("LIVE_LLM_AGENT_MCP_OK", lres.stdout)
            finally:
                for p in (mcp_proc, llm_proc):
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
