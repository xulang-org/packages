"""End-to-end tests for github.com/xulang-org/packages/{openai,mcp}."""
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
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
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

            has_tools = bool(req.get("tools"))
            if has_tools:
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
            # Exercise SSE wrapping path on Streamable HTTP!
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
    ev_ch <- openai.StreamDelta{{content: "", finish_reason: "", tool_calls: []openai.ToolCall{{}}, has_usage: false, usage: resp2.usage, done: false}}
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
        stdio_src = """@main
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
    $srv := mcp.new_server("xu-test-mcp", "1.4.0")
    srv.add_tool("add", "Add a and b", "{\\"type\\":\\"object\\"}", add_handler)
    srv.add_resource("memo://hello", "hello", "Greeting resource", "text/plain", res_handler)
    srv.add_prompt("greet", "Greet by name", "[{\\"name\\":\\"name\\"}]", prompt_handler)
    << srv.serve_stdio() >>> (err) {
        eprintln(err.msg())
        << 1
    }
}
"""
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
                srv_exe = build_xu(work, "mcp_stdio_srv", stdio_src)
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
                self.assertEqual(len(lines), 9)  # notification skipped
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


if __name__ == "__main__":
    unittest.main()
