import http.server
import json
import socket
import socketserver
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.test_v116_batch1_hardening import build_xu_no_codesign, xu_env


class MockRedisState:
    def __init__(self):
        self.lock = threading.Lock()
        self.kv = {}      # key -> (str_val, expire_at_or_None)
        self.hashes = {}  # key -> {field: int_val}
        self.lists = {}   # key -> [str_val, ...]
        self.expires = {} # key -> expire_at_or_None
        self.cmds = []

    def _prune(self, key: str):
        now = time.monotonic()
        if key in self.expires and now >= self.expires[key]:
            self.kv.pop(key, None)
            self.hashes.pop(key, None)
            self.lists.pop(key, None)
            del self.expires[key]
        if key in self.kv:
            _, exp = self.kv[key]
            if exp is not None and now >= exp:
                del self.kv[key]


def _read_exact(sock: socket.socket, buf: bytearray, n: int) -> bytes:
    while len(buf) < n:
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError("unexpected EOF")
        buf.extend(chunk)
    out = bytes(buf[:n])
    del buf[:n]
    return out


def _read_line(sock: socket.socket, buf: bytearray) -> bytes:
    while True:
        idx = buf.find(b"\r\n")
        if idx >= 0:
            out = bytes(buf[:idx])
            del buf[: idx + 2]
            return out
        chunk = sock.recv(4096)
        if not chunk:
            raise EOFError("EOF waiting line")
        buf.extend(chunk)


def _read_resp_cmd(sock: socket.socket, buf: bytearray):
    line = _read_line(sock, buf)
    if not line:
        return None
    if line.startswith(b"*"):
        n = int(line[1:])
        parts = []
        for _ in range(n):
            hdr = _read_line(sock, buf)
            assert hdr.startswith(b"$"), hdr
            slen = int(hdr[1:])
            data = _read_exact(sock, buf, slen)
            crlf = _read_exact(sock, buf, 2)
            assert crlf == b"\r\n"
            parts.append(data.decode("utf-8", errors="replace"))
        return parts
    return line.decode("utf-8", errors="replace").split()


def _enc_bulk(s: str) -> bytes:
    b = s.encode("utf-8")
    return f"${len(b)}\r\n".encode("ascii") + b + b"\r\n"


def start_mock_redis_server():
    st = MockRedisState()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    port = srv.getsockname()[1]
    stop_evt = threading.Event()

    def handle_conn(conn: socket.socket):
        buf = bytearray()
        try:
            while not stop_evt.is_set():
                try:
                    parts = _read_resp_cmd(conn, buf)
                except EOFError:
                    break
                if not parts:
                    break
                cmd = parts[0].upper()
                with st.lock:
                    st.cmds.append(cmd)
                    if cmd == "PING":
                        conn.sendall(b"+PONG\r\n")
                    elif cmd == "AUTH":
                        conn.sendall(b"+OK\r\n")
                    elif cmd == "QUIT":
                        conn.sendall(b"+OK\r\n")
                        break
                    elif cmd == "GET" and len(parts) >= 2:
                        k = parts[1]
                        st._prune(k)
                        if k in st.kv:
                            conn.sendall(_enc_bulk(st.kv[k][0]))
                        else:
                            conn.sendall(b"$-1\r\n")
                    elif cmd == "SETEX" and len(parts) >= 4:
                        k, ttl, val = parts[1], max(1, int(parts[2])), parts[3]
                        st.kv[k] = (val, time.monotonic() + ttl)
                        conn.sendall(b"+OK\r\n")
                    elif cmd == "DEL":
                        removed = 0
                        for k in parts[1:]:
                            if k in st.kv or k in st.hashes or k in st.lists:
                                removed += 1
                            st.kv.pop(k, None)
                            st.hashes.pop(k, None)
                            st.lists.pop(k, None)
                            st.expires.pop(k, None)
                        conn.sendall(f":{removed}\r\n".encode("ascii"))
                    elif cmd == "INCR" and len(parts) >= 2:
                        k = parts[1]
                        st._prune(k)
                        cur = int(st.kv[k][0]) if k in st.kv else 0
                        cur += 1
                        exp = st.kv[k][1] if k in st.kv else None
                        st.kv[k] = (str(cur), exp)
                        conn.sendall(f":{cur}\r\n".encode("ascii"))
                    elif cmd == "DECR" and len(parts) >= 2:
                        k = parts[1]
                        st._prune(k)
                        cur = int(st.kv[k][0]) if k in st.kv else 0
                        cur -= 1
                        exp = st.kv[k][1] if k in st.kv else None
                        st.kv[k] = (str(cur), exp)
                        conn.sendall(f":{cur}\r\n".encode("ascii"))
                    elif cmd == "EXPIRE" and len(parts) >= 3:
                        k, ttl = parts[1], int(parts[2])
                        exp = time.monotonic() + ttl
                        st.expires[k] = exp
                        if k in st.kv:
                            st.kv[k] = (st.kv[k][0], exp)
                        conn.sendall(b":1\r\n")
                    elif cmd == "HINCRBY" and len(parts) >= 4:
                        k, f, d = parts[1], parts[2], int(parts[3])
                        st._prune(k)
                        h = st.hashes.setdefault(k, {})
                        h[f] = h.get(f, 0) + d
                        conn.sendall(f":{h[f]}\r\n".encode("ascii"))
                    elif cmd == "HGETALL" and len(parts) >= 2:
                        k = parts[1]
                        st._prune(k)
                        h = st.hashes.get(k, {})
                        items = []
                        for f, v in h.items():
                            items.append(f)
                            items.append(str(v))
                        out = bytearray(f"*{len(items)}\r\n".encode("ascii"))
                        for it in items:
                            out.extend(_enc_bulk(it))
                        conn.sendall(bytes(out))
                    elif cmd == "LPUSH" and len(parts) >= 3:
                        k, val = parts[1], parts[2]
                        st._prune(k)
                        lst = st.lists.setdefault(k, [])
                        lst.insert(0, val)
                        conn.sendall(f":{len(lst)}\r\n".encode("ascii"))
                    elif cmd == "LTRIM" and len(parts) >= 4:
                        k, s_i, e_i = parts[1], int(parts[2]), int(parts[3])
                        st._prune(k)
                        lst = st.lists.get(k, [])
                        if e_i < 0:
                            e_i = len(lst) + e_i
                        st.lists[k] = lst[max(0, s_i): e_i + 1]
                        conn.sendall(b"+OK\r\n")
                    elif cmd == "LRANGE" and len(parts) >= 4:
                        k, s_i, e_i = parts[1], int(parts[2]), int(parts[3])
                        st._prune(k)
                        lst = st.lists.get(k, [])
                        if e_i < 0:
                            e_i = len(lst) + e_i
                        sub = lst[max(0, s_i): e_i + 1]
                        out = bytearray(f"*{len(sub)}\r\n".encode("ascii"))
                        for it in sub:
                            out.extend(_enc_bulk(it))
                        conn.sendall(bytes(out))
                    else:
                        conn.sendall(f"-ERR unsupported {cmd}\r\n".encode("ascii"))
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def accept_loop():
        while not stop_evt.is_set():
            try:
                srv.settimeout(0.2)
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=handle_conn, args=(conn,), daemon=True)
            t.start()

    th = threading.Thread(target=accept_loop, daemon=True)
    th.start()

    def stop():
        stop_evt.set()
        try:
            srv.close()
        except OSError:
            pass

    return f"127.0.0.1:{port}", st, stop


class MockGatewayProviderState:
    def __init__(self):
        self.lock = threading.Lock()
        self.openai_reqs = []
        self.anthropic_reqs = []
        self.gemini_reqs = []


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class MockGatewayProviderHandler(http.server.BaseHTTPRequestHandler):
    state: MockGatewayProviderState = None

    def log_message(self, format, *args):
        return

    def _send_json(self, code: int, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        st = self.state

        if self.path == "/v1/chat/completions":
            with st.lock:
                st.openai_reqs.append(body)
            if body.get("model") == "fail-model":
                self._send_json(500, {"error": {"message": "injected primary failure"}})
                return
            self._send_json(
                200,
                {
                    "id": "oai-gw-1",
                    "model": body.get("model", "gpt-4o"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "OPENAI_OK"},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
            return

        if self.path == "/v1/messages":
            with st.lock:
                st.anthropic_reqs.append(body)
            self._send_json(
                200,
                {
                    "id": "anth-gw-1",
                    "model": body.get("model", "claude-3-7-sonnet"),
                    "role": "assistant",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "ANTHROPIC_OK"}],
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                },
            )
            return

        if ":generateContent" in self.path:
            with st.lock:
                st.gemini_reqs.append(body)
            self._send_json(
                200,
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "GEMINI_OK"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 14, "candidatesTokenCount": 6, "totalTokenCount": 20},
                },
            )
            return

        self._send_json(404, {"error": "unknown route"})


XU_BATCH2_GATEWAY_SOURCE = '''@main
#errors
#strings
#time
anthropic #"github.com/xulang-org/packages/anthropic"
gemini #"github.com/xulang-org/packages/gemini"
gateway #"github.com/xulang-org/packages/gateway"
openai #"github.com/xulang-org/packages/openai"

fn make_req(prompt string) openai.ChatRequest {
    msgs := [openai.message("system", "You are concise."), openai.message("user", prompt)]
    << openai.chat_request("auto", ~msgs)
}

fn try_chat_ok(g $&gateway.Gateway, req &openai.ChatRequest) bool {
    _ := g.chat_with_fallback(req) >>> (_) { << false }
    << true
}

fn run_all(base_url string, redis_addr string) int {
    // 1. Shared Circuit Breaker across Replica A & Replica B
    $ga_cb := gateway.new_gateway()
    ga_cb.set_redis_store(redis_addr + "", "cb_cluster:")
    ga_cb.set_circuit_breaker(2, 5000)
    $oai_fail := openai.new_client(base_url + "/v1", "k1")
    oai_fail.set_max_retries(0)
    ga_cb.add_route("primary", ~oai_fail, "fail-model", 1.0, 2.0)
    ga_cb.add_anthropic_route("backup", anthropic.new_client_with_base(base_url + "", "ak1"), "claude-3-7-sonnet", 1.0, 2.0, 1)

    $gb_cb := gateway.new_gateway()
    gb_cb.set_redis_store(redis_addr + "", "cb_cluster:")
    gb_cb.set_circuit_breaker(2, 5000)
    $oai_fail2 := openai.new_client(base_url + "/v1", "k1")
    oai_fail2.set_max_retries(0)
    gb_cb.add_route("primary", ~oai_fail2, "fail-model", 1.0, 2.0)
    gb_cb.add_anthropic_route("backup", anthropic.new_client_with_base(base_url + "", "ak1"), "claude-3-7-sonnet", 1.0, 2.0, 1)

    r_req := make_req("hello")
    res1 := ga_cb.chat_with_fallback(&r_req) >>>
    if res1.content != "ANTHROPIC_OK" { >>> errors.new("cb fallback 1 failed") }
    res2 := ga_cb.chat_with_fallback(&r_req) >>>
    if res2.content != "ANTHROPIC_OK" { >>> errors.new("cb fallback 2 failed") }
    // Replica B must observe open circuit from Redis and skip primary completely
    res3 := gb_cb.chat_with_fallback(&r_req) >>>
    if res3.content != "ANTHROPIC_OK" { >>> errors.new("cb replica B fallback failed") }
    ga_cb.reset_circuits()
    ga_cb.close_redis()
    gb_cb.close_redis()

    // 2. Shared Sliding-Window RPM Limit across Replica A & Replica B
    $ga_rpm := gateway.new_gateway()
    ga_rpm.set_redis_store(redis_addr + "", "rpm_cluster:")
    ga_rpm.set_rate_limit(3)
    ga_rpm.add_route("ok", openai.new_client(base_url + "/v1", "k1"), "gpt-4o", 1.0, 2.0)

    $gb_rpm := gateway.new_gateway()
    gb_rpm.set_redis_store(redis_addr + "", "rpm_cluster:")
    gb_rpm.set_rate_limit(3)
    gb_rpm.add_route("ok", openai.new_client(base_url + "/v1", "k1"), "gpt-4o", 1.0, 2.0)

    if !try_chat_ok($&ga_rpm, &r_req) { >>> errors.new("rpm 1 failed") }
    if !try_chat_ok($&ga_rpm, &r_req) { >>> errors.new("rpm 2 failed") }
    if !try_chat_ok($&gb_rpm, &r_req) { >>> errors.new("rpm 3 failed") }
    if try_chat_ok($&gb_rpm, &r_req) { >>> errors.new("shared cluster RPM limit did not block 4th request") }
    ga_rpm.close_redis()
    gb_rpm.close_redis()

    // 3. Cluster-wide stats + Semantic Cache + TTL across Replica A & Replica B
    $ga_sc := gateway.new_gateway()
    ga_sc.set_redis_store(redis_addr + "", "sc_cluster:")
    ga_sc.enable_semantic_cache(0.90, 8)
    ga_sc.set_cache_ttl(1)
    ga_sc.add_route("ok", openai.new_client(base_url + "/v1", "k1"), "gpt-4o", 1.0, 2.0)

    $gb_sc := gateway.new_gateway()
    gb_sc.set_redis_store(redis_addr + "", "sc_cluster:")
    gb_sc.enable_semantic_cache(0.90, 8)
    gb_sc.set_cache_ttl(1)
    gb_sc.add_route("ok", openai.new_client(base_url + "/v1", "k1"), "gpt-4o", 1.0, 2.0)

    emb1 := [f32(1.0), f32(0.0), f32(0.0)]
    emb2 := [f32(0.995), f32(0.05), f32(0.0)]
    sc1 := ga_sc.chat_cached(&r_req, &emb1) >>>
    if sc1.id == "cache_hit" || sc1.content != "OPENAI_OK" { >>> errors.new("replica A initial miss failed") }
    sc2 := gb_sc.chat_cached(&r_req, &emb2) >>>
    if sc2.id != "cache_hit" || sc2.content != "OPENAI_OK" { >>> errors.new("replica B semantic cache hit failed") }

    st := gb_sc.get_stats()
    if st.total_requests != 2 || st.cache_hits != 1 || st.total_prompt_tokens != 10 || st.total_completion_tokens != 5 {
        >>> errors.new("cluster stats mismatch: req=" + str(st.total_requests) + " hits=" + str(st.cache_hits) + " pt=" + str(st.total_prompt_tokens))
    }
    prom := ga_sc.format_metrics_prometheus()
    if !strings.contains(&prom, "xu_gateway_requests_total 2") || !strings.contains(&prom, "xu_gateway_cache_hits_total 1") {
        >>> errors.new("prometheus metrics mismatch: " + prom)
    }

    // Wait > 1s for TTL expiry -> Replica B must miss and hit OpenAI again
    time.sleep(1250 * time.millisecond())
    sc3 := gb_sc.chat_cached(&r_req, &emb1) >>>
    if sc3.id == "cache_hit" { >>> errors.new("expected expired cache entry to miss after TTL") }
    ga_sc.close_redis()
    gb_sc.close_redis()

    // 4. Vendor Passthrough: Anthropic & Gemini (route-level + req-level / chat_with_passthrough)
    $g_pt := gateway.new_gateway()
    g_pt.add_anthropic_route("anth", anthropic.new_client_with_base(base_url + "", "ak2"), "claude-3-7-sonnet", 1.0, 2.0, 1)
    g_pt.add_gemini_route("gem", gemini.new_client_with_base(base_url + "", "gk2"), "gemini-2.5-flash", 1.0, 2.0, 1)

    $pt_anth := gateway.empty_passthrough()
    pt_anth.anthropic_cache_system = true
    pt_anth.anthropic_cache_control = "ephemeral"
    pt_anth.anthropic_thinking_budget = 2048
    pt_anth.anthropic_extra_body_json = "{\\"metadata\\":{\\"user_id\\":\\"u-42\\"}}"
    if !g_pt.set_route_passthrough("anth", pt_anth) { >>> errors.new("set_route_passthrough anth failed") }

    res_a := g_pt.chat_with_fallback(&r_req) >>>
    if res_a.content != "ANTHROPIC_OK" { >>> errors.new("anth passthrough resp failed") }

    // Disable anth route so gem route runs with request-level extra_body_json passthrough
    $g_gem := gateway.new_gateway()
    g_gem.add_gemini_route("gem", gemini.new_client_with_base(base_url + "", "gk2"), "gemini-2.5-flash", 1.0, 2.0, 1)
    $gem_req := make_req("gemini prompt")
    gem_req.extra_body_json = "{\\"gemini\\":{\\"safetySettings\\":[{\\"category\\":\\"HARM_CATEGORY_HATE_SPEECH\\",\\"threshold\\":\\"BLOCK_ONLY_HIGH\\"}],\\"thinkingBudget\\":512,\\"extra_body\\":{\\"cachedContent\\":\\"cachedContents/abc\\"}}}"
    res_g := g_gem.chat_with_fallback(&gem_req) >>>
    if res_g.content != "GEMINI_OK" { >>> errors.new("gemini passthrough resp failed") }

    << 0
}

fn main() int {
    << 0
}
'''


class Batch2GatewayTests(unittest.TestCase):
    def test_redis_shared_state_and_vendor_passthrough(self):
        redis_addr, redis_st, stop_redis = start_mock_redis_server()
        prov_st = MockGatewayProviderState()
        handler_cls = type("BoundHandler", (MockGatewayProviderHandler,), {"state": prov_st})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                work = Path(td)
                src = (
                    XU_BATCH2_GATEWAY_SOURCE.replace(
                        "fn main() int {\n    << 0\n}",
                        f'fn main() int {{\n    << run_all("http://127.0.0.1:{port}", "{redis_addr}") >>> (err) {{ << 1 }}\n}}',
                    )
                )
                exe = build_xu_no_codesign(work, "batch2_gw", src)
                res = subprocess.run([str(exe)], env=xu_env(work), capture_output=True, text=True, timeout=30)
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

            # 1. Shared CB: Replica A failed 2 times -> opened CB -> Replica B skipped primary (only 2 fail-model hits total!)
            fail_hits = [r for r in prov_st.openai_reqs if r.get("model") == "fail-model"]
            self.assertEqual(len(fail_hits), 2, f"Expected Replica B to skip primary via Redis CB, got {len(fail_hits)}")

            # 2. Verify Anthropic wire JSON from passthrough test (last Anthropic request)
            anth_pt_req = prov_st.anthropic_reqs[-1]
            self.assertIsInstance(anth_pt_req.get("system"), list)
            self.assertEqual(anth_pt_req["system"][0]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(anth_pt_req["messages"][-1]["content"][0]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(anth_pt_req.get("thinking"), {"type": "enabled", "budget_tokens": 2048})
            self.assertEqual(anth_pt_req.get("metadata"), {"user_id": "u-42"})

            # 3. Verify Gemini wire JSON from passthrough test
            gem_pt_req = prov_st.gemini_reqs[-1]
            self.assertEqual(
                gem_pt_req.get("safetySettings"),
                [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"}],
            )
            self.assertEqual(gem_pt_req.get("generationConfig", {}).get("thinkingConfig"), {"thinkingBudget": 512})
            self.assertEqual(gem_pt_req.get("cachedContent"), "cachedContents/abc")

            # 4. Verify Redis commands were exercised
            for expected_cmd in ("INCR", "DECR", "SETEX", "GET", "DEL", "HINCRBY", "HGETALL", "LPUSH", "LRANGE"):
                self.assertIn(expected_cmd, redis_st.cmds)
        finally:
            srv.shutdown()
            srv.server_close()
            stop_redis()


if __name__ == "__main__":
    unittest.main()
