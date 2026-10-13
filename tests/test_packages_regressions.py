"""Run package behavior regressions with the product compiler and local peers.

No external services or credentials are required. Set XU to select a compiler.
"""
import contextlib
import json
import os
import signal
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import threading
import unittest

PACKAGES = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("XU_ROOT", PACKAGES.parent / "xu")).expanduser().resolve()
XU = Path(os.environ.get("XU", str(REPO / "bin/xu"))).resolve()


def compiler_env(work):
    env = dict(os.environ, XUPATH="", XU_CACHE=str(work / "cache"))
    for key in ("XUOS", "XUARCH", "XU_HEAP_STATS"):
        env.pop(key, None)
    return env


def exact(peer, size):
    data = b""
    while len(data) < size:
        chunk = peer.recv(size - len(data))
        if not chunk:
            raise EOFError("peer closed before complete message")
        data += chunk
    return data


@contextlib.contextmanager
def server(handler):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.2)
    stopped = threading.Event()
    errors = []

    def serve():
        try:
            with listener:
                while not stopped.is_set():
                    try:
                        peer, _ = listener.accept()
                        break
                    except socket.timeout:
                        continue
                else:
                    return
                with peer:
                    peer.settimeout(10)
                    handler(peer)
        except Exception as err:
            errors.append(err)

    endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    failed = False
    try:
        yield endpoint
    except BaseException:
        failed = True
        raise
    finally:
        stopped.set()
        worker.join(12)
        if worker.is_alive():
            raise AssertionError("mock server did not terminate")
        if errors and not failed:
            raise AssertionError(f"mock server failed: {errors!r}")


class PackageRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runpy
        checker = runpy.run_path(str(PACKAGES / "scripts/check_product_compiler.py"))
        checker["main"]()

    def test_channel_payload_layout_can_be_declared_before_struct(self):
        source = '''@main
struct Holder { pub st chan State }
struct State { pub closed int pub idle []int pub borrowed []int }
fn new_state() chan State {
    ch := chan State{cap: 1}
    ch <- State{closed: 0, idle: [11, 12], borrowed: [21]}
    << ch
}
fn main() int {
    st := new_state()
    holder := Holder{st: st}
    received := <-holder.st
    if received.closed != 0 { println("wrong closed=" + str(received.closed))
        << 1 }
    if received.idle.len != 2 || received.idle[0] != 11 || received.idle[1] != 12 { << 2 }
    if received.borrowed.len != 1 || received.borrowed[0] != 21 { << 3 }
    println("forward channel layout preserved")
    << 0
}
'''
        for flags in [("-O0",), ("-O1",), ("--no-regcache",)]:
            with self.subTest(flags=flags):
                self.assertIn("preserved", self.run_xu(source, [], build_flags=flags))

    @contextlib.contextmanager
    def http_pool_peer(self, responses):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(0.2)
        stopped = threading.Event()
        errors = []
        stats = {"connections": 0, "requests": []}

        lock = threading.Lock()
        clients = []

        def handle(peer):
            try:
                with peer:
                    peer.settimeout(0.2)
                    data = b""
                    while not stopped.is_set():
                        try:
                            chunk = peer.recv(4096)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        data += chunk
                        if b"\r\n\r\n" not in data:
                            continue
                        head, body = data.split(b"\r\n\r\n", 1)
                        length = next((int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")), 0)
                        if len(body) < length:
                            continue
                        with lock:
                            index = len(stats["requests"])
                            if index >= len(responses):
                                raise AssertionError(f"unexpected request {index + 1}: {head!r}")
                            stats["requests"].append((head, body[:length]))
                            response, close = responses[index]
                        data = body[length:]
                        peer.sendall(response)
                        if close:
                            break
            except Exception as err:
                errors.append(err)

        def serve():
            try:
                with listener:
                    while not stopped.is_set():
                        try:
                            peer, _ = listener.accept()
                        except socket.timeout:
                            continue
                        with lock:
                            stats["connections"] += 1
                        client = threading.Thread(target=handle, args=(peer,), daemon=True)
                        clients.append(client)
                        client.start()
            except Exception as err:
                errors.append(err)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        failed = False
        try:
            yield f"127.0.0.1:{listener.getsockname()[1]}", stats
        except BaseException:
            failed = True
            raise
        finally:
            stopped.set()
            worker.join(3)
            for client in clients:
                client.join(1)
            if worker.is_alive() or any(client.is_alive() for client in clients):
                raise AssertionError("pool peer did not terminate")
            if errors and not failed:
                raise AssertionError(f"pool peer failed: {errors!r}")

    def test_milvus_pool_reuses_business_connection(self):
        body = b'{"code":0,"data":[]}'
        response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        source = '''@main
#milvus
#os
fn main() int {
    p := milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)
    for i in 0:2 {
        result := p.search("demo", ["1"], 1) or (err) { println(err.msg())
            << 1 }
        if result.hits.len != 0 { << 2 }
    }
    p.close()
    << 0
}
'''
        with self.http_pool_peer([(response, False)] * 2) as (endpoint, stats):
            self.run_xu(source, ["milvus"], endpoint)
        self.assertEqual(len(stats["requests"]), 2)
        self.assertEqual(stats["connections"], 1, "business methods bypassed the pool")

    def test_qdrant_pool_reuses_business_connection(self):
        body = b'{"result":[]}'
        response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        source = '''@main
#qdrant
#os
fn main() int {
    p := qdrant.open_pool(os.getenv("AUDIT_ENDPOINT"), "secret", 1)
    for i in 0:2 {
        result := p.search("demo", ["1"], 1) or (err) { println(err.msg())
            << 1 }
        if result.len != 0 { << 2 }
    }
    p.close()
    << 0
}
'''
        with self.http_pool_peer([(response, False)] * 2) as (endpoint, stats):
            self.run_xu(source, ["qdrant"], endpoint)
        self.assertEqual(stats["connections"], 1, "Qdrant business methods bypassed the pool")
        self.assertEqual(len(stats["requests"]), 2)
        self.assertTrue(all(b"api-key: secret" in head for head, _ in stats["requests"]))

    def remaining_http_pool_cases(self):
        return [
            ("consul", 'consul.open_pool(endpoint, "secret", 1)', 'p.get_kv("key")',
             'result != "value"', b"value", b"/v1/kv/key?raw", b"X-Consul-Token: secret"),
            ("elasticsearch", 'elasticsearch.open_pool(endpoint, 1)', 'p.search_match("demo", "title", "term")',
             'result.total != 0', b'{"hits":{"total":0,"hits":[]}}', b"/demo/_search", b"Accept: application/json"),
            ("etcd", 'etcd.open_pool(endpoint, 1)', 'p.get_key("key")',
             'result != "value"', b'{"kvs":[{"value":"dmFsdWU="}]}', b"/v3/kv/range", b"Content-Type: application/json"),
            ("influxdb", 'influxdb.open_pool(endpoint, "secret", "org", "bucket", 1)', 'p.query_flux("flux")',
             'result != "value"', b"value", b"/api/v2/query?org=org", b"Authorization: Token secret"),
            ("nacos", 'nacos.open_pool(endpoint, "namespace", 1)', 'p.get_config("id", "")',
             'result != "value"', b"value", b"/nacos/v1/cs/configs?dataId=id&group=DEFAULT_GROUP", b"Content-Type: application/x-www-form-urlencoded"),
            ("prometheus", 'prometheus.open_pool(endpoint, 1)', 'p.push_metrics("job", "metric 1")',
             '!result', b"", b"/metrics/job/job", b"Content-Type: text/plain"),
            ("qdrant", 'qdrant.open_pool(endpoint, "secret", 1)', 'p.search("demo", ["1"], 1)',
             'result.len != 0', b'{"result":[]}', b"/collections/demo/points/search", b"api-key: secret"),
            ("s3", 's3.open_pool(endpoint, "access", "secret", "us-west-2", 1)', 'p.get_object("bucket", "key")',
             'result != "value"', b"value", b"/bucket/key", b"Authorization: AWS4-HMAC-SHA256"),
        ]

    def test_remaining_http_pools_reuse_business_connections(self):
        for package, constructor, call, invalid, body, path, header in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                source = f'''@main
#{package}
#os
fn main() int {{
    endpoint := os.getenv("AUDIT_ENDPOINT")
    p := {constructor}
    for i in 0:2 {{
        result := {call} or (err) {{ println(err.msg())
            << 1 }}
        if {invalid} {{ << 2 }}
    }}
    p.close()
    << 0
}}
'''
                # Pushgateway preserves its boolean failure API.
                if package == "prometheus":
                    source = source.replace(' or (err) { println(err.msg())\n            << 1 }', '')
                response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                with self.http_pool_peer([(response, False)] * 2) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)
                self.assertEqual(len(stats["requests"]), 2)
                self.assertTrue(all(path in head and header in head and b"Connection: keep-alive" in head for head, _ in stats["requests"]))

    def test_remaining_http_pools_discard_failures_and_nonpersistent_responses(self):
        for package, constructor, call, invalid, body, _, _ in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                failures = [
                    (b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n", False),
                    (b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n", False),
                    (b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\n\r\n" + body, True),
                    (b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nx", True),
                ]
                if package in ("elasticsearch", "qdrant"):
                    failures.append((b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", False))
                successes = [
                    (b"HTTP/1.1 200 OK\r\nConnection: keep-alive, ClOsE\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, True),
                    (b"HTTP/1.1 200 OK\r\n\r\n" + body, True),
                    (good.replace(b"HTTP/1.1", b"HTTP/1.0"), True),
                ]
                source = f'''@main
#{package}
#os
fn check_call(p &{package}.Pool, expected bool) int {{
    result := {call} or (err) {{
        if expected {{ println(err.msg())
            << 1 }}
        << 0
    }}
    if !expected || {invalid} {{ << 2 }}
    << 0
}}
fn main() int {{
    endpoint := os.getenv("AUDIT_ENDPOINT")
    p := {constructor}
    for i in 0:{len(failures)} {{
        if check_call(&p, false) != 0 || check_call(&p, true) != 0 {{ << 3 }}
    }}
    for i in 0:{len(successes) * 2} {{
        if check_call(&p, true) != 0 {{ << 4 }}
    }}
    p.close()
    << 0
}}
'''
                if package == "prometheus":
                    start = source.index('    result :=')
                    end = source.index('    << 0\n}', start)
                    source = source[:start] + f'    result := {call}\n    if result != expected {{ << 2 }}\n' + source[end:]
                responses = []
                for response in failures + successes:
                    responses += [response, (good, False)]
                with self.http_pool_peer(responses) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(len(stats["requests"]), len(responses))
                self.assertEqual(stats["connections"], 1 + len(failures) + len(successes))

    def test_remaining_http_pool_copies_share_close_and_return_borrowed_sockets(self):
        for package, constructor, call, _, _, _, _ in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                source = f'''@main
#{package}
#os
#net
fn main() int {{
    endpoint := os.getenv("AUDIT_ENDPOINT")
    p := {constructor}
    copy := p.dup()
    fd := copy.get_conn()
    if fd < 0 {{ << 1 }}
    p.close()
    copy.close()
    copy.put_conn(fd)
    if p.get_conn() >= 0 || copy.get_conn() >= 0 {{ << 2 }}
    text := "probe"
    written := net.conn_write(fd, &text) or (err) {{ << 0 }}
    if written > 0 {{ << 3 }}
    << 0
}}
'''
                with self.http_pool_peer([]) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertLessEqual(stats["connections"], 1)

    def test_remaining_http_pools_all_business_methods(self):
        search = b'{"hits":{"total":0,"hits":[]}}'
        methods = {
            "consul": [
                ('p.put_kv("key", "value")', False, '!r', 200, b"true", b"PUT /v1/kv/key "),
                ('p.get_kv("key")', True, 'r != "value"', 200, b"value", b"GET /v1/kv/key?raw "),
                ('p.delete_kv("key")', False, '!r', 200, b"true", b"DELETE /v1/kv/key "),
                ('p.register_service("id", "name", "address", 80)', False, '!r', 200, b"true", b"PUT /v1/agent/service/register "),
                ('p.deregister_service("id")', False, '!r', 200, b"true", b"PUT /v1/agent/service/deregister/id "),
                ('p.get_service("name")', True, 'r != "[]"', 200, b"[]", b"GET /v1/catalog/service/name "),
                ('p.ping()', False, '!r', 200, b'"leader"', b"GET /v1/status/leader "),
            ],
            "elasticsearch": [
                ('p.search("demo", "{}")', True, 'r.total != 0', 200, search, b"POST /demo/_search "),
                ('p.search_match("demo", "title", "term")', True, 'r.total != 0', 200, search, b"POST /demo/_search "),
                ('p.get_doc("demo", "id")', True, 'r != "{}"', 200, b"{}", b"GET /demo/_doc/id "),
                ('p.index_doc("demo", "id", "{}")', True, 'r != "{}"', 201, b"{}", b"PUT /demo/_doc/id "),
                ('p.delete_doc("demo", "id")', True, '!r', 200, b"{}", b"DELETE /demo/_doc/id "),
                ('p.ping()', False, '!r', 200, b"{}", b"GET / "),
            ],
            "etcd": [
                ('p.put("key", "value")', True, 'r != 0', 200, b"{}", b"POST /v3/kv/put "),
                ('p.get_key("key")', True, 'r != "value"', 200, b'{"kvs":[{"value":"dmFsdWU="}]}', b"POST /v3/kv/range "),
                ('p.delete("key")', True, 'r != 0', 200, b"{}", b"POST /v3/kv/deleterange "),
                ('p.exists("key")', True, '!r', 200, b'{"kvs":[]}', b"POST /v3/kv/range "),
            ],
            "influxdb": [
                ('p.write_point(point)', True, '!r', 204, b"", b"POST /api/v2/write?org=org&bucket=bucket&precision=ns "),
                ('p.write_line("cpu value=1i")', True, '!r', 204, b"", b"POST /api/v2/write?org=org&bucket=bucket&precision=ns "),
                ('p.write_raw("cpu value=1i")', True, '!r', 204, b"", b"POST /api/v2/write?org=org&bucket=bucket&precision=ns "),
                ('p.query_flux("flux")', True, 'r != "value"', 200, b"value", b"POST /api/v2/query?org=org "),
                ('p.ping()', True, '!r', 204, b"", b"GET /ping "),
            ],
            "nacos": [
                ('p.register_instance("service", "127.0.0.1", 80, "group")', False, '!r', 200, b"ok", b"POST /nacos/v1/ns/instance?"),
                ('p.get_instances("service", "group")', True, 'r != "value"', 200, b"value", b"GET /nacos/v1/ns/instance/list?"),
                ('p.deregister_instance("service", "127.0.0.1", 80, "group")', False, '!r', 200, b"ok", b"DELETE /nacos/v1/ns/instance?"),
                ('p.publish_config("id", "group", "value")', False, '!r', 200, b"ok", b"POST /nacos/v1/cs/configs "),
                ('p.get_config("id", "group")', True, 'r != "value"', 200, b"value", b"GET /nacos/v1/cs/configs?"),
                ('p.remove_config("id", "group")', False, '!r', 200, b"ok", b"DELETE /nacos/v1/cs/configs?"),
            ],
            "prometheus": [('p.push_metrics("job", "metric 1")', False, '!r', 200, b"", b"POST /metrics/job/job ")],
            "qdrant": [
                ('p.create_collection("demo", 1, "Cosine")', False, '!r', 200, b"{}", b"PUT /collections/demo "),
                ('p.collection_exists("demo")', False, '!r', 200, b"{}", b"GET /collections/demo "),
                ('p.delete_collection("demo")', False, '!r', 200, b"{}", b"DELETE /collections/demo "),
                ('p.upsert_point("demo", 1, ["1"], "{}")', False, '!r', 200, b"{}", b"PUT /collections/demo/points?wait=true "),
                ('p.search("demo", ["1"], 1)', True, 'r.len != 0', 200, b'{"result":[]}', b"POST /collections/demo/points/search "),
                ('p.ping()', False, '!r', 200, b"{}", b"GET /healthz "),
            ],
            "s3": [
                ('p.make_bucket("bucket")', True, 'r != 0', 204, b"", b"PUT /bucket "),
                ('p.bucket_exists("bucket")', False, '!r', 200, b"", b"HEAD /bucket "),
                ('p.remove_bucket("bucket")', True, 'r != 0', 204, b"", b"DELETE /bucket "),
                ('p.put_object("bucket", "key", "value", "")', True, 'r != 0', 204, b"", b"PUT /bucket/key "),
                ('p.get_object("bucket", "key")', True, 'r != "value"', 200, b"value", b"GET /bucket/key "),
                ('p.remove_object("bucket", "key")', True, 'r != 0', 204, b"", b"DELETE /bucket/key "),
                ('p.object_exists("bucket", "key")', False, '!r', 200, b"", b"HEAD /bucket/key "),
            ],
        }
        for package, constructor, _, _, _, _, _ in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                statements, responses = [], []
                for i, (call, throwing, invalid, status, body, path) in enumerate(methods[package]):
                    handling = ' or (err) { println(err.msg())\n        << 1 }' if throwing else ''
                    statements.append(f'    r{i} := {call}{handling}\n    if {invalid.replace("r", "r" + str(i))} {{ << 2 }}')
                    length = 999 if path.startswith(b"HEAD ") else len(body)
                    responses.append((f"HTTP/1.1 {status} OK\r\nContent-Length: {length}\r\n\r\n".encode() + body, False))
                setup = '    mut point := influxdb.new_point("cpu")\n    point.add_field_int("value", 1)\n' if package == "influxdb" else ''
                source = f'@main\n#{package}\n#os\nfn main() int {{\n    endpoint := os.getenv("AUDIT_ENDPOINT")\n    p := {constructor}\n' + setup + '\n'.join(statements) + '\n    p.close()\n    << 0\n}\n'
                with self.http_pool_peer(responses) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)
                self.assertEqual(len(stats["requests"]), len(methods[package]))
                for (head, _), (_, _, _, _, _, path) in zip(stats["requests"], methods[package]):
                    self.assertTrue(head.startswith(path), head)

    def test_remaining_http_pool_mutations_discard_boolean_and_throwing_failures(self):
        calls = {
            "consul": 'p.put_kv("key", "value")',
            "elasticsearch": 'p.delete_doc("demo", "id")',
            "etcd": 'p.exists("key")',
            "influxdb": 'p.write_line("cpu value=1i")',
            "nacos": 'p.publish_config("id", "group", "value")',
            "prometheus": 'p.push_metrics("job", "metric 1")',
            "qdrant": 'p.upsert_point("demo", 1, ["1"], "{}")',
            "s3": 'p.bucket_exists("bucket")',
        }
        for package, constructor, _, _, _, _, _ in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                handling = ' or (err) { << false }' if package in ("elasticsearch", "etcd", "influxdb") else ''
                source = f'''@main
#{package}
#os
fn attempt(p &{package}.Pool) bool {{ << {calls[package]}{handling} }}
fn main() int {{
    endpoint := os.getenv("AUDIT_ENDPOINT")
    p := {constructor}
    if attempt(&p) {{ << 1 }}
    if !attempt(&p) {{ << 2 }}
    p.close()
    << 0
}}
'''
                body = b'{"kvs":[]}' if package == "etcd" else b"{}"
                if package == "s3":
                    body = b""
                good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                bad = b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n"
                with self.http_pool_peer([(bad, False), (good, False)]) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 2)

    def test_remaining_http_pool_concurrent_business_calls(self):
        for package, constructor, call, invalid, body, _, _ in self.remaining_http_pool_cases():
            with self.subTest(package=package):
                # Retain all three successful connections. max_idle=1 may
                # legitimately close overflow sockets and redial later.
                constructor = constructor[:-2] + '3)'
                source = f'''@main
#{package}
#os
fn worker(p {package}.Pool, start chan int, ready chan int, done chan int) {{
    ready <- 1
    _ := <-start
    mut failed := 0
    for i in 0:3 {{
        result := {call} or (err) {{ failed = 1
            done <- failed
            << }}
        if {invalid} {{ failed = 2 }}
    }}
    done <- failed
}}
fn main() int {{
    endpoint := os.getenv("AUDIT_ENDPOINT")
    p := {constructor}
    start := chan int{{cap: 3}}
    ready := chan int{{cap: 3}}
    done := chan int{{cap: 3}}
    for i in 0:3 {{ xu worker(p.dup(), start, ready, done) }}
    for i in 0:3 {{ _ := <-ready }}
    for i in 0:3 {{ start <- 1 }}
    for i in 0:3 {{ if <-done != 0 {{ << 1 }} }}
    p.close()
    if p.get_conn() >= 0 {{ << 2 }}
    << 0
}}
'''
                if package == "prometheus":
                    source = source.replace(' or (err) { failed = 1\n            done <- failed\n            << }', '')
                response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                with self.http_pool_peer([(response, False)] * 9) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(len(stats["requests"]), 9)
                self.assertGreaterEqual(stats["connections"], 1)
                self.assertLessEqual(stats["connections"], 3)

    def test_milvus_pool_close_rejects_borrows_and_closes_returned_socket(self):
        source = '''@main
#milvus
#net
#os
fn main() int {
    p := milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)
    fd := p.get_conn()
    if fd < 0 { << 1 }
    p.close()
    p.close()
    p.put_conn(fd)
    if p.get_conn() >= 0 { println("closed pool opened a connection")
        << 2 }
    text := "probe"
    n := net.conn_write(fd, &text) or (err) { << 0 }
    if n > 0 { println("returned socket survived close")
        << 3 }
    << 0
}
'''
        with self.http_pool_peer([]) as (endpoint, stats):
            self.run_xu(source, ["milvus"], endpoint)
        self.assertLessEqual(stats["connections"], 1)

    def test_milvus_pool_honors_response_connection_close(self):
        body = b'{"code":0,"data":[]}'
        base = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n"
        source = '''@main
#milvus
#os
fn main() int {
    p := milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)
    for i in 0:2 {
        result := p.search("demo", ["1"], 1) or (err) { println(err.msg())
            << 1 }
    }
    p.close()
    << 0
}
'''
        responses = [(base + b"Connection: keep-alive, ClOsE\r\n\r\n" + body, True),
                     (base + b"\r\n" + body, False)]
        with self.http_pool_peer(responses) as (endpoint, stats):
            self.run_xu(source, ["milvus"], endpoint)
        self.assertEqual(stats["connections"], 2)

    def test_meilisearch_and_neo4j_pool_business_reuse(self):
        cases = [
            ("meilisearch", 'meilisearch.open_pool(os.getenv("AUDIT_ENDPOINT"), "", 1)',
             'p.search("demo", "query", 1)', b'{"hits":[]}'),
            ("neo4j", 'neo4j.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)',
             'p.execute("RETURN 1")', b'{"results":[],"errors":[]}'),
        ]
        for package, constructor, call, body in cases:
            with self.subTest(package=package):
                source = f'''@main
#{package}
#os
fn main() int {{
    p := {constructor}
    for i in 0:2 {{
        result := {call} or (err) {{ println(err.msg())
            << 1 }}
    }}
    p.close()
    if p.get_conn() >= 0 {{ << 2 }}
    << 0
}}
'''
                response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                with self.http_pool_peer([(response, False)] * 2) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)
                self.assertEqual(len(stats["requests"]), 2)

    def test_chroma_and_clickhouse_pool_business_reuse(self):
        cases = [
            ("chroma", 'chroma.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)',
             'p.query("demo", "[[1]]", 1)', b'{"ids":[["demo"]]}'),
            ("clickhouse", 'clickhouse.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)',
             'p.query("SELECT 1")', b'1\n'),
        ]
        for package, constructor, call, body in cases:
            with self.subTest(package=package):
                source = f'''@main
#{package}
#os
fn main() int {{
    p := {constructor}
    for i in 0:2 {{
        result := {call} or (err) {{ println(err.msg())
            << 1 }}
    }}
    p.close()
    if p.get_conn() >= 0 {{ << 2 }}
    << 0
}}
'''
                response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                with self.http_pool_peer([(response, False)] * 2) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)

    def test_chroma_and_clickhouse_all_pool_methods(self):
        cases = [
            ("chroma", 'chroma.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)', '\n'.join([
                '    if !p.heartbeat() { << 1 }',
                '    id := p.create_collection("demo") or (err) { << 2 }',
                '    if id != "demo" { << 3 }',
                '    if !p.add(id, "[1]", "[[1]]", "[]") { << 4 }',
                '    p.query(id, "[[1]]", 1) or (err) { << 5 }',
                '    if !p.delete_collection("demo") { << 6 }']),
                [b'{}', b'{"id":"demo"}', b'{}', b'{"ids":[[1]]}', b'{}']),
            ("clickhouse", 'clickhouse.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)', '\n'.join([
                '    p.exec("CREATE TABLE demo") or (err) { << 1 }',
                '    p.query("SELECT 1") or (err) { << 2 }',
                '    rows := p.query_tsv("SELECT 1, 2") or (err) { << 3 }',
                '    if rows.len != 1 || rows[0].len != 2 || rows[0][1] != "2" { << 4 }']),
                [b'', b'1\n', b'1\t2\n']),
        ]
        for package, constructor, calls, bodies in cases:
            with self.subTest(package=package):
                source = f'@main\n#{package}\n#os\nfn main() int {{\n    p := {constructor}\n{calls}\n    p.close()\n    << 0\n}}\n'
                responses = [(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, False) for body in bodies]
                with self.http_pool_peer(responses) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)
                self.assertEqual(len(stats["requests"]), len(bodies))

    def test_chroma_clickhouse_discard_http_failures_and_close_responses(self):
        cases = [
            ("chroma", 'chroma.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)', 'p.query("demo", "[[1]]", 1)', b'{"ids":[[1]]}'),
            ("clickhouse", 'clickhouse.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)', 'p.query("SELECT 1")', b'1\n'),
        ]
        for package, constructor, call, body in cases:
            good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            first_responses = [
                (b"HTTP/1.1 503 Failed\r\nContent-Length: 0\r\n\r\n", False, True),
                (b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\nx", True, True),
                (b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + body, True, False),
                (b"HTTP/1.0 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, True, False),
            ]
            for response, closes, fails in first_responses:
                with self.subTest(package=package, response=response):
                    source = f'@main\n#{package}\n#os\nfn failed(p &{package}.Pool) bool {{\n    {call} or (err) {{ << true }}\n    << false\n}}\nfn main() int {{\n    p := {constructor}\n    if failed(&p) != {str(fails).lower()} {{ << 1 }}\n    {call} or (err) {{ << 2 }}\n    p.close()\n    << 0\n}}\n'
                    with self.http_pool_peer([(response, closes), (good, False)]) as (endpoint, stats):
                        self.run_xu(source, [package], endpoint)
                    self.assertEqual(stats["connections"], 2)

    def test_pool_mutations_and_health_share_business_connection(self):
        cases = [
            ("milvus", 'milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)', '''
    p.create_collection(milvus.CollectionConfig{name: "demo", dimension: 1, metric_type: "L2"}) or (err) { << 1 }
    p.insert("demo", [milvus.VectorEntity{id: 1, vector: ["1"], text: "demo"}]) or (err) { << 2 }
    p.drop_collection("demo") or (err) { << 3 }
    p.search("demo", ["1"], 1) or (err) { << 4 }
''', [b'{"code":0}'] * 3 + [b'{"code":0,"data":[]}'],
             ["/v2/vectordb/collections/create", "/v2/vectordb/entities/insert", "/v2/vectordb/collections/drop", "/v2/vectordb/entities/search"]),
            ("meilisearch", 'meilisearch.open_pool(os.getenv("AUDIT_ENDPOINT"), "", 1)', '''
    if !p.health() { << 1 }
    p.create_index("demo", "id") or (err) { << 2 }
    p.add_documents("demo", "[]") or (err) { << 3 }
    p.delete_index("demo") or (err) { << 4 }
    p.search("demo", "query", 1) or (err) { << 5 }
''', [b'{"status":"available"}'] + [b'{"taskUid":1}'] * 3 + [b'{"hits":[]}'],
             ["/health", "/indexes", "/indexes/demo/documents", "/indexes/demo", "/indexes/demo/search"]),
        ]
        for package, constructor, calls, bodies, paths in cases:
            with self.subTest(package=package):
                source = f'''@main
#{package}
#os
fn main() int {{
    p := {constructor}
{calls}
    p.close()
    << 0
}}
'''
                responses = [(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, False) for body in bodies]
                with self.http_pool_peer(responses) as (endpoint, stats):
                    self.run_xu(source, [package], endpoint)
                self.assertEqual(stats["connections"], 1)
                self.assertEqual([head.split(b" ")[1].decode() for head, body in stats["requests"]], paths)

    def test_pools_discard_failures_then_recover(self):
        cases = [
            ("milvus", 'milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)',
             'p.search("demo", ["1"], 1)', b'{"code":0,"data":[]}', b'{"code":1,"data":[]}'),
            ("meilisearch", 'meilisearch.open_pool(os.getenv("AUDIT_ENDPOINT"), "", 1)',
             'p.search("demo", "query", 1)', b'{"hits":[]}', b'{"hits":{}}'),
            ("neo4j", 'neo4j.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)',
             'p.execute("RETURN 1")', b'{"results":[],"errors":[]}', b'{"errors":[{"code":"fail"}]}'),
        ]
        for package, constructor, call, body, bad_body in cases:
            good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            service_error = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(bad_body)).encode() + b"\r\n\r\n" + bad_body
            failures = [(b"HTTP/1.1 503 Failed\r\nContent-Length: 0\r\n\r\n", False),
                        (b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\n{", True),
                        (service_error, False)]
            for bad, close in failures:
                with self.subTest(package=package, response=bad):
                    pool_type = {"milvus": "milvus.MilvusPool", "meilisearch": "meilisearch.MeiliPool", "neo4j": "neo4j.Pool"}[package]
                    source = f'''@main
#{package}
#os
fn failed(p &{pool_type}) bool {{
    first := {call} or (err) {{ << true }}
    << false
}}
fn main() int {{
    p := {constructor}
    if !failed(&p) {{ << 1 }}
    second := {call} or (err) {{ println(err.msg())
        << 2 }}
    p.close()
    << 0
}}
'''
                    with self.http_pool_peer([(bad, close), (good, False)]) as (endpoint, stats):
                        self.run_xu(source, [package], endpoint)
                    self.assertEqual(stats["connections"], 2)

    def test_meilisearch_health_failure_discards_connection(self):
        source = '''@main
#meilisearch
#os
fn main() int {
    p := meilisearch.open_pool(os.getenv("AUDIT_ENDPOINT"), "", 1)
    if p.health() { << 1 }
    if !p.health() { << 2 }
    p.close()
    if p.health() { << 3 }
    << 0
}
'''
        body = b'{"status":"available"}'
        good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        with self.http_pool_peer([(b"HTTP/1.1 503 Failed\r\nContent-Length: 0\r\n\r\n", False), (good, False)]) as (endpoint, stats):
            self.run_xu(source, ["meilisearch"], endpoint)
        self.assertEqual(stats["connections"], 2)

    def test_socket_pool_capacity_shared_close_and_late_adopt(self):
        source = '''@main
#net
#net/pool
#os
fn closed(fd int) bool {
    text := "probe"
    n := net.conn_write(fd, &text) or (err) { << true }
    << n <= 0
}
fn main() int {
    state := pool.new_state()
    p := pool.SocketPool{state: state, max_idle: 1}
    copy := p
    if p.take() != -1 { << 1 }
    first := net.dial(os.getenv("AUDIT_ENDPOINT")) or (err) { << 2 }
    second := net.dial(os.getenv("AUDIT_ENDPOINT")) or (err) { << 3 }
    if !p.adopt(first) || !p.adopt(second) { << 4 }
    p.release(first, true)
    p.release(second, true)
    if !closed(second) { << 5 }
    if copy.take() != first { << 6 }
    late := net.dial(os.getenv("AUDIT_ENDPOINT")) or (err) { << 9 }
    copy.close()
    if p.take() != -2 { << 7 }
    p.release(first, true)
    p.release(first, true)
    if !closed(first) { << 8 }
    if p.adopt(late) || !closed(late) { << 10 }
    p.close()
    << 0
}
'''
        with server(lambda peer: self.assertEqual(peer.recv(1), b"")) as endpoint:
            self.run_xu(source, [], endpoint)

    def test_http_response_reuse_requires_persistence_and_framing(self):
        source = '''@main
#net/http
fn reusable(status string, headers string, method string, code int) bool {
    r := http.Response{status_code: code, status: status, headers: headers, body: "", sticky: ""}
    << r.reusable(method)
}
fn main() int {
    prefix := "HTTP/1.1 200 OK\\r\\n"
    if !reusable("HTTP/1.1 200 OK", prefix + "Content-Length: 0", "GET", 200) { << 1 }
    if !reusable("HTTP/1.1 200 OK", prefix + "Transfer-Encoding: chunked", "GET", 200) { << 2 }
    if reusable("HTTP/1.1 200 OK", prefix, "GET", 200) { << 3 }
    if reusable("HTTP/1.1 200 OK", prefix + "Content-Length: 0\\r\\nConnection: keep-alive\\r\\nconnection: ClOsE", "GET", 200) { << 4 }
    if reusable("HTTP/1.0 200 OK", "HTTP/1.0 200 OK\\r\\nContent-Length: 0", "GET", 200) { << 5 }
    if !reusable("HTTP/1.0 200 OK", "HTTP/1.0 200 OK\\r\\nContent-Length: 0\\r\\nConnection: keep-alive", "GET", 200) { << 6 }
    if !reusable("HTTP/1.1 200 OK", prefix, "HEAD", 200) { << 7 }
    if !reusable("HTTP/1.1 204 OK", "HTTP/1.1 204 OK", "GET", 204) { << 8 }
    if !reusable("HTTP/1.1 304 OK", "HTTP/1.1 304 OK", "GET", 304) { << 9 }
    << 0
}
'''
        self.run_xu(source, [])

    def test_milvus_pool_concurrent_calls_and_close(self):
        source = '''@main
#milvus
#os
fn worker(p milvus.MilvusPool, start chan int, ready chan int, done chan int) {
    ready <- 1
    _ := <-start
    for i in 0:3 {
        result := p.search("demo", ["1"], 1) or (err) { done <- 1
            << }
    }
    done <- 0
}
fn main() int {
    p := milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 4)
    start := chan int{cap: 4}
    ready := chan int{cap: 4}
    done := chan int{cap: 4}
    for i in 0:4 { xu worker(p.dup(), start, ready, done) }
    for i in 0:4 { _ := <-ready }
    for i in 0:4 { start <- 1 }
    for i in 0:4 { if <-done != 0 { << 1 } }
    p.close()
    if p.get_conn() >= 0 { << 2 }
    << 0
}
'''
        body = b'{"code":0,"data":[]}'
        response = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        with self.http_pool_peer([(response, False)] * 12) as (endpoint, stats):
            self.run_xu(source, ["milvus"], endpoint)
        self.assertEqual(len(stats["requests"]), 12)
        self.assertLessEqual(stats["connections"], 4)

    def test_pools_preserve_success_without_reusing_eof_or_http10_socket(self):
        cases = [
            ("milvus", 'milvus.open_pool(os.getenv("AUDIT_ENDPOINT"), 1)',
             'p.search("demo", ["1"], 1)', b'{"code":0,"data":[]}'),
            ("meilisearch", 'meilisearch.open_pool(os.getenv("AUDIT_ENDPOINT"), "", 1)',
             'p.search("demo", "query", 1)', b'{"hits":[]}'),
            ("neo4j", 'neo4j.open_pool(os.getenv("AUDIT_ENDPOINT"), "", "", "", 1)',
             'p.execute("RETURN 1")', b'{"results":[],"errors":[]}'),
        ]
        for package, constructor, call, body in cases:
            good = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            for first, close in [(b"HTTP/1.1 200 OK\r\n\r\n" + body, True),
                                 (b"HTTP/1.0 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, False),
                                 (b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body, True)]:
                with self.subTest(package=package, response=first):
                    source = f'''@main
#{package}
#os
fn main() int {{
    p := {constructor}
    for i in 0:2 {{
        result := {call} or (err) {{ println(err.msg())
            << 1 }}
    }}
    p.close()
    << 0
}}
'''
                    with self.http_pool_peer([(first, close), (good, False)]) as (endpoint, stats):
                        self.run_xu(source, [package], endpoint)
                    self.assertEqual(stats["connections"], 2)

    def test_all_package_consumers_typecheck(self):
        for package in sorted(PACKAGES.iterdir()):
            if not (package / "mod.xu").is_file():
                continue
            with self.subTest(package=package.name), tempfile.TemporaryDirectory(prefix="xu-package-import-") as path:
                work = Path(path)
                (work / "main.xu").write_text(f"@main\n{package.name} #github.com/xulang-org/packages/{package.name}\nfn main() {{}}\n")
                (work / "xu.xun").write_text(f'module: packages_regression\nversion: !ver 0.1.0\nrequire:\n  "github.com/xulang-org/packages": file:{PACKAGES}\n')
                result = subprocess.run([str(XU), "check", str(work)], cwd=REPO, env=compiler_env(work),
                                        capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def run_xu(self, source, packages, endpoint="", runtime_env=None, build_flags=()):
        with tempfile.TemporaryDirectory(prefix="xu-packages-regression-") as path:
            work = Path(path)
            for name in packages:
                source = source.replace("#"+name+"\n", name+" #github.com/xulang-org/packages/"+name+"\n")
            (work / "main.xu").write_text(source)
            (work / "xu.xun").write_text(f'module: packages_regression\nversion: !ver 0.1.0\nrequire:\n  "github.com/xulang-org/packages": file:{PACKAGES}\n')
            built = subprocess.run([str(XU), "build", str(work), *build_flags, "-o", str(work / "probe")],
                cwd=REPO, env=compiler_env(work), capture_output=True, text=True, timeout=90)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            env = dict(os.environ, AUDIT_ENDPOINT=endpoint)
            env.update(runtime_env or {})
            process = subprocess.Popen([str(work / "probe")], cwd=work,
                env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, start_new_session=os.name == "posix")
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                self.fail(f"probe timed out: {stdout}{stderr}")
            result = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

    def test_exec_start_returns_while_child_waits_for_input(self):
        source = '''@main
#os/exec
fn main() int {
    mut cmd := exec.command("python3", "-u", "-c", "import sys,os; os.closerange(3,65536); print(sys.stdin.readline().strip(),flush=True)")
    mut input := cmd.stdin_pipe() or (err) { << 1 }
    mut output := cmd.stdout_pipe() or (err) { << 2 }
    cmd.start() or (err) { << 3 }
    input.write("ready\\n") or (err) { << 4 }
    input.close() or (err) { << 5 }
    value := output.read(64) or (err) { << 6 }
    output.close() or (err) { << 7 }
    cmd.wait() or (err) { << 8 }
    if value != "ready\\n" { << 9 }
    println("start returned before child exit")
    << 0
}
'''
        self.assertIn("before child exit", self.run_xu(source, []))

    def test_mongodb_ignores_ok_and_fields_inside_string_payloads(self):
        source = '''@main
#mongodb
fn main() int {
    decoy := mongodb.bson_bool("ok", true) + mongodb.bson_string("errmsg", "decoy")
    doc := mongodb.build_doc(mongodb.bson_bool("ok", false) + mongodb.bson_string("payload", decoy))
    if mongodb.is_bson_ok(doc + "") { println("false ok accepted")
        << 1 }
    if mongodb.find_bson_string(doc + "", "errmsg").len != 0 { << 2 }
    good := mongodb.build_doc(mongodb.bson_int32("ok", 1) + mongodb.bson_string("value", "actual"))
    if !mongodb.is_bson_ok(good + "") || mongodb.find_bson_string(good + "", "value") != "actual" { << 3 }
    bad := good[0:good.len - 1]
    if mongodb.is_bson_ok(bad) { << 4 }
    invalid_inner := mongodb.build_doc(string_from_byte(2) + "x" + string_from_byte(0) + mongodb.encode_i32(10) + "a" + string_from_byte(0))
    invalid_nested := mongodb.build_doc(mongodb.bson_int32("ok", 1) + mongodb.bson_doc("data", invalid_inner))
    if mongodb.is_bson_ok(invalid_nested) { println("malformed nested document accepted")
        << 5 }
    println("BSON boundaries preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["mongodb"]))

    def test_redis_mixed_array_cannot_silently_convert_to_strings(self):
        source = '''@main
#redis
fn main() int {
    mut c := redis.Client{fd: -1, addr: "", buf: "*3\\r\\n:7\\r\\n$$-1\\r\\n$$0\\r\\n\\r\\n"}
    v := c.read_value() or (err) { println(err.msg())
        << 1 }
    v.as_strings() or (err) { println("mixed array rejected")
        << 0 }
    println("mixed array silently converted")
    << 2
}
'''
        self.assertIn("rejected", self.run_xu(source, ["redis"]))

    def test_redis_nested_types_and_resp3_map_are_preserved(self):
        source = '''@main
#redis
fn main() int {
    mut c := redis.Client{fd: -1, addr: "", buf: "%1\\r\\n+k\\r\\n*4\\r\\n:7\\r\\n_\\r\\n$$0\\r\\n\\r\\n*1\\r\\n#t\\r\\n"}
    v := c.read_value() or (err) { println(err.msg())
        << 1 }
    if v.typ != redis.type_map { << 2 }
    entries := v.elements() or (err) { << 3 }
    if entries.len != 2 { << 4 }
    key := entries[0].as_string() or (err) { << 5 }
    if key != "k" { << 6 }
    items := entries[1].elements() or (err) { << 7 }
    n := items[0].as_int() or (err) { << 8 }
    empty := items[2].as_string() or (err) { << 9 }
    if n != 7 || !items[1].is_nil() || empty.len != 0 { << 10 }
    nested := items[3].elements() or (err) { << 11 }
    truth := nested[0].as_bool() or (err) { << 12 }
    if !truth { << 13 }
    println("RESP types preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["redis"]))

    def test_redis_invalid_bulk_terminator_is_rejected(self):
        source = '''@main
#redis
fn main() int {
    mut c := redis.Client{fd: -1, addr: "", buf: "$$1\\r\\nx!!"}
    c.read_value() or (err) { println("bad CRLF rejected")
        << 0 }
    << 1
}
'''
        self.assertIn("rejected", self.run_xu(source, ["redis"]))

    def test_postgres_null_and_empty_fields_are_distinct(self):
        def packet(kind, body):
            return kind + struct.pack(">I", len(body) + 4) + body

        def peer(conn):
            length = struct.unpack(">I", exact(conn, 4))[0]
            exact(conn, length - 4)
            conn.sendall(packet(b"R", struct.pack(">I", 0)) + packet(b"Z", b"I"))
            self.assertEqual(exact(conn, 1), b"Q")
            length = struct.unpack(">I", exact(conn, 4))[0]
            exact(conn, length - 4)
            columns = b""
            for name in (b"nil", b"empty", b"text"):
                columns += name + b"\0" + struct.pack(">IhIhih", 0, 0, 25, -1, -1, 0)
            row = struct.pack(">Hii", 3, -1, 0) + struct.pack(">i", 5) + b"hello"
            conn.sendall(packet(b"T", struct.pack(">H", 3) + columns) + packet(b"D", row)
                + packet(b"C", b"SELECT 1\0") + packet(b"Z", b"I"))

        source = '''@main
#postgres
#os
#database/sql/driver
fn main() int {
    c := postgres.open("host=" + os.getenv("AUDIT_ENDPOINT") + " user=test dbname=test")
    r := c.query("SELECT NULL, '', 'hello'", []driver.NamedValue{})
    c.close()
    if r.err.len > 0 { println(r.err)
        << 1 }
    if r.ncols != 3 || r.cells.len != 3 { << 2 }
    if r.cells[0] != "n:" || r.cells[1] != "s:" || r.cells[2] != "s:hello" { << 3 }
    println("NULL, empty and text preserved")
    << 0
}
'''
        # DSN host and port are separate fields.
        source = source.replace('"host=" + os.getenv("AUDIT_ENDPOINT") + " user=test dbname=test"',
            '"postgres://test@" + os.getenv("AUDIT_ENDPOINT") + "/test?sslmode=disable"')
        with server(peer) as endpoint:
            self.assertIn("preserved", self.run_xu(source, ["postgres"], endpoint))

    def test_mongodb_write_errors_are_not_success(self):
        def doc(fields):
            return struct.pack("<I", len(fields) + 5) + fields + b"\0"

        def text(name, value):
            return b"\x02" + name + b"\0" + struct.pack("<I", len(value) + 1) + value + b"\0"

        def reply(conn, fields):
            header = exact(conn, 16)
            length, request, _, _ = struct.unpack("<IIII", header)
            exact(conn, length - 16)
            body = b"\0\0\0\0\0" + doc(fields)
            conn.sendall(struct.pack("<IIII", len(body) + 16, 99, request, 2013) + body)

        def peer(conn):
            ok = b"\x10ok\0" + struct.pack("<i", 1)
            reply(conn, ok)
            error = doc(b"\x10code\0" + struct.pack("<i", 11000) + text(b"errmsg", b"duplicate key"))
            errors = doc(b"\x030\0" + error)
            reply(conn, ok + b"\x04writeErrors\0" + errors)

        source = '''@main
#mongodb
#os
fn main() int {
    mut c := mongodb.connect(os.getenv("AUDIT_ENDPOINT")) or (err) { println(err.msg())
        << 1 }
    c.insert_one("test", "test", mongodb.build_doc(mongodb.bson_int32("x", 1))) or (err) {
        println("write rejected: " + err.msg())
        c.close()
        << 0
    }
    c.close()
    println("write error accepted")
    << 2
}
'''
        with server(peer) as endpoint:
            self.assertIn("duplicate key", self.run_xu(source, ["mongodb"], endpoint))

    def test_grpc_trailers_and_fragmented_message(self):
        def frame(kind, flags, data=b""):
            return len(data).to_bytes(3, "big") + bytes([kind, flags]) + struct.pack(">I", 1 if kind in (0, 1, 9) else 0) + data

        def readframe(conn):
            header = exact(conn, 9)
            return header, exact(conn, int.from_bytes(header[:3], "big"))

        def peer(conn):
            self.assertEqual(exact(conn, 24), b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
            readframe(conn)
            conn.sendall(frame(4, 0))
            for _ in range(3):
                readframe(conn)
            conn.sendall(frame(1, 4, b"\x88\x0f\x10\x10application/grpc"))
            packet = b"\0" + struct.pack(">I", 6) + b"abcdef"
            conn.sendall(frame(0, 0, packet[:2]) + frame(0, 0, packet[2:8]) + frame(0, 0, packet[8:]))
            trailer = b"\x00\x0bgrpc-status\x017\x00\x0cgrpc-message\x06denied"
            conn.sendall(frame(1, 1, trailer[:6]) + frame(9, 4, trailer[6:]))
            # Keep the peer open until the client consumes all trailers.
            while conn.recv(4096):
                pass

        source = '''@main
#grpc
#os
fn main() int {
    c := grpc.new_client(os.getenv("AUDIT_ENDPOINT"))
    r := c.call_unary("/test.Service/Call", "") or (err) { println(err.msg())
        << 1 }
    if r.status != 7 || r.message != "denied" || r.payload != "abcdef" {
        println("status=" + str(r.status) + " payload=" + r.payload)
        << 2
    }
    println("trailers and DATA preserved")
    << 0
}
'''
        with server(peer) as endpoint:
            self.assertIn("preserved", self.run_xu(source, ["grpc"], endpoint))

    def test_grpc_huffman_long_path_and_invalid_message_boundaries(self):
        # Huffman bytes from RFC 7541 Appendix B; fixtures do not use the decoder.
        trailers = bytes.fromhex("00889acac8b21234da8f013000899acac8b5254207317f8590b518b27f")
        cases = [
            (b"\0\0\0\0\3abc", trailers, "preserved"),
            (b"\1\0\0\0\3abc", trailers, "compressed"),
            (b"\0\0\0\0\4abc", trailers, "truncated"),
            (b"\0\0\0\0\3abc", b"", "missing grpc-status"),
        ]
        path = "/" + "x" * 20000
        def frame(kind, flags, data=b""):
            stream = 1 if kind in (0, 1, 9) else 0
            return len(data).to_bytes(3, "big") + bytes([kind, flags]) + struct.pack(">I", stream) + data

        for message, trailer, expected in cases:
            with self.subTest(expected=expected):
                def peer(conn):
                    self.assertEqual(exact(conn, 24), b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
                    hdr = exact(conn, 9)
                    settings = exact(conn, int.from_bytes(hdr[:3], "big"))
                    self.assertIn(b"\0\1\0\0\0\0", settings)
                    conn.sendall(frame(4, 0))
                    headers = b""
                    while True:
                        hdr = exact(conn, 9)
                        data = exact(conn, int.from_bytes(hdr[:3], "big"))
                        self.assertLessEqual(len(data), 16384)
                        if hdr[3] in (1, 9):
                            headers += data
                        if hdr[3] == 0 and hdr[4] & 1:
                            break
                    # The literal :path length must use HPACK's continuation integer.
                    self.assertIn(b"\x00\x05:path\x7f\xa2\x9b\x01" + path.encode(), headers)
                    conn.sendall(frame(1, 4, b"\x88\x0f\x10\x10application/grpc") + frame(0, 0, message) + frame(1, 5, trailer))
                    try:
                        while conn.recv(4096):
                            pass
                    except ConnectionResetError:
                        pass

                source = '''@main
#grpc
#os
fn main() int {
    c := grpc.new_client(os.getenv("AUDIT_ENDPOINT"))
    r := c.call_unary("PATH", "") or (err) { println(err.msg())
        << 0 }
    if r.status != 0 || r.payload != "abc" || r.message != "denied" { << 1 }
    println("Huffman trailers preserved")
    << 0
}
'''.replace("PATH", path)
                with server(peer) as endpoint:
                    self.assertIn(expected, self.run_xu(source, ["grpc"], endpoint))

    def test_elasticsearch_decodes_chunked_and_rejects_truncation(self):
        for response, expected in [
            (b"HTTP/1.1 500 Error\r\nContent-Length: 2\r\n\r\n{}", "rejected"),
            (b"HTTP/1.1 200 OK\r\ntRaNsFeR-EnCoDiNg: Chunked\r\n\r\n2;test=1\r\n{}\r\n0\r\nX-End: yes\r\n\r\n", "body={}"),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n{}", "rejected"),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n", "rejected"),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", "rejected"),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n{}", "rejected"),
        ]:
            with self.subTest(expected=expected, response=response):
                def peer(conn):
                    request = b""
                    while b"\r\n\r\n" not in request:
                        request += conn.recv(4096)
                    conn.sendall(response)

                source = '''@main
#elasticsearch
#os
fn main() {
    mut c := elasticsearch.open(os.getenv("AUDIT_ENDPOINT"))
    r := c.info() or (err) { println("rejected: " + err.msg())
        << }
    println("body=" + r)
}
'''
                with server(peer) as endpoint:
                    self.assertIn(expected, self.run_xu(source, ["elasticsearch"], endpoint))

    def test_http_head_no_content_and_interim_keepalive(self):
        responses = [
            b"HTTP/1.1 200 OK\r\nContent-Length: 200\r\n\r\n",
            b"HTTP/1.1 204 No Content\r\n\r\n",
            b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
        ]
        def peer(conn):
            for response in responses:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += conn.recv(4096)
                conn.sendall(response)
            self.assertEqual(conn.recv(1), b"")

        source = '''@main
#net/http
#net
#os
fn main() int {
    fd := net.dial(os.getenv("AUDIT_ENDPOINT")) or (err) { << 1 }
    methods := ["HEAD", "POST", "GET"]
    for i in 0:3 {
        req := methods[i] + " / HTTP/1.1\\r\\nHost: test\\r\\nContent-Length: 0\\r\\n\\r\\n"
        net.conn_write(fd, &req) or (err) { << 2 }
        resp := http.read_response(fd, methods[i] + "") or (err) { println(err.msg())
            << 3 }
        if i < 2 && resp.body.len != 0 { << 4 }
        if i == 1 && resp.status_code != 204 { << 5 }
        if i == 2 && resp.body != "{}" { << 6 }
    }
    net.conn_close(fd) or (err) { << 7 }
    println("keepalive framing preserved")
    << 0
}
'''
        with server(peer) as endpoint:
            self.assertIn("preserved", self.run_xu(source, [], endpoint))

    def test_prometheus_http_500_is_failure(self):
        def peer(conn):
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(4096)
            conn.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")

        source = '''@main
#prometheus
#os
fn main() int {
    p := prometheus.new_pusher(os.getenv("AUDIT_ENDPOINT"), "test")
    if p.push("test 1\\n") { << 1 }
    println("500 rejected")
    << 0
}
'''
        with server(peer) as endpoint:
            self.assertIn("rejected", self.run_xu(source, ["prometheus"], endpoint))

    def test_milvus_returns_actual_distance_and_text(self):
        body = b'{"code": 0, "data": [{"distance": 0.125, "text": "actual text", "id": 1}]}'
        def peer(conn):
            request = b""
            while b"\r\n\r\n" not in request:
                request += conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)

        source = '''@main
#milvus
#os
fn main() int {
    c := milvus.new_client(os.getenv("AUDIT_ENDPOINT"))
    r := c.search("demo", ["0.1"], 1) or (err) { println(err.msg())
        << 1 }
    if r.hits.len != 1 || r.hits[0].id != 1 || r.hits[0].distance != "0.125" || r.hits[0].text != "actual text" {
        println("incorrect search result")
        << 2
    }
    println("search fields preserved")
    << 0
}
'''
        with server(peer) as endpoint:
            self.assertIn("preserved", self.run_xu(source, ["milvus"], endpoint))

    def test_search_results_follow_json_structure(self):
        cases = [
            ("elasticsearch", '{"value":999,"hits":{"hits":[{"_source":{"nested":{"_id":"decoy"},"text":"a\\\"b"},"_score":0.125,"_id":"actual"}],"total":{"value":42,"relation":"eq"}}}',
             'mut c := elasticsearch.open(endpoint)\n    r := c.search("test", "") or (err) { println(err.msg())\n        << 1 }\n    if r.total != 42 || r.hits.len != 1 || r.hits[0].id != "actual" || r.hits[0].score != "0.125" { << 2 }\n    println(r.hits[0].source)',
             {"nested": {"_id": "decoy"}, "text": 'a"b'}),
            ("qdrant", '{"result":[{"payload":{"nested":{"id":99},"text":"payload"},"score":1.25e-2,"id":7}]}',
             'mut c := qdrant.open(endpoint)\n    r := c.search("test", ["0.1"], 1) or (err) { println(err.msg())\n        << 1 }\n    if r.len != 1 || r[0].id != 7 || r[0].score != "1.25e-2" { << 2 }\n    println(r[0].payload_json)',
             {"nested": {"id": 99}, "text": "payload"}),
            ("meilisearch", '{"estimatedTotalHits":42,"hits":[{"title":"actual title","id":7,"nested":{"id":99}}]}',
             'c := meilisearch.client(endpoint, "")\n    r := c.search("test", "query", 1) or (err) { println(err.msg())\n        << 1 }\n    if r.total_hits != 42 || r.hits.len != 1 || r.hits[0].id != 7 || r.hits[0].title != "actual title" { << 2 }\n    println(r.hits[0].raw)',
             {"title": "actual title", "id": 7, "nested": {"id": 99}}),
        ]
        for package, body_text, statements, expected in cases:
            with self.subTest(package=package):
                body = body_text.encode()
                def peer(conn):
                    request = b""
                    while b"\r\n\r\n" not in request:
                        request += conn.recv(4096)
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                source = f'@main\n#{package}\n#os\nfn main() int {{\n    endpoint := os.getenv("AUDIT_ENDPOINT")\n    {statements}\n    << 0\n}}\n'
                with server(peer) as endpoint:
                    self.assertEqual(json.loads(self.run_xu(source, [package], endpoint)), expected)

    def test_search_query_quotes_control_characters(self):
        query = "a\n\t\"b\\c"
        cases = [
            ("elasticsearch", 'mut c := elasticsearch.open(endpoint)\n    c.search_match("test", "field", "a\\n\\t\\\"b\\\\c") or (err) { println(err.msg())\n        << 1 }',
             {"query": {"match": {"field": query}}}, {"hits": {"hits": [], "total": 0}}),
            ("meilisearch", 'c := meilisearch.client(endpoint, "")\n    c.search("test", "a\\n\\t\\\"b\\\\c", 1) or (err) { println(err.msg())\n        << 1 }',
             {"q": query, "limit": 1}, {"hits": [], "totalHits": 0}),
        ]
        for package, statements, expected, response in cases:
            with self.subTest(package=package):
                def peer(conn):
                    request = b""
                    while b"\r\n\r\n" not in request:
                        request += conn.recv(4096)
                    head, body = request.split(b"\r\n\r\n", 1)
                    length = int(next(line.split(b":", 1)[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")))
                    body += exact(conn, length - len(body))
                    self.assertEqual(json.loads(body), expected)
                    data = json.dumps(response).encode()
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
                source = f'@main\n#{package}\n#os\nfn main() int {{\n    endpoint := os.getenv("AUDIT_ENDPOINT")\n    {statements}\n    println("query preserved")\n    << 0\n}}\n'
                with server(peer) as endpoint:
                    self.assertIn("preserved", self.run_xu(source, [package], endpoint))

    def test_sqlite_connections_are_independent_and_open_errors_propagate(self):
        source = '''@main
#sqlite
#database/sql/driver
fn main() int {
    first := sqlite.open(":memory:")
    second := sqlite.open(":memory:")
    created := first.exec("CREATE TABLE isolated (x INTEGER)", []driver.NamedValue{})
    if created.err.len > 0 { << 1 }
    absent := second.query("SELECT x FROM isolated", []driver.NamedValue{})
    if absent.err.len == 0 { << 2 }
    if first.close().len > 0 || second.close().len > 0 { << 3 }
    bad := sqlite.open("/nonexistent/xu-regression/db.sqlite")
    if bad.ping().len == 0 || bad.is_valid() { << 4 }
    bad.close()
    println("connection identity and open errors preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["sqlite"]))

    def test_sqlite_runs_without_python_or_database_process(self):
        source = '''@main
#sqlite
fn main() int {
    c := sqlite.open(":memory:")
    if !c.is_valid() || c.ping().len > 0 { << 1 }
    c.close()
    c.close()
    println("native SQLite works without Python")
    << 0
}
'''
        self.assertIn("without Python", self.run_xu(source, ["sqlite"], runtime_env={"PATH": "/nonexistent/xu-regression-bin"}))

    def test_sqlite_session_transactions_and_typed_values(self):
        source = '''@main
#sqlite
#database/sql/driver
fn main() int {
    c := sqlite.open(":memory:")
    e := c.exec("CREATE TABLE demo (id INTEGER)", []driver.NamedValue{})
    if e.err.len > 0 { println(e.err)
        << 1 }
    tx := c.begin()
    ins := c.exec("INSERT INTO demo VALUES (?)", [driver.NamedValue{name: "", ordinal: 1, value: 123}])
    if ins.err.len > 0 { println(ins.err)
        << 2 }
    if tx.rollback().len > 0 { << 3 }
    r := c.query("SELECT COUNT(*) AS n FROM demo", []driver.NamedValue{})
    if r.err.len > 0 { println(r.err)
        << 4 }
    if r.cells.len != 1 || r.cells[0] != "i:0" { println("rollback ineffective")
        << 5 }
    tx2 := c.begin()
    ins2 := c.exec("INSERT INTO demo VALUES (7)", []driver.NamedValue{})
    if ins2.err.len > 0 || tx2.commit().len > 0 { << 6 }
    row := c.query("SELECT '  padded  ' AS text, NULL AS nil, X'00ff' AS blob, 1.5 AS f", []driver.NamedValue{})
    if row.err.len > 0 { println(row.err)
        << 7 }
    if row.cells.len != 4 || row.cells[0] != "s:  padded  " || row.cells[1] != "n:" || row.cells[2] != "y:00ff" || row.cells[3] != "f:1.5" { << 8 }
    empty := c.query("SELECT id FROM demo WHERE false", []driver.NamedValue{})
    if empty.err.len > 0 || empty.columns.len != 1 || empty.cells.len != 0 { << 9 }
    stmt := c.prepare("SELECT '?' AS marker, ? AS value -- ?")
    if stmt.num_input() != -1 { << 14 }
    prepared := stmt.query([driver.NamedValue{name: "", ordinal: 1, value: 9}])
    if prepared.err.len > 0 || prepared.cells.len != 2 || prepared.cells[0] != "s:?" || prepared.cells[1] != "i:9" { << 15 }
    stmt.close()
    bad := c.exec("INSERT INTO missing VALUES (1)", []driver.NamedValue{})
    if bad.err.len == 0 { << 10 }
    after := c.query("SELECT id FROM demo", []driver.NamedValue{})
    if after.err.len > 0 || after.cells[0] != "i:7" { << 11 }
    if c.close().len > 0 { << 12 }
    closed := c.query("SELECT 1", []driver.NamedValue{})
    if closed.err.len == 0 { << 13 }
    println("SQLite session and values preserved")
    << 0
}
'''
        for path in (":memory:", "test.db"):
            with self.subTest(path=path):
                self.assertIn("preserved", self.run_xu(source.replace(":memory:", path), ["sqlite"]))

    def test_sqlite_native_binding_boundaries_and_result_ownership(self):
        source = r'''@main
#sqlite
#database/sql/driver
fn main() int {
    c := sqlite.open(":memory:")
    nul := string_from_bytes([97, 0, 98])
    row := c.query("SELECT ?, ?, ?, ?, ?, ?, ?, ?", [driver.NamedValue{name: "", ordinal: 1, value: 9223372036854775807}, driver.NamedValue{name: "", ordinal: 1, value: true}, driver.NamedValue{name: "", ordinal: 1, value: -1.25}, driver.NamedValue{name: "", ordinal: 1, value: [u8(0), u8(255)]}, driver.NamedValue{name: "", ordinal: 1, value: []u8{}}, driver.NamedValue{name: "", ordinal: 1, value: nul + ""}, driver.NamedValue{name: "", ordinal: 1, value: ""}, driver.NamedValue{name: "", ordinal: 1, value: "中文\n  "}])
    if row.err.len > 0 { println(row.err)
        << 1 }
    if row.cells.len != 8 || row.cells[0] != "i:9223372036854775807" || row.cells[1] != "i:1" || row.cells[2] != "f:-1.25" || row.cells[3] != "y:00ff" || row.cells[4] != "y:" || row.cells[5] != "s:" + nul || row.cells[6] != "s:" || row.cells[7] != "s:中文\n  " { << 2 }
    named := c.query("SELECT :value, :value, @other, $$other", [driver.NamedValue{name: "value", ordinal: 1, value: 9}, driver.NamedValue{name: "other", ordinal: 2, value: "test"}])
    if named.err.len > 0 || named.cells.len != 4 || named.cells[0] != "i:9" || named.cells[1] != "i:9" || named.cells[2] != "s:test" || named.cells[3] != "s:test" { println(named.err)
        << 3 }
    c.close()
    if row.cells[5] != "s:" + nul || named.cells[3] != "s:test" { << 4 }
    println("native typed buffers and snapshot ownership preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["sqlite"]))

    def test_native_symbol_collision_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="xu-native-collision-") as path:
            work = Path(path)
            (work / "sqlite").mkdir()
            (work / "main.xu").write_text("@main\n#sqlite_native\nfn main() {}\n")
            (work / "sqlite/shadow.xu").write_text('@sqlite\nfn native_error(handle int) string { << "shadow" }\n')
            result = subprocess.run([str(XU), "build", str(work / "main.xu"),
                str(work / "sqlite/shadow.xu"), str(PACKAGES / "sqlite/native/sqlite.xulib"),
                "-o", str(work / "probe")], cwd=REPO, env=compiler_env(work),
                capture_output=True, text=True, timeout=45)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("native symbol collision: _xu_sqlite_native_error", result.stderr)
            self.assertFalse((work / "probe").exists())

    def test_native_link_preserves_function_addresses(self):
        source = '''@main
#sqlite
fn add(x int) int { << x + 1 }
fn apply(f fn(int) int, x int) int { << f(x) }
fn main() int {
    c := sqlite.open(":memory:")
    if c.ping().len > 0 || apply(add, 41) != 42 { << 1 }
    c.close()
    println("native function address preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["sqlite"], build_flags=("-O0",)))

    def test_sqlite_native_rejects_multiple_statements_and_recovers(self):
        source = r'''@main
#sqlite
#database/sql/driver
fn main() int {
    c := sqlite.open(":memory:")
    multi := c.exec("CREATE TABLE hidden (v); ; INSERT INTO hidden VALUES (1)", []driver.NamedValue{})
    if multi.err.len == 0 { << 1 }
    absent := c.query("SELECT name FROM sqlite_master WHERE name='hidden'", []driver.NamedValue{})
    if absent.err.len > 0 || absent.cells.len != 0 { << 2 }
    for i in 0:25 {
        missing := c.query("SELECT ?", []driver.NamedValue{})
        if missing.err.len == 0 { << 3 }
    }
    nul := c.query(string_from_bytes([83, 69, 76, 69, 67, 84, 32, 49, 0, 59, 83, 69, 76, 69, 67, 84, 32, 50]), []driver.NamedValue{})
    if nul.err.len == 0 { << 4 }
    missing_name := c.query("SELECT :value", [driver.NamedValue{name: "other", ordinal: 2, value: 1}])
    if missing_name.err.len == 0 { << 5 }
    mixed := c.query("SELECT :value, ?", [driver.NamedValue{name: "value", ordinal: 1, value: 1}, driver.NamedValue{name: "", ordinal: 1, value: 2}])
    if mixed.err.len == 0 { << 6 }
    c.exec("CREATE TABLE main (id INTEGER PRIMARY KEY)", []driver.NamedValue{})
    c.exec("CREATE TABLE audit (id INTEGER)", []driver.NamedValue{})
    c.exec("CREATE TRIGGER inserted AFTER INSERT ON main BEGIN INSERT INTO audit VALUES (new.id); END", []driver.NamedValue{})
    inserted := c.exec("INSERT INTO main VALUES (7)", []driver.NamedValue{})
    if inserted.err.len > 0 || inserted.rows_affected != 1 || inserted.last_insert_id != 7 { << 7 }
    ddl := c.exec("CREATE TABLE another (id)", []driver.NamedValue{})
    if ddl.err.len > 0 || ddl.rows_affected != 0 { << 8 }
    recovered := c.query("SELECT 1; ; -- trailing comment", []driver.NamedValue{})
    if recovered.err.len > 0 || recovered.cells[0] != "i:1" { println(recovered.err)
        << 9 }
    if c.close().len > 0 { << 10 }
    println("native statement boundaries and error recovery preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["sqlite"]))

    def test_sqlite_reset_rolls_back_and_readonly_restores_writes(self):
        source = '''@main
#sqlite
#database/sql/driver
#context
fn main() int {
    c := sqlite.open(":memory:")
    ctx := context.background()
    created := c.exec("CREATE TABLE demo (v TEXT)", []driver.NamedValue{})
    if created.err.len > 0 { << 1 }
    tx := c.begin()
    text := "quotes ' and ?;\\n  preserved  "
    args := [driver.NamedValue{name: "v", ordinal: 1, value: text + ""}]
    inserted := c.exec("INSERT INTO demo VALUES (:v)", args)
    if inserted.err.len > 0 { << 2 }
    bound := c.query("SELECT v FROM demo", []driver.NamedValue{})
    if bound.err.len > 0 || bound.cells[0] != "s:" + text { << 3 }
    if c.reset_session(&ctx).len > 0 { << 4 }
    if tx.commit().len == 0 { << 5 }
    empty := c.query("SELECT COUNT(*) FROM demo", []driver.NamedValue{})
    if empty.err.len > 0 || empty.cells[0] != "i:0" { println("reset did not rollback")
        << 6 }
    ro := c.begin_tx(6, true)
    denied := c.exec("INSERT INTO demo VALUES ('bad')", []driver.NamedValue{})
    if denied.err.len == 0 { << 7 }
    if ro.rollback().len > 0 { << 8 }
    allowed := c.exec("INSERT INTO demo VALUES ('allowed')", []driver.NamedValue{})
    if allowed.err.len > 0 { << 9 }
    bad := c.begin_tx(2, false)
    blocked := c.exec("INSERT INTO demo VALUES ('blocked')", []driver.NamedValue{})
    if blocked.err.len == 0 { << 10 }
    if bad.rollback().len == 0 { << 11 }
    if c.ping().len > 0 || c.close().len > 0 { << 12 }
    println("reset and transaction errors preserved")
    << 0
}
'''
        self.assertIn("preserved", self.run_xu(source, ["sqlite"]))


if __name__ == "__main__":
    unittest.main()
