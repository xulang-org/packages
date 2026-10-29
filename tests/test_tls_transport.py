import os
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path


class TlsTransportTests(unittest.TestCase):
    def test_tls13_tls12_certificate_and_http_framing(self):
        root = Path(__file__).resolve().parents[1]
        xu = root.parent / "xu" / "bin" / "xu"
        with tempfile.TemporaryDirectory(prefix="xu-tls-") as td:
            td = Path(td)
            key = td / "key.pem"
            cert = td / "cert.pem"
            cfg = td / "openssl.cnf"
            cfg.write_text(
                """[req]
distinguished_name=req_dn
x509_extensions=v3
prompt=no
[req_dn]
CN=localhost
[v3]
subjectAltName=DNS:localhost,IP:127.0.0.1
"""
            )
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-config", str(cfg),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen(3)
            seen_sni = []
            server_errors = []

            def make_context(max_version=None):
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(certfile=cert, keyfile=key)
                if max_version is not None:
                    context.minimum_version = ssl.TLSVersion.TLSv1_2
                    context.maximum_version = max_version

                def on_sni(sock, name, _ctx):
                    seen_sni.append(name)

                context.set_servername_callback(on_sni)
                return context

            contexts = [make_context(), make_context(ssl.TLSVersion.TLSv1_2), make_context()]

            def serve():
                try:
                    for index, context in enumerate(contexts):
                        raw, _ = listener.accept()
                        try:
                            conn = context.wrap_socket(raw, server_side=True)
                            conn.recv(4096)
                            if index == 2:
                                conn.close()
                                continue
                            if index == 1:
                                response = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello world"
                            else:
                                response = (
                                    b"HTTP/1.1 200 OK\r\n"
                                    b"Transfer-Encoding: chunked\r\n"
                                    b"Connection: close\r\n\r\n"
                                    b"5\r\nhello\r\n"
                                    b"6;part=2\r\n world\r\n"
                                    b"0\r\nX-Trailer: yes\r\n\r\n"
                                )
                            for byte in response:
                                conn.send(bytes([byte]))
                            conn.close()
                        except Exception as exc:  # the hostname-negative case is expected
                            if index != 2:
                                raise
                except Exception as exc:
                    server_errors.append(exc)
                finally:
                    listener.close()

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            source = td / "main.xu"
            source.write_text(
                f'''@main
transport #github.com/xulang-org/packages/transport
#crypto/tls
#crypto/x509
fn main() {{
    cert := read_file("{cert}") or (err) {{
        println(err.msg())
        <<
    }}
    $cfg := tls.new_config("localhost")
    n := x509.append_certificates($&cfg.root_cas, &cert) or (err) {{
        println(err.msg())
        <<
    }}
    if n <= 0 {{
        println("no-root")
        <<
    }}
    req := "GET / HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n"
    ep1 := transport.Endpoint{{scheme: "https", host: "localhost", port: {port}, path: "", secure: true}}
    a := transport.request_raw(ep1, cfg, req, "GET") or (err) {{
        println(err.msg())
        <<
    }}
    if a.body != "hello world" {{
        println("bad-body-1")
        <<
    }}
    $cfg2 := tls.new_config("localhost")
    x509.append_certificates($&cfg2.root_cas, &cert) or (err) {{
        println(err.msg())
        <<
    }}
    ep2 := transport.Endpoint{{scheme: "https", host: "localhost", port: {port}, path: "", secure: true}}
    req2 := "GET / HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n"
    b := transport.request_raw(ep2, cfg2, req2, "GET") or (err) {{
        println(err.msg())
        <<
    }}
    if b.body != "hello world" {{
        println("bad-body-2")
        <<
    }}
    $bad := tls.new_config("wrong.test")
    x509.append_certificates($&bad.root_cas, &cert) or (err) {{
        println(err.msg())
        <<
    }}
    ep3 := transport.Endpoint{{scheme: "https", host: "localhost", port: {port}, path: "", secure: true}}
    req3 := "GET / HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n"
    rejected := hostname_fails(ep3, bad, req3)
    if !rejected {{
        println("hostname-not-rejected")
        <<
    }}
    println("tls-ok")
}}

fn hostname_fails(ep transport.Endpoint, cfg tls.Config, req string) bool {{
    transport.request_raw(ep, cfg, req, "GET") or (err) {{
        << true
    }}
    << false
}}
'''
            )
            (td / "xu.xun").write_text(
                "module: \"example/tls\"\nversion: !ver 0.1.0\nxu: !ver 0.8.3\n"
                f"require:\n  github.com/xulang-org/packages: file:{root}\n"
            )
            env = {**os.environ, "XUPATH": "", "XU_CACHE": str(td / "cache")}
            proc = subprocess.run(
                [str(xu), "run", str(source)],
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            thread.join(timeout=5)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("tls-ok", proc.stdout)
            self.assertEqual(seen_sni[:2], ["localhost", "localhost"])
            self.assertFalse(server_errors, server_errors)
