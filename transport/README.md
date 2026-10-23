# Shared transport

`packages/transport` is the common TCP/TLS transport used by the HTTP, gRPC,
and WebSocket drivers.

- `parse_endpoint` recognizes `http`, `https`, `tls`, `tcp`, and host/port forms.
- `dial` uses Xu `crypto/tls` for secure endpoints, with a 30-second default
  connect/handshake timeout, system root verification, and SNI from the host.
- `Conn` retains TLS session state and buffered HTTP bytes; callers must keep the
  connection object rather than reducing it to an integer file descriptor.
- `read_response` validates HTTP/1.x framing, supports `Content-Length` and
  chunked bodies, and preserves bytes belonging to the next response.

`insecure_skip_verify`, custom roots, and client certificates can be supplied by
callers that use `dial` directly through `tls.Config`. Driver convenience
constructors intentionally keep verification enabled by default.
