# packages/websocket

Xu 语言原生 WebSocket 客户端驱动包，严格遵循 RFC 6455 规范。

## 特性

- **标准 HTTP Upgrade 握手**：支持 `Sec-WebSocket-Key` 动态协商与 `Sec-WebSocket-Accept`（SHA-1 + Base64）安全校验；
- **全格式二进制帧支持**：支持 Text (0x1)、Binary (0x2)、Close (0x8)、Ping (0x9)、Pong (0xA)；
- **客户端掩码自动封装**：符合 RFC 6455 规范的 4 字节掩码异或机制，兼容变长载荷（<=125, 126 扩展 16-bit, 127 扩展 64-bit）；
- **心跳保活**：自动识别 Ping 帧并回复 Pong 帧；
- **高并发连接池**：内置 `WsPool` 线程安全借还复用与生命周期管理。

## 快速使用

```xu
websocket #github.com/xulang-org/packages/websocket

mut p := websocket.pool("ws://127.0.0.1:8080/ws", 5)

// 借还连接并发送文本消息
echoed := p.echo_text("Hello WebSocket!") or panic

// 关闭连接池
p.close()
```

## TLS

`wss://` endpoints use `WsClient.connect_secure()`, `send_text_secure()`, and `read_text_secure()` with certificate verification and SNI. The legacy integer-fd API remains for `ws://`.
