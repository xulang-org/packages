# gRPC & HTTP/2 Native Driver for Xu

Xu 语言gRPC 远程过程调用客户端与 HTTP/2 连接池接口。

## 特性

- **原生 HTTP/2 协议实现**：支持 Client Preface、SETTINGS 协商握手、PING/ACK 保持、HEADERS 帧与 DATA 帧交互。
- **Protocol Buffers 编解码**：内置高效 Varint、Tag、String 与 Int 字段序列化器。
- **gRPC 消息帧打包**：支持未压缩消息的 5 字节消息头（压缩位 + 4 字节大端长度）。
- **一元调用 (Unary RPC)**：支持有限大小请求发送、响应窗口更新与跨帧消息组装。
- **连接池接口**：基于 channel 队列提供手工借还，高层调用复用与生命周期尚待修复。

## 快速使用

```xu
grpc #github.com/xulang-org/packages/grpc

fn main() {
    mut p := grpc.open_pool("127.0.0.1:9000", 5)
    defer p.close()

    // 构造 Protobuf 请求 (field 1 string: "Xu Language")
    req := grpc.encode_string_field(1, "Xu Language")

    // 发起 gRPC Unary RPC 调用
    resp := p.call_unary("/grpcbin.GRPCBin/DummyUnary", req) or (err) {
        println("gRPC error: " + err.msg())
        <<
    }
    println("Response payload bytes: " + str(resp.payload.len))
}
```

## 当前传输范围

- 支持明文 HTTP/2（h2c prior knowledge）与 `https://` 的 HTTP/2 TLS（ALPN h2）；仍不提供压缩或 streaming RPC。
- 每次 unary 调用读取完整 DATA 消息及 HPACK trailers；`Response.status` 来自 `grpc-status`，非零状态由调用方处理。缺失状态、消息截断、多条 unary 消息或压缩标志会报错。
- HPACK 支持静态表、literal 与 Huffman；通过 `HEADER_TABLE_SIZE=0` 禁用动态表。支持 CONTINUATION、PADDED 和 PING/SETTINGS ACK。
- 请求消息最多 65,530 字节，依赖初始请求窗口至少 65,535；更小窗口明确拒绝。响应消息最多 16 MiB、帧最多 16 KiB、头部最多 64 KiB。
- 连接池的高层调用复用与完整 HTTP/2 会话管理仍在 packages 审计待办中，当前回归覆盖的是直接 unary 调用。

在仓库根执行 `make test` 可验证非零状态、DATA 分帧、Huffman trailers、长路径、压缩与截断拒绝。

## TLS

`grpc.new_client("https://host:port")` performs certificate verification with SNI. The secure unary path keeps the same frame and trailer limits as h2c; the pool's legacy fd checkout remains h2c-only.
