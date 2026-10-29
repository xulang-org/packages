# RabbitMQ Native Driver for Xu Language

AMQP 0-9-1 compliant native protocol driver and high-performance connection pool for RabbitMQ.

## Features
- **AMQP 0-9-1 Protocol**: Handshake (Connection.Start, Tune, Open) & Channel Lifecycle
- **Exchanges & Queues**: `declare_exchange`, `declare_queue`, `bind_queue`
- **Publish & Consume**: `publish`, `get` (with auto-ack or manual ack)
- **High-Throughput Pool**: Socket reuse, concurrency allocation, and thread-safe operations

## Quick Start

```xu
rabbitmq #github.com/xulang-org/packages/rabbitmq

fn main() {
    $client := rabbitmq.open("127.0.0.1", 5672, "guest", "guest", "/") or (err) {
        println("Connect failed: " + err.msg())
        <<
    }

    client.declare_queue("orders", false)
    client.publish("", "orders", "{\"order_id\": 1024}")

    msg := client.get("orders", true)
    println("Received: " + msg)

    client.close()
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
