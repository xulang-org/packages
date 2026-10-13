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
    mut client := rabbitmq.open("127.0.0.1", 5672, "guest", "guest", "/") or (err) {
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
