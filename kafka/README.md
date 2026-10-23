# Apache Kafka Driver for Xu

`kafka` 是专为 Xu 语言设计的原生 Apache Kafka 分布式流处理与消息队列客户端。通过纯 Xu 语言实现 Kafka 二进制协议，无须借助 C 库（如 librdkafka）或 JNI 包装。

## 特性

- **纯原生实现**：基于 TCP Socket 与 Kafka 官方二进制协议（Big-Endian 编码），开箱即用。
- **CRC32 校验**：无缝对接 Xu 语言标准库 `#hash/crc32` 校验引擎。
- **生产与消费**：支持单分区/多分区消息发布（`produce`、`produce_key_value`）以及偏移量消息拉取（`fetch`）。
- **探活与元数据**：内置 `ping()`（基于 ApiVersions 协议）与 `metadata(topic)` 探活及 Topic 自动发现。
- **连接池支持**：基于标量套接字通道的轻量级连接池（`open_pool_client` / `new_pool`）。
- **多版本兼任**：全面支持 Kafka 2.x/3.x/4.x，支持现代 KRaft 无 Zookeeper 架构。

## 快速上手

```xu
kafka #github.com/xulang-org/packages/kafka

fn main() {
    mut client := kafka.open("127.0.0.1:9092") or (err) {
        println("Connect failed: " + err.msg())
        <<
    }

    // 1. 探活检测
    if client.ping() {
        println("Kafka broker is healthy!")
    }

    // 2. 生产消息
    topic := "xu-events"
    offset := client.produce(topic, 0, "Hello Kafka from Xu!") or (err) {
        println("Produce failed: " + err.msg())
        client.close()
        <<
    }
    println("Message produced at offset: " + str(offset))

    // 3. 拉取消费消息
    messages := client.fetch(topic, 0, offset)
    for msg in messages {
        println("Consumed: " + msg)
    }

    client.close()
}
```

## 连接池使用

```xu
kafka #github.com/xulang-org/packages/kafka

fn main() {
    pool := kafka.open_pool_client("127.0.0.1:9092", "my-app", 5)

    // 从连接池获取客户端实例
    mut client := pool.get() or (err) {
        println("Pool get failed: " + err.msg())
        <<
    }

    client.ping()

    // 归还至连接池
    pool.put(client)
    pool.close()
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
