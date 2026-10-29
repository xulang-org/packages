# Cassandra & ScyllaDB Native Driver for Xu

Xu 语言官方原生 Apache Cassandra 与 ScyllaDB 宽列分布式数据库驱动，基于 CQL Binary Protocol v4 二进制协议实现。

## 特性

- **原生 CQL 二进制协议 (v4)**：实现 STARTUP、QUERY、RESULT、ERROR 等帧的编码与解析。
- **CQL 执行**：通过 `execute` 提交语句；支持的语法和操作由服务端决定。
- **原生连接池与并发借还**：基于 channel 队列的高性能连接复用，支持多连接借还与生命周期管理。
- **纯 Xu 原生实现**：协议逻辑由 Xu 实现，无需 Cassandra C/C++ 驱动。

## 快速使用

```xu
cassandra #github.com/xulang-org/packages/cassandra

fn main() {
    $p := cassandra.open_pool("127.0.0.1:9042", "", 5)
    defer p.close()

    // 创建 Keyspace 与表
    p.execute("CREATE KEYSPACE IF NOT EXISTS test_xu WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
    p.execute("CREATE TABLE IF NOT EXISTS test_xu.users (id int PRIMARY KEY, name text);")

    // 写入与查询
    p.execute("INSERT INTO test_xu.users (id, name) VALUES (1, 'Alice');")
    res := p.execute("SELECT * FROM test_xu.users WHERE id = 1;")
    println("Query: " + res)
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
