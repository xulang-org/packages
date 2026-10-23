# Cassandra & ScyllaDB Native Driver for Xu

Xu 语言官方原生 Apache Cassandra 与 ScyllaDB 宽列分布式数据库驱动，基于 CQL Binary Protocol v4 二进制协议实现。

## 特性

- **原生 CQL 二进制协议 (v4)**：完整支持 STARTUP、QUERY、RESULT、ERROR 原生二进制帧打包与解析。
- **DDL 与 DML 全支持**：支持 Keyspace 创建、表定义、数据插入 (INSERT)、查询 (SELECT)、更新与删除。
- **原生连接池与并发借还**：基于 channel 队列的高性能连接复用，支持多连接借还与生命周期管理。
- **纯 Xu 原生实现**：无需任何 C/C++ 动态库或外部运行时依赖，零开销。

## 快速使用

```xu
cassandra #github.com/xulang-org/packages/cassandra

fn main() {
    mut p := cassandra.open_pool("127.0.0.1:9042", "", 5)
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
