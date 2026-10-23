# Xu ClickHouse 原生分析型数据库驱动 (ClickHouse OLAP Driver)

`packages/clickhouse` 是为 Xu 语言编写的高性能 ClickHouse 分析型数据库（OLAP）底层驱动客户端。

## 特性

- **纯原生依赖**：基于 Xu 标准库 HTTP/1.1 REST 实现，无需 CGo 或外部动态链接库。
- **高性能分析**：支持流式 SQL 查询、聚合分析与批量导入。
- **连接池支持**：内置基于 `chan int` 的套接字连接池（`open_pool`），支持高并发请求复用与连接优雅关闭。
- **版本兼容**：全兼容 ClickHouse 22.x/23.x/24.x+。

## 引入依赖

在项目的 `xu.xun` 中声明：

```xun
require:
  "github.com/xulang-org/packages": file:../../packages
```

## 快速使用

```xu
clickhouse #github.com/xulang-org/packages/clickhouse

fn main() {
    mut ch := clickhouse.open("http://127.0.0.1:8123", "default", "", "default")

    // 执行 DDL
    ch.exec("CREATE TABLE IF NOT EXISTS demo (id UInt64, val Float64) ENGINE = Memory") or (err) {
        println("建表失败: " + err.msg())
        <<
    }

    // 批量写入
    ch.insert("demo", "(1, 10.5), (2, 20.8)") or (err) {
        println("写入失败: " + err.msg())
        <<
    }

    // 查询分析
    res := ch.query("SELECT avg(val) FROM demo") or (err) {
        println("查询失败: " + err.msg())
        <<
    }
    println("平均值: " + res)
}
```

## 1.0.0 连接池迁移

池的业务方法现在复用实际 TCP 连接，使用完整 HTTP 响应的连接策略判断能否归还。服务错误、截断响应以及 `Connection: close` 会丢弃连接；关闭池后不能再借用连接，关闭期间已借出的连接在归还时释放。

池结构的 `idle` 字段替换为共享 `state`，请通过构造函数创建池。并发任务使用 `p.dup()` 复制配置并共享关闭状态。`max_idle` 只限制空闲连接数，不限制并发打开数；目前没有闲置过期、最大连接数或等待超时策略。

手工借用 `get_conn()` 后，必须恰好调用一次 `put_conn(fd)` 或 `discard_conn(fd)`；协议或网络出错时使用后者。`close()` 可重复调用，池副本共享关闭状态。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. Secure business calls establish a verified TLS connection per request; manual fd pool APIs remain for plaintext sockets until the connection-object pool migration is complete.
