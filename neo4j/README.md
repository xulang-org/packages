# Neo4j Native Driver for Xu

Xu 语言官方原生 Neo4j 图数据库驱动，基于 HTTP Cypher Transactional API 实现。

## 特性

- **Cypher 执行**：提交单条 Cypher 语句并自动提交事务 (`/db/{database}/tx/commit`)。
- **节点与关系图谱操作**：支持创建节点 (Node)、关系 (Relationship)、标签 (Labels) 及路径模式匹配。
- **原生连接池与并发借还**：业务方法复用串行 HTTP 连接；并发调用使用独立借出的连接，关闭后拒绝新借用。
- **自动认证**：内置 HTTP Basic 认证与 Base64 编码。
- **纯 Xu 原生实现**：无需 Cgo/外部动态库，安全跨平台。

## 快速使用

```xu
neo4j #github.com/xulang-org/packages/neo4j

fn main() {
    mut p := neo4j.open_pool("http://127.0.0.1:7474", "neo4j", "secret123", "neo4j", 5)
    defer p.close()

    // 创建节点
    res := p.execute("CREATE (p:Person {name: 'Alice', age: 30}) RETURN p.name, p.age") or (err) {
        println("Cypher error: " + err.msg())
        <<
    }
    println("Result: " + res)
}
```

## 连接池 1.0.0 迁移与生命周期

请使用 `open_pool` / `pool` 构造器；公开池结构的 `idle` 通道已替换为内部 `state` 通道，旧结构字面量需要迁移。不要读写 `state`、关闭或复制管理借出的 fd。`p.dup()` 复制配置并共享生命周期，可将返回的值传入 Xu 并发任务；关闭任何副本会关闭同一个池。

`get_conn()` 在关闭或拨号失败时返回 -1；每个借出的 fd 必须且只能归还一次。确认协议流同步时用 `put_conn(fd)`，读写失败或无法确认流状态时用 `discard_conn(fd)`；不要先手动关闭再归还。`close()` 可重复调用，关闭空闲连接并拒绝新借用；正在借出的连接可完成操作，归还时关闭。并发拨号期间关闭池，新拨出的连接也会立即关闭。

`max_idle` 仅限制保留的空闲连接数量，不限制总打开数量；当前没有 max_open、等待队列、空闲超时或请求截止时间。每个借出的连接仅支持串行 HTTP/1.x 请求，不支持 pipelining。业务方法遇到 HTTP 错误、截断或解析失败会淘汰连接；Connection: close、默认 HTTP/1.0 和依赖 EOF 定界的成功响应保留结果但不入池。不会自动重试业务请求。
