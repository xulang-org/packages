# Chroma Vector Database Driver for Xu

Xu 语言官方原生 Chroma AI 向量数据库驱动。

## 特性

- **集合管理**：支持集合创建 (`/api/v1/collections`)、检索与删除。
- **向量与文档 Upsert**：支持嵌入向量 (Embeddings) 与文档 (Documents) 批量入库。
- **近邻相似度检索 (Similarity Search)**：基于向量距离的高性能 Top-K 近邻检索。
- **连接池与生命周期管理**：提供原生 TCP 连接池与高层委托代理。

## 快速使用

```xu
chroma #github.com/xulang-org/packages/chroma

fn main() {
    mut p := chroma.open_pool("http://127.0.0.1:8000", 5)
    defer p.close()

    // 创建集合
    col_id := p.create_collection("tech_docs") or (err) {
        println("Create failed")
        <<
    }

    // 插入向量与文档
    p.add(col_id, "[\"doc1\"]", "[[0.05, 0.61, 0.76]]", "[\"Xu Language Guide\"]")

    // 向量近邻检索
    res := p.query(col_id, "[[0.05, 0.61, 0.76]]", 1) or (err) {
        println("Query failed")
        <<
    }
    println("Results: " + res)
}
```

## 1.0.0 连接池迁移

池的业务方法现在复用实际 TCP 连接，使用完整 HTTP 响应的连接策略判断能否归还。服务错误、截断响应以及 `Connection: close` 会丢弃连接；关闭池后不能再借用连接，关闭期间已借出的连接在归还时释放。

池结构的 `idle` 字段替换为共享 `state`，请通过构造函数创建池。并发任务使用 `p.dup()` 复制配置并共享关闭状态。`max_idle` 只限制空闲连接数，不限制并发打开数；目前没有闲置过期、最大连接数或等待超时策略。

手工借用 `get_conn()` 后，必须恰好调用一次 `put_conn(fd)` 或 `discard_conn(fd)`；协议或网络出错时使用后者。`close()` 可重复调用，池副本共享关闭状态。
