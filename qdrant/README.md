# Qdrant Vector Database Native Driver for Xu Language

Native client and connection pool for Qdrant Vector Search Engine, designed for high-performance AI Agent / RAG vector embedding retrieval.

## Features
- **Collection Management**: Create (`Cosine`, `Euclid`, `Dot`), check existence, delete collections
- **Vector Point Upsert**: Batch vector embeddings with JSON payloads
- **Similarity Search**: K-nearest neighbors search returning scored matches
- **Connection Pool API**: Manual socket checkout/return; business-path reuse and lifecycle fixes remain pending

## Quick Start

```xu
qdrant #github.com/xulang-org/packages/qdrant

fn main() {
    mut client := qdrant.open("http://127.0.0.1:6333")
    
    // Create collection for 4-dimensional embeddings
    client.create_collection("products", 4, "Cosine")

    // Upsert vector points
    client.upsert_point("products", 1, ["0.05", "0.61", "0.76", "0.74"], "{\"name\": \"AI Laptop\"}")

    // Search nearest neighbors
    results := client.search("products", ["0.04", "0.60", "0.75", "0.70"], 3)
    println("Matches: " + str(results.len))
}
```

## 搜索结果与边界

`search` 返回真实 score 与完整 payload JSON 子树。当前 `ScoredPoint.id` 为 int，因此 UUID/字符串 ID 明确报错；缺失 score 也会报错，不再返回伪造的 0。

HTTP 响应使用共享 framing 解析，支持 chunked，拒绝截断与歧义长度；仅支持串行请求，当前报文总量限制约 10 MiB。连接池高层调用复用尚待修复。回归入口：`make packages-test`。

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。
