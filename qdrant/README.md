# Qdrant Vector Database Native Driver for Xu Language

Native client and connection pool for Qdrant Vector Search Engine, designed for high-performance AI Agent / RAG vector embedding retrieval.

## Features
- **Collection Management**: Create (`Cosine`, `Euclid`, `Dot`), check existence, delete collections
- **Vector Point Upsert**: Write one point per call with a numeric ID, vector, and JSON payload
- **Similarity Search**: K-nearest neighbors search returning scored matches
- **Connection Pool API**: Plaintext business methods reuse sockets with checkout/return, failed-connection eviction, and shared close state; HTTPS limits are listed below

## Quick Start

```xu
@main
qdrant #github.com/xulang-org/packages/qdrant

fn main() int {
    $client := qdrant.open("http://127.0.0.1:6333")
    
    // Create collection for 4-dimensional embeddings
    if !client.create_collection("products", 4, "Cosine") {
        println("Could not create collection")
        << 1
    }

    // Upsert one vector point
    if !client.upsert_point("products", 1, ["0.05", "0.61", "0.76", "0.74"], "{\"name\": \"AI Laptop\"}") {
        println("Could not upsert point")
        << 1
    }

    // Search nearest neighbors
    results := client.search("products", ["0.04", "0.60", "0.75", "0.70"], 3) or (err) {
        println(err.msg())
        << 1
    }
    println("Matches: " + str(results.len))
    << 0
}
```

## 搜索结果与边界

`search` 返回真实 score 与完整 payload JSON 子树。当前 `ScoredPoint.id` 为 int，因此 UUID/字符串 ID 明确报错；缺失 score 也会报错，不再返回伪造的 0。

HTTP 响应使用共享 framing 解析，支持 chunked，拒绝截断与歧义长度；仅支持串行请求，当前报文总量限制约 10 MiB。明文 Pool 业务方法复用连接；HTTPS 限制见下文。在仓库根运行 `make test` 验证。

## 连接池迁移（1.0.0）

明文 HTTP 路径下，Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. `Client` calls establish a verified TLS connection per request. On HTTPS, `Pool.create_collection`, `collection_exists`, `delete_collection`, `upsert_point`, and `ping` delegate to a fresh Client connection. `Pool.search` currently fails because it uses the plaintext-only `get_conn()`; use `Client.search` for HTTPS. Manual fd pool APIs are for plaintext sockets; TLS connections are not pooled.
