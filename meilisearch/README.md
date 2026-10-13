# packages/meilisearch

Xu 语言原生 Meilisearch 搜索引擎客户端驱动包。

## 特性

- **轻量开箱即用**：支持健康探针（`health`）、索引管理（`create_index`, `delete_index`）；
- **文档批量写入**：支持批量 JSON 文档推送（`add_documents`）；
- **全文即时检索**：支持毫秒级 Top-K 全文搜索（`search`）；
- **连接池接口**：业务方法实际复用串行 HTTP 连接；支持手工借还、关闭及失败连接淘汰。

## 快速使用

```xu
meilisearch #github.com/xulang-org/packages/meilisearch

mut p := meilisearch.pool("http://127.0.0.1:7700", "masterKey", 5)

// 创建电影索引
p.create_index("movies", "id")

// 写入文档
docs := "[{\"id\": 1, \"title\": \"Inception\"}, {\"id\": 2, \"title\": \"Interstellar\"}]"
p.add_documents("movies", docs)

// 检索
res := p.search("movies", "Inception", 5)
p.close()
```

## 搜索结果与边界

`search` 返回真实 title 与完整命中 JSON，total_hits 使用 totalHits 或 estimatedTotalHits，缺失为 -1。当前 SearchHit.id 为 int，字符串 ID 明确报错；title 缺失/null 为 ""。查询文本使用标准 JSON 转义，HTTP 非 2xx 明确报错。

HTTP 响应使用共享 framing 解析，支持 chunked，拒绝截断与歧义长度；仅支持串行请求，当前报文总量限制约 10 MiB。连接池业务方法实际借还连接，服务端要求关闭或无长度正文时丢弃连接。回归入口：`make packages-test`。

## 连接池 1.0.0 迁移与生命周期

请使用 `open_pool` / `pool` 构造器；公开池结构的 `idle` 通道已替换为内部 `state` 通道，旧结构字面量需要迁移。不要读写 `state`、关闭或复制管理借出的 fd。`p.dup()` 复制配置并共享生命周期，可将返回的值传入 Xu 并发任务；关闭任何副本会关闭同一个池。

`get_conn()` 在关闭或拨号失败时返回 -1；每个借出的 fd 必须且只能归还一次。确认协议流同步时用 `put_conn(fd)`，读写失败或无法确认流状态时用 `discard_conn(fd)`；不要先手动关闭再归还。`close()` 可重复调用，关闭空闲连接并拒绝新借用；正在借出的连接可完成操作，归还时关闭。并发拨号期间关闭池，新拨出的连接也会立即关闭。

`max_idle` 仅限制保留的空闲连接数量，不限制总打开数量；当前没有 max_open、等待队列、空闲超时或请求截止时间。每个借出的连接仅支持串行 HTTP/1.x 请求，不支持 pipelining。业务方法遇到 HTTP 错误、截断或解析失败会淘汰连接；Connection: close、默认 HTTP/1.0 和依赖 EOF 定界的成功响应保留结果但不入池。不会自动重试业务请求。
