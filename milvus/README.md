# packages/milvus

Xu 语言原生 Milvus 向量数据库客户端驱动，兼容 Milvus v2 RESTful API 与高效长连接。

## 特性

- **集合管理**：支持 `create_collection`（自定义维度与度量方式：COSINE, L2, IP）与 `drop_collection`；
- **向量实体批量写入**：支持结构化向量数据批量写入（`insert`）；
- **近邻相似度搜索**：支持 Top-K 向量相似度召回（`search`）；
- **连接池接口**：业务方法实际复用串行 HTTP 连接；支持手工借还、关闭及失败连接淘汰。

## 快速使用

```xu
milvus #github.com/xulang-org/packages/milvus

mut p := milvus.pool("http://127.0.0.1:19530", 5)

// 创建向量集合 (4 维, Cosine 相似度)
p.create_collection(milvus.CollectionConfig{
    name: "test_articles",
    dimension: 4,
    metric_type: "COSINE"
})

// 插入向量实体
p.insert("test_articles", [
    milvus.VectorEntity{id: 1, vector: ["0.1", "0.2", "0.3", "0.4"], text: "AI Architecture"},
    milvus.VectorEntity{id: 2, vector: ["0.9", "0.8", "0.7", "0.6"], text: "Database Systems"}
])

// 近邻检索
res := p.search("test_articles", ["0.1", "0.2", "0.3", "0.4"], 2)
p.close()
```

## 搜索结果与边界

`search` 按 JSON 层级解析响应，要求根 `code` 为 0，返回真实的 `id`、`distance`、`text`。ID 当前必须为整数；缺失 distance 或错误类型会报错，text 缺失/null 为 ""。HTTP 非 2xx 明确报错。

HTTP 响应使用共享 framing 解析，支持 chunked，拒绝截断与歧义长度；仅支持串行请求，当前报文总量限制约 10 MiB。连接池业务方法实际借还连接，服务端要求关闭或无长度正文时丢弃连接。回归入口：`make packages-test`。

## 连接池 1.0.0 迁移与生命周期

请使用 `open_pool` / `pool` 构造器；公开池结构的 `idle` 通道已替换为内部 `state` 通道，旧结构字面量需要迁移。不要读写 `state`、关闭或复制管理借出的 fd。`p.dup()` 复制配置并共享生命周期，可将返回的值传入 Xu 并发任务；关闭任何副本会关闭同一个池。

`get_conn()` 在关闭或拨号失败时返回 -1；每个借出的 fd 必须且只能归还一次。确认协议流同步时用 `put_conn(fd)`，读写失败或无法确认流状态时用 `discard_conn(fd)`；不要先手动关闭再归还。`close()` 可重复调用，关闭空闲连接并拒绝新借用；正在借出的连接可完成操作，归还时关闭。并发拨号期间关闭池，新拨出的连接也会立即关闭。

`max_idle` 仅限制保留的空闲连接数量，不限制总打开数量；当前没有 max_open、等待队列、空闲超时或请求截止时间。每个借出的连接仅支持串行 HTTP/1.x 请求，不支持 pipelining。业务方法遇到 HTTP 错误、截断或解析失败会淘汰连接；Connection: close、默认 HTTP/1.0 和依赖 EOF 定界的成功响应保留结果但不入池。不会自动重试业务请求。
