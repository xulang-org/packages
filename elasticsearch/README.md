# xu-elasticsearch (Elasticsearch & OpenSearch Driver for Xu)

`packages/elasticsearch` 是为 Xu 语言构建的企业级搜索引擎官方驱动库，全面兼容 Elasticsearch 7.x/8.x 与 OpenSearch 1.x/2.x，通过 `xu.xun` 外部包机制引入。

## 特性

- **零 C 库依赖**：纯 Xu 基于标准库 `#net/http` 发送高性能 RESTful 请求；
- **索引管理**：支持索引创建（`create_index`）、存在检测（`index_exists`）、强制刷新（`refresh`）与删除（`delete_index`）；
- **文档 CRUD**：支持按 ID 写入文档（`index_doc`）、获取文档（`get_doc`）、更新（`update_doc`）及删除（`delete_doc`）；
- **全文检索**：支持完整的 Query DSL（`search`）以及高频快捷全文匹配（`search_match`）；
- **集群状态**：支持集群 Ping 探测（`ping`）与信息汇总（`info`）。

## 安装与使用

在项目 `xu.xun` 中声明：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

在源文件中使用：

```xu
@main

elasticsearch #github.com/xulang-org/packages/elasticsearch

fn main() int {
    mut es := elasticsearch.open("http://127.0.0.1:9200")
    
    // 1. 集群健康探测
    if !es.ping() {
        println("ES is offline")
        << 1
    }

    // 2. 创建索引
    es.create_index("articles", "{}") or >>

    // 3. 写入文档
    es.index_doc("articles", "1", "{\"title\": \"Xu Language Overview\", \"views\": 1024}") or >>
    es.refresh("articles") or >>

    // 4. 全文检索
    res := es.search_match("articles", "title", "Xu") or >>
    println("Total hits: " + str(res.total))

    // 5. 获取与删除
    doc := es.get_doc("articles", "1") or >>
    es.delete_doc("articles", "1") or >>
    es.delete_index("articles") or >>
    << 0
}
```

## 搜索结果与边界

`search` 从 `hits.hits` 读取真实 `_id`、`_score` 和 `_source`；source 保留完整 JSON 子树。缺失/null 的 score 为 "null"；total 缺失为 -1，不以当前页数量代替。`search_match` 对查询文本与字段名使用标准 JSON 转义。

HTTP 响应使用共享 framing 解析，支持 chunked，拒绝截断与歧义长度；仅支持串行请求，当前报文总量限制约 10 MiB。连接池高层调用复用尚待修复。回归入口：`make packages-test`。

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. Secure business calls establish a verified TLS connection per request; manual fd pool APIs remain for plaintext sockets until the connection-object pool migration is complete.
