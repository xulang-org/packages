# Prometheus Native Metrics Driver for Xu Language

Native Prometheus metrics collection, formatting, registry, and Pushgateway client for Xu applications.

## Features
- **Metric Types**: `Counter`, `Gauge` with multidimensional label dimensions
- **Registry**: Centralized collection, sample grouping, and standard exposition formatting
- **Pushgateway**: HTTP/1.1 metric pushing to remote monitoring servers
- **Connection Pool**: Socket reuse and multi-connection concurrency

## Quick Start

```xu
prometheus #github.com/xulang-org/packages/prometheus

fn main() {
    mut reg := prometheus.new_registry()

    mut http_requests := prometheus.new_counter("http_requests_total", "Total incoming HTTP requests")
    http_requests.inc_labels([
        prometheus.Label{name: "method", value: "GET"},
        prometheus.Label{name: "handler", value: "/api/v1/users"}
    ], 10)

    reg.register_counter(http_requests)

    output := reg.gather()
    println(output)
}
```

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。
