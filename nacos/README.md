# Nacos Native Driver for Xu

Xu 语言官方原生 Alibaba Nacos 驱动，提供微服务动态服务注册与发现（Naming Service）以及分布式配置中心（Config Service）功能。

## 特性

- **服务发现与注册 (Naming Service)**：
  - 支持实例注册与健康状态上报 (`/nacos/v1/ns/instance`)
  - 支持实例列表发现与健康检查过滤
  - 支持服务实例下线与优雅注销
- **动态配置中心 (Config Service)**：
  - 支持配置发布 (`/nacos/v1/cs/configs`)
  - 支持配置精准读取
  - 支持配置动态删除
- **原生连接池与并发借还**：基于 channel 队列的高性能连接复用与生命周期管理。
- **纯 Xu 原生实现**：无需 Java/C 动态链接库，直接对接 Nacos HTTP OpenAPI。

## 快速使用

```xu
nacos #github.com/xulang-org/packages/nacos

fn main() {
    mut p := nacos.open_pool("http://127.0.0.1:8848", "public", 5)
    defer p.close()

    // 注册微服务
    p.register_instance("order-service", "127.0.0.1", 8080, "DEFAULT_GROUP")

    // 发布动态配置
    p.publish_config("application.yaml", "DEFAULT_GROUP", "server:\n  port: 8080\n")

    // 读取配置
    cfg := p.get_config("application.yaml", "DEFAULT_GROUP") or (err) {
        println("Config not found")
        <<
    }
    println("Config: " + cfg)
}
```

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. Secure business calls establish a verified TLS connection per request; manual fd pool APIs remain for plaintext sockets until the connection-object pool migration is complete.
