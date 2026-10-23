# xu-nats (NATS Cloud-Native Messaging Driver for Xu)

`packages/nats` 是为 Xu 语言构建的高性能、纯原生 NATS 消息队列驱动包，遵循标准 NATS 协议，通过 `xu.xun` 外部包机制引入。

## 特性

- **零依赖纯原生**：纯 Xu 实现 NATS 文本流协议与连接握手；
- **全模式支持**：
  - **发布 / 订阅 (Pub/Sub)**：基于 Subject 主题的毫秒级即时广播；
  - **工作队列 (Queue Groups)**：支持指定队列组，多个消费者自动均摊负载均衡；
  - **请求-响应模式 (Request-Reply)**：微服务同步 RPC 调用的理想选择；
- **心跳保活**：全自动 PING / PONG 心跳握手与维持。

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

nats #github.com/xulang-org/packages/nats

fn main() int {
    mut nc := nats.connect("127.0.0.1:4222") or >>

    // 1. 订阅主题
    sub := nc.subscribe("orders.new") or >>

    // 2. 发布消息
    nc.publish("orders.new", "{\"order_id\": 1001, \"item\": \"Xu Book\"}") or >>

    // 3. 接收消息
    msg := nc.next_msg() or >>
    println("Received on [" + msg.subject + "]: " + msg.data)

    // 4. 工作队列负载均衡订阅
    worker_sub := nc.queue_subscribe("tasks", "task_workers") or >>

    nc.close()
    << 0
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
