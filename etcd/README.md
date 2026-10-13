# Xu etcd v3 原生分布式配置与协调驱动 (etcd v3 Driver)

`packages/etcd` 是为 Xu 语言编写的高性能 etcd v3 云原生分布式键值存储与微服务协调客户端驱动。

## 特性

- **纯原生依赖**：基于 etcd v3 官方 HTTP/JSON gRPC-Gateway 接口与 Xu 标准库 `#encoding/base64`，无需 CGo 或复杂 protobuf 编译器。
- **KV 核心操作**：支持标准的 `put`、`get`、`delete`。
- **租约管理 (Lease)**：支持分布式租约授予（`lease_grant`）、带租约存储（`put_with_lease`）、续约与撤销，适合分布式锁与服务心跳。
- **连接池支持**：基于共享状态通道管理套接字的借还与关闭（`open_pool`）。

## 引入依赖

在项目的 `xu.xun` 中声明：

```xun
require:
  "github.com/xulang-org/packages": file:../../packages
```

## 快速使用

```xu
etcd #github.com/xulang-org/packages/etcd

fn main() {
    mut client := etcd.open("http://127.0.0.1:2379")

    // 1. 写入配置
    client.put("/config/app/title", "Xu High-Performance Service") or (err) {
        println("写入失败: " + err.msg())
        <<
    }

    // 2. 读取配置
    val := client.get("/config/app/title") or (err) {
        println("读取失败: " + err.msg())
        <<
    }
    println("配置值: " + val)

    // 3. 创建 30 秒分布式租约
    lease_id := client.lease_grant(30) or (err) {
        println("申请租约失败: " + err.msg())
        <<
    }
    client.put_with_lease("/services/node-1", "127.0.0.1:8080", lease_id) or (err) {
        println("服务注册失败: " + err.msg())
        <<
    }

    // 4. 清理
    client.delete("/config/app/title") or (err) { }
    client.lease_revoke(lease_id) or (err) { }
}
```

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。
