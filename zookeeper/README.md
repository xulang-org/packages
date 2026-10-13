# ZooKeeper Native Driver for Xu

Xu 语言官方原生 Apache ZooKeeper 分布式协调服务驱动。

## 特性

- **4LW 运维与状态监控**：支持 `ruok`、`stat`、`mntr`、`conf` 等 4LW 命令协议。
- **健康探测与状态检查**：开箱即用的健康检查与高可用监控探针。
- **连接池与生命周期管理**：提供高性能 channel 连接池复用。

## 快速使用

```xu
zookeeper #github.com/xulang-org/packages/zookeeper

fn main() {
    mut p := zookeeper.open_pool("127.0.0.1:2181", 5)
    defer p.close()

    if p.ruok() {
        println("ZooKeeper is healthy (imok)")
    }
    stats := p.stat() or (err) {
        <<
    }
    println("Stats:\n" + stats)
}
```
