# xu-memcached (Memcached Client for Xu)

`packages/memcached` 是为 Xu 语言构建的高性能、零依赖 Memcached 纯原生协议客户端，通过 `xu.xun` 的 `require:` 机制作为外部生态库引入。

## 特性

- **纯原生实现**：采用标准 Memcached 文本协议，零外部 C 库依赖；
- **全套存储与获取**：支持 `set`, `add`, `replace`, `append`, `prepend`, `get`；
- **原子计数与 TTL**：支持原子计数器 `incr`/`decr` 以及更新有效期的 `touch`；
- **运维与度量**：支持服务版本获取 `version`、清空缓存 `flush_all` 以及运行状态 `stats`；
- **并发连接池**：基于 Go/Xu 风格的通道机制实现并发安全无锁连接池。

## 安装与引入

在项目 `xu.xun` 中声明：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

在源文件中使用：

```xu
@main

memcached #github.com/xulang-org/packages/memcached

fn main() int {
    $mc := memcached.open("127.0.0.1:11211") or >>

    // 1. 设置与读取
    mc.set("user:token", "xyz_token_value", 0, 3600) or >>
    tok := mc.get("user:token") or >>
    println("Token: " + tok)

    // 2. 原子计数
    mc.set("counter", "100", 0, 0) or >>
    new_val := mc.incr("counter", 5) or >> // 105

    // 3. 连接池
    pool := memcached.new_pool("127.0.0.1:11211", 5)
    $conn := pool.get() or >>
    ver := conn.version() or >>
    pool.put(conn)

    mc.close()
    << 0
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
