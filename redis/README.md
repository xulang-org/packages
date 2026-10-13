# xu-redis (Redis Client for Xu)

`packages/redis` 是为 Xu 语言构建的高性能、零依赖独立 Redis 客户端库，支持 RESP2 与常用 RESP3 类型，通过 `xu.xun` 的 `require:` 机制作为外部生态库引入。

## 特性

- **协议编解码**：纯 Xu 实现 RESP2 与常用 RESP3 标量、聚合类型；
- **高性能缓冲**：内置高效 Socket 行与 Bulk 字节缓冲；
- **全套数据结构 API**：支持 String、Hash、List、Set、Sorted Set (ZSet)、Key/TTL 管理及通用底层 `do()` 命令；
- **批量与流水线**：支持 `mget`/`mset` 批量操作与 `Pipeline` 流水线高吞吐批量执行；
- **URL 标准解析**：支持 `redis://[:password@]host[:port][/db]` 标准连接字符串；
- **连接池支持**：提供通道借还接口（支持 URL 初始化）；完整生命周期尚待修复；
- **类型安全**：`Value` 抽象包装，提供 `as_string`、`as_int`、`as_bool`、`as_strings`、`is_nil` 等安全转换。

## 安装与引入

在你的项目 `xu.xun` 中声明外部依赖：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

在源文件中引入并使用：

```xu
@main

redis #github.com/xulang-org/packages/redis

fn main() int {
    // 1. 直连或通过 URL 连接
    mut client := redis.open_url("redis://127.0.0.1:6379/0") or >>
    
    // 2. 基础 Key-Value
    client.set("greeting", "Hello Xu") or >>
    val := client.get("greeting") or >>
    println(val)
    
    // 3. 批量操作 (MSET / MGET)
    client.mset(["k1", "v1", "k2", "v2"]) or >>
    vals := client.mget(["k1", "k2"]) or >>

    // 4. 有序集合 (Sorted Set / ZSet)
    client.zadd("leaderboard", 100.5, "Alice") or >>
    client.zadd("leaderboard", 95.0, "Bob") or >>
    leaders := client.zrange("leaderboard", 0, -1) or >>

    // 5. 流水线 (Pipeline)
    mut pipe := client.pipeline()
    pipe.set("p1", "v1")
    pipe.incr("p_cnt")
    results := client.exec_pipeline(mut &pipe) or >>

    // 6. 连接池
    pool := redis.new_pool_url("redis://127.0.0.1:6379/0", 10)
    mut conn := pool.get() or >>
    pong := conn.ping() or >>
    pool.put(conn)

    client.close()
    << 0
}
```

## Value 类型与 1.0.0 迁移

`Value.raw` 保存完整 RESP 编码。数组、map、set、push 的 `elements()` 返回保留类型的 `[]Value`；map 按键、值交替排列。NULL、空字符串、整数、错误和嵌套聚合不会互相转换。推荐使用 Value 构造函数；旧的手工结构字面量需要新增 `raw` 字段。

`as_strings()` 只接受全部元素都是字符串的数组/set，遇到 NULL、整数或嵌套值会报错。包含缺失键的 MGET 请调用 `do(["MGET", ...])` 后逐项读取 `elements()` 与 `is_nil()`；不能把 NULL 当作空字符串。`arr_val` 是旧的字符串视图，通用回复应使用 `elements()`。

当前支持 RESP3 的 map、set、push、bool、double、big number、verbatim 和 blob error；attributes、流式字符串与未知长度聚合明确拒绝。double/big number 保留数值文本。深度最多 64 层、聚合最多 100,000 项、单条响应最多 16 MiB。收到 push 不会自动派发订阅消息。

协议回归入口：`make packages-test`。连接池生命周期仍见仓库的 packages 审计待办。
