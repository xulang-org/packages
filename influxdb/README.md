# InfluxDB 2.x Driver for Xu

`influxdb` 是专为 Xu 语言设计的原生 InfluxDB 2.x 时序数据库驱动与连接池库。无需依赖外部 C 库或第三方工具，通过纯 Xu 原生实现与 InfluxDB v2 REST API 的交互。

## 特性

- **Line Protocol 支持**：提供 `Point` 结构体，支持流畅追加 Measurement、Tags、Fields（string/int/float/bool）与时间戳，并支持原生批量 Line 协议字符串写入。
- **Flux 查询引擎**：支持向 InfluxDB 发送 Flux 查询脚本并获取 Annotated CSV 结果。
- **连接池支持**：通过共享状态通道管理连接借还与关闭，提供连接复用池 `open_pool_client` / `new_pool`。
- **服务健康探测**：内置 `health()` 与 `ping()` API，随时探活节点健康状况。

## 快速上手

```xu
influxdb #github.com/xulang-org/packages/influxdb

fn main() {
    $client := influxdb.open(
        "http://127.0.0.1:8086",
        "my-super-secret-token",
        "xu-org",
        "xu-bucket"
    )

    // 1. 服务健康检查
    is_ok := client.health() or (err) {
        println("Health check failed: " + err.msg())
        <<
    }
    println("InfluxDB health: " + str(is_ok))

    // 2. 构造数据点并写入 (Line Protocol)
    $pt := influxdb.new_point("cpu_metrics")
    pt.add_tag("host", "server-01")
    pt.add_tag("region", "ap-east")
    pt.add_field_float("usage_user", "12.35")
    pt.add_field_int("cores", 8)
    pt.add_field_bool("is_active", true)

    client.write_point(pt) or (err) {
        println("Write point error: " + err.msg())
        <<
    }
    println("Metric written successfully!")

    // 3. 执行 Flux 查询
    flux := "from(bucket: \"xu-bucket\") |> range(start: -1h) |> filter(fn: (r) => r._measurement == \"cpu_metrics\")"
    csv := client.query_flux(flux) or (err) {
        println("Flux query error: " + err.msg())
        <<
    }
    println("Flux Result:\n" + csv)
}
```

## 连接池使用

```xu
influxdb #github.com/xulang-org/packages/influxdb

fn main() {
    pool := influxdb.open_pool_client(
        "http://127.0.0.1:8086",
        "my-super-secret-token",
        "xu-org",
        "xu-bucket",
        5
    )

    $client := pool.get()
    client.ping() or (err) {
        println("Ping failed: " + err.msg())
    }

    pool.close()
}
```

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. Secure business calls establish a verified TLS connection per request; manual fd pool APIs remain for plaintext sockets until the connection-object pool migration is complete.
