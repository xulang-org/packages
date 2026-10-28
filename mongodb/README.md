# Xu MongoDB 原生文档型数据库驱动 (MongoDB Driver)

`packages/mongodb` 是为 Xu 语言编写的高性能 MongoDB 原生客户端驱动。

## 特性

- **纯原生实现**：直接基于 TCP Socket 实现 MongoDB Wire Protocol（`OP_MSG` opcode 2013）。
- **轻量 BSON 引擎**：内置高效的 BSON 二进制格式编码与解码器。
- **核心操作**：支持 `hello` 握手、`insert_one`、`find_first_string`、`drop_collection` 及通用 `run_command`。
- **连接池支持**：内置基于 `chan int` 的套接字连接池（`open_pool`），提供手工借还接口；会话复用与完整生命周期尚待修复。

## 引入依赖

在项目的 `xu.xun` 中声明：

```xun
require:
  "github.com/xulang-org/packages": file:../../packages
```

## 快速使用

```xu
mongodb #github.com/xulang-org/packages/mongodb

fn main() {
    mut client := mongodb.open("127.0.0.1:27017") or (err) { println(err.msg())
        << }
    defer client.close()

    // 构造 BSON 文档: { "name": "Ada", "role": "Engineer" }
    doc := mongodb.build_doc(
        mongodb.bson_string("name", "Ada") + 
        mongodb.bson_string("role", "Engineer")
    )

    // 写入文档
    client.insert_one("testdb", "users", doc) or (err) {
        println("写入失败: " + err.msg())
        <<
    }

    // 查询文档
    role := client.find_first_string("testdb", "users", "role") or (err) {
        println("查询失败: " + err.msg())
        <<
    }
    println("Ada 的角色: " + role)
}
```

## 响应校验

按 [BSON 1.1 格式](https://bsonspec.org/spec.html) 的字段边界读取根字段，并校验嵌套文档/数组、长度与终止字节，深度最多 64 层。字符串内容和子文档里的 `ok` 不会决定根命令状态。`run_command` 检查 `responseTo`、OP_MSG framing、根 `ok`、`writeErrors` 和 `writeConcernError`，写错误会返回错误。

目前仅处理未压缩、flags=0、单个 kind=0 文档的 OP_MSG；其他 flags/sections 明确拒绝。`find_first_string` 从 `cursor.firstBatch[0]` 读取字符串字段。报文最大 16 MiB；认证与拓扑发现仍由服务端负责；TLS 连接可通过 `tls_connect(endpoint, 27017)` 建立，并复用同一 BSON 帧接口。

在仓库根运行 `make test` 验证。


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
