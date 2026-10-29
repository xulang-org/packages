# Xu S3 / MinIO 兼容对象存储驱动 (S3 Compatible Object Storage Driver)

`packages/s3` 是为 Xu 语言编写的高性能 AWS S3 及兼容对象存储（如 MinIO、Cloudflare R2、Ceph、RustFS 等）的原生客户端驱动。

## 特性

- **标准 AWS Signature Version 4 (SigV4)**：基于 Xu 原生 `crypto/hmac`、`crypto/sha256` 和 `encoding/hex` 实现规范请求哈希、CredentialScope 和多层密钥派生。
- **Bucket 操作**：创建 Bucket (`make_bucket`)、检查存在 (`bucket_exists`)、删除 Bucket (`remove_bucket`)。
- **Object 操作**：上传对象 (`put_object`)、下载对象 (`get_object`)、删除对象 (`remove_object`)、检查对象存在 (`object_exists`)。
- **纯原生依赖**：无需 CGo 或外部动态链接库，直接依赖 Xu 标准库网络与密码学模块。

## 安装与引入

在您的项目根目录 `xu.xun` 中声明对 `s3` 的依赖：

```xun
name: my_app
version: 0.1.0

require:
  "github.com/xulang-org/packages": file:../../packages
```

## 快速使用示例

```xu
s3 #github.com/xulang-org/packages/s3

fn main() {
    $client := s3.open("http://127.0.0.1:9000", "minioadmin", "minioadmin", "us-east-1")

    // 1. 创建 Bucket
    client.make_bucket("xu-test-bucket") or (err) {
        println("创建 Bucket 失败: " + err.msg())
        <<
    }

    // 2. 上传对象
    content := "Hello Xu S3 Object Storage!"
    client.put_object("xu-test-bucket", "notes/hello.txt", content, "text/plain") or (err) {
        println("上传失败: " + err.msg())
        <<
    }

    // 3. 读取对象
    got := client.get_object("xu-test-bucket", "notes/hello.txt") or (err) {
        println("读取失败: " + err.msg())
        <<
    }
    println("读取内容: " + got)

    // 4. 删除对象与 Bucket
    client.remove_object("xu-test-bucket", "notes/hello.txt") or (err) {
        println("删除对象失败: " + err.msg())
    }
    client.remove_bucket("xu-test-bucket") or (err) {
        println("删除 Bucket 失败: " + err.msg())
    }
}
```

## 连接池迁移（1.0.0）

Pool 的业务方法使用池中借出的连接，完整成功响应按 HTTP 持久性归还；传输、HTTP、解析失败或返回 false 的操作淘汰连接。EOF、Connection: close 和未协商 keep-alive 的 HTTP/1.0 响应仍返回当前结果，但连接不再复用。

池结构字面量中的 `idle` 改为共享 `state`，建议使用 `new_pool`/`open_pool` 构造。构造器和现有业务方法的调用方式保持一致；布尔方法继续返回 false，原来抛错的方法继续抛错。新增 `dup()` 克隆配置并共享连接和关闭状态，新增 `discard_conn(fd)` 用于失败连接。

`close()` 可重复调用，关闭后 `get_conn()` 返回 -1。关闭时正在使用的连接在归还时关闭；关闭与建连竞争时新连接也会关闭。手动借出的 fd 必须恰好归还或淘汰一次，不能在归还后继续持有它。`put_conn` 只接收本池实际借出的 fd。

`max_idle` 仅限制保留的空闲连接数量，未实现总连接上限、借用等待取消、闲置超时和寿命。配置型 Client/Pusher 的直接调用仍自行建连并关闭，应使用 Pool 业务方法获得连接复用。

## TLS

`http://` and `https://` endpoints are accepted. HTTPS uses Xu `crypto/tls`, verifies the server against the system root pool, and sends the endpoint host as SNI. Secure business calls establish a verified TLS connection per request; manual fd pool APIs remain for plaintext sockets until the connection-object pool migration is complete.
