# Xu 扩展模块仓库

本仓库以根模块 `github.com/xulang-org/packages` 维护 30 个外部包，包括驱动和共享 `transport` 传输层。编译器和标准库由兄弟 `xu` 仓库维护；标准库不依赖这里的具体服务驱动。根 `xu.xun` 统一管理模块版本；内部包不单独建立模块清单。

推荐本机目录：

```text
xulanguage/
  xu/          编译器与标准库
  packages/    本仓库
  examples/    示例仓库
  bench/       基准仓库
  xulanguage/  站点仓库
```

位置可由 `XU_ROOT` 指定。主仓库使用 `XU_PACKAGES_ROOT` 定位本仓库；核心构建不要求扩展模块存在。

## 项目依赖

```xun
module: my_app
version: !ver 0.1.0
xu: !ver 0.8.2
require:
  "github.com/xulang-org/packages": file:../packages
```

`file:` 相对于消费项目。导入路径使用 `/` 分隔，例如 `#database/sql/driver`；`#github.com/xulang-org/packages/mysql`、`#github.com/xulang-org/packages/postgres` 是该模块中的包；调用别名仍为 mysql、postgres。调用 `sql.open(mysql.driver(), dsn)` 或 `sql.open(postgres.driver(), dsn)` 使用通用 `database/sql` API。

## 验证与原生资源

```bash
make test
XU_ROOT=/path/to/xu make test
XU=/path/to/xu/bin/xu make test
make native
python3 scripts/module_versions.py --check
```

测试使用产品编译器和本机协议服务，不需要外部数据库或凭据。SQLite 的 `.xulib` 是忽略的构建产物；首次使用或清洁检出后先运行 `make native`，源码、元数据和下载摘要保存在 `sqlite/native`、`sqlite/vendor`。默认生成两个 macOS 原生切片；其他目标需显式指定工具链与 `--target`，尚未完成全平台验证。

SQL 驱动接口、连接池和事务管理位于标准库，驱动实现位于本仓库的 mysql、postgres、sqlite 模块。其余数据库、消息、HTTP 服务模块的 API 和限制见各自 README。

HTTP 驱动的 `https://` 与 gRPC 的 TLS 路径使用 `transport` 包和 Xu `crypto/tls`，默认启用系统根证书验证与 SNI；HTTP 响应解析支持分片、chunked trailers、严格长度限制和 EOF 语义。PostgreSQL 的 `sslmode=require|verify-ca|verify-full` 与 MySQL 的 `tls=true|required|verify` 都在原连接上完成 TLS 升级，证书失败不会回退到明文。WebSocket 的 `wss://` 使用随机握手 key。

其余 TCP 协议包都提供同一形式的 `tls_connect(endpoint, default_port)`，返回带证书验证、SNI、读写超时和 close-notify 的 `TlsClient`；协议帧通过 `read_exact`、`write_all` 在该连接上收发，避免驱动各自绕过 TLS。SQLite 是本地原生驱动，没有网络 TLS 层。整数 fd 池接口继续只代表明文 socket，TLS 调用使用 `TlsClient` 或各驱动的安全业务入口。

## GitHub 来源

代码导入使用完整包路径，例如 `#github.com/xulang-org/packages/mysql`。
依赖声明使用根模块：

```xun
require:
  "github.com/xulang-org/packages": git:https://github.com/xulang-org/packages.git@<commit>
```

`<commit>` 替换为实际审查过的 Git 提交。当前编译器使用显式 Git 定位符与内容锁，尚不自动从裸版本发现仓库或执行 MVS。无需在定位符中拼接 mysql 子目录；包路径负责选择子目录。仓库尚未推送，本机验证使用相同结构的临时 Git 来源。

```xu
@main
mysql #github.com/xulang-org/packages/mysql
postgres #github.com/xulang-org/packages/postgres
#database/sql
fn main() {
    _ := mysql.driver()
    _ := postgres.driver()
}
```

Xu 外部导入需显式别名，别名不改变完整路径身份。Git 来源不包含忽略的 SQLite 原生构建产物；本机使用 SQLite 先执行 make native。远程 SQLite 需要包含原生资源的模块归档，该发行流程尚未发布。
