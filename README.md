# Xu 扩展模块仓库

本仓库以根模块 `github.com/xulang-org/packages` 维护 30 个外部包，包括驱动和共享 `transport` 传输层。编译器和标准库由兄弟 `xu` 仓库维护；标准库不依赖这里的具体服务驱动。根 `xu.xun` 统一管理模块版本；内部包不单独建立模块清单。

推荐本机目录：

```text
xulanguage/
  xu/          编译器与标准库
  packages/    本仓库
  examples/    示例仓库
  bench/       基准仓库
  xulang-org/  站点仓库
```

位置可由 `XU_ROOT` 指定。主仓库使用 `XU_PACKAGES_ROOT` 定位本仓库；核心构建不要求扩展模块存在。

## 项目依赖

```xun
module: my_app
version: !ver 0.1.0
xu: !ver 0.10.3
require:
  "github.com/xulang-org/packages": file:../packages
```

`file:` 相对于消费项目。示例中的 `xu:` 对应本仓库根 `xu.xun` 选定的编译器版本，与模块的 `version:` 独立。导入路径使用 `/` 分隔，例如 `#database/sql/driver`；`#github.com/xulang-org/packages/mysql`、`#github.com/xulang-org/packages/postgres` 是该模块中的包；调用别名仍为 mysql、postgres。调用 `sql.open(mysql.driver(), dsn)` 或 `sql.open(postgres.driver(), dsn)` 使用通用 `database/sql` API。

## 验证与原生资源

在本仓库根执行。`XU_ROOT` 指向包含 `scripts/xunative.py` 的编译器源码 checkout；`XU` 指向可执行的产品编译器，可来自已安装 SDK：

```bash
XU_ROOT=/path/to/xu make native
XU_ROOT=/path/to/xu XU=/path/to/product-sdk/bin/xu make test
python3 scripts/module_versions.py --check
```

若环境中未设置 `XU_ROOT`，默认使用相邻 `../xu`；`XU` 默认是该目录下的 `bin/xu`。测试使用产品编译器和本机协议服务，不需要外部数据库或凭据。SQLite 的 `.xulib` 是忽略的构建产物；首次使用或清洁检出后先运行 `make native`，源码、元数据和下载摘要保存在 `sqlite/native`、`sqlite/vendor`。构建默认查找 Clang，并需要目标 SDK；默认生成两个 macOS 原生切片，当前一步原生链接及 SQLite 运行测试限于这两个目标。其他目标需通过 `python3 sqlite/native/build.py --target <target> --cc <compiler>` 显式指定工具链并采用外部链接流程，尚未完成全平台验证。

TLS 回归当前固定调用相邻 `../xu/bin/xu`，不读取 `XU` 或 `XU_ROOT`；运行完整测试前，该路径也须有可用的产品编译器。

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

`<commit>` 替换为实际审查过的 Git 提交。当前 Xu 支持 GitHub 模块的精确语义版本发现、MVS 依赖图解析，以及通过 `xu.lock`、缓存和 vendor 重放依赖；显式 Git 定位符仍可用于固定提交。GitHub 直接版本发现要求存在相应标签，且版本与模块清单匹配；根 `xu.xun` 中的模块版本本身不代表已发布同名标签。其他主机的版本来源需配置 `XUPROXY` 或使用显式 Git 定位符；公共代理与校验服务的可用性需单独验证。无需在定位符中拼接 mysql 子目录；包路径负责选择子目录。

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

Xu 外部导入需显式别名，别名不改变完整路径身份。Git 来源不包含忽略的 SQLite 原生构建产物；本机使用 SQLite 先在本仓库根执行 `make native`。需要 SQLite 时，使用已构建原生资源的本地 `file:` 依赖，或确认所选 SDK/模块归档包含匹配目标的原生资源。

### Constant naming migration

Constants use `UPPER_SNAKE_CASE`, including exported driver constants such as `redis.TYPE_STR` and `transport.DEFAULT_TIMEOUT_NS`. Update references to the former lowercase constant names. This breaking change moved the root packages module from 0.2.0 to 1.0.0. The current module version and selected Xu compiler release are recorded separately in the root `xu.xun`; internal packages do not have independent versions.

## 许可证

本仓库自有代码和文档采用 [MIT License](LICENSE)。第三方源码保留各自声明；例如 SQLite 的来源与公有领域说明见 [sqlite/vendor/README.md](sqlite/vendor/README.md)。
