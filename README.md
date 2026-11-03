# Xu 扩展模块仓库

本仓库以根模块 `github.com/xulang-org/packages` 维护 31 个外部包，包括驱动、共享 `transport` 传输层和可选 `metal` 计算接口。编译器和标准库由兄弟 `xu` 仓库维护；标准库不依赖这里的具体服务驱动。根 `xu.xun` 统一管理模块版本；内部包不单独建立模块清单。

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
xu: !ver 0.11.3-dev.10
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

### 可选 Metal 计算

`metal` 包通过 `xu-c-2` 同步借用接口提供显式整数 GPU buffer、批量复制和完成等待，当前唯一算子为 `madd_i64`，按模 2⁶⁴ 计算 `dst[i] = a[i] * factor + b[i]`。本开发批次选择 Xu `0.11.3-dev.10`；旧 SDK 会拒绝新原生格式，需要旧 SDK 的应用继续使用此前的包版本。它不改变 Xu `int` 的位宽，不执行浮点精度转换，也不自动迁移普通循环。首次使用须单独构建：

```bash
XU_ROOT=/path/to/xu make metal-native
```

默认构建 macOS ARM64/x64 的 Metal 切片，以及 Linux/Windows ARM64/x64 的无设备 C 切片。Metal 切片需要 Clang 与匹配的 macOS SDK；其他切片不包含 Apple 头文件。`make native` 继续只构建 SQLite，未导入 `metal` 的应用不需要构建 Metal 资源。生成的 `.xulib` 不进入 Git；Git/file 依赖使用前须在本机准备原生资源。macOS 一步链接会读取原生对象中的 framework autolink 信息；其他平台目前使用 `xu build --emit-obj` 和外部链接流程，尚未完成应用真机运行验证。

```xu
@main
metal #"github.com/xulang-org/packages/metal"
fn run() int {
    d := metal.open() or >>
    $a := d.buffer_i64(1) or >>
    $b := d.buffer_i64(1) or >>
    $dst := d.buffer_i64(1) or >>
    a.put(0, 3) or >>
    b.put(0, 4) or >>
    f := metal.madd_i64($&dst, &a, &b, 3, 1) or >>
    f.wait() or >>
    values := dst.ints() or >>
    println(values[0]) // 13
    << 0
}
fn main() int {
    run() or (err) {
        eprintln(err.msg())
        << 1
    }
    << 0
}
```

`Device` 持有一个队列及一个初始化时建立的整数 pipeline；同一 Device 的后续提交复用 pipeline。`buffer_i64(n)` 分配零初始化、连续、步长为 8 字节的一维存储，`max_words()` 返回设备与索引共同允许的上限。提交的长度可以小于容量；零长度合法且不提交 GPU 命令。三个 buffer 必须属于同一 Device，输出不能与输入重叠，两个只读输入可以相同。`put` 按元素写入；`snapshot()` 复制已完成数据，`Snapshot.word/ints/byte_at/bytes` 读取这份不可变副本，`Buffer.ints()` 是创建副本并读取的便捷方法。当前接口不提供零复制视图或通用 shape/stride tensor。

提交后参与计算的 buffer 保持忙状态，直到 `Fence.wait()` 确认完成；期间主机读写、再次提交或显式关闭返回 `BUSY`，即使 GPU 已物理完成。成功等待可重复调用。命令失败返回 `COMMAND` 并禁止继续使用输出内容；应丢弃输出 buffer。错误通过 Xu `errors.CodeErr` 传播，可用本包导出的状态常量判断。无设备/不支持平台返回 `UNAVAILABLE`，不能把它当作已执行 CPU 回退。

四种 owner 均不可复制，离开作用域和错误展开会释放其句柄。丢弃 buffer/Device 时，pending fence 仍保留实际 GPU 资源；丢弃 fence 会阻塞等待并清理，失败时同样回收。显式 `close()` 成功后可重复调用；关闭 pending fence、忙 buffer 或仍有子资源的 Device 返回 `BUSY` 并保留所有权。`live_handles()`、`pipeline_count()`、`dispatch_count()` 用于诊断，不作为应用同步机制。每进程最多有 4096 个同时存活的原生句柄，耗尽时返回 `RESOURCE_LIMIT`，释放后可继续分配。

本接口用于建立正确性、同步和成本测量基础。它不承诺简单整数 map 比 CPU 快；首次 shader 编译、逐元素跨 ABI 写入、分配、提交等待和复制读回都应计入应用总成本。MPS GEMM、显式浮点数值契约、自动选路、GPU Tile 与算子融合尚待实现。

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

`Buffer.put_all(&values)` 同步上传完整数组；`put_range(offset, &values, first, count)`
复制指定区间。`Snapshot.copy_into($&out)` 写入已有目标元素，`copy_range(first, $&out,
 offset, count)` 支持局部读回；数组长度和容量保持不变。`Buffer.ints()` 和
`Snapshot.ints()` 继续返回独立数组，内部采用批量读回。原有 `put`、`word` 等标量
方法保留，运行时 ABI 查询为 2。

批量入口先检查句柄、完整范围、大小溢出与 busy/命令失败状态，再执行复制；失败时
目标内容不变。零长度复制仍检查句柄及 busy 状态。上传在返回前完成，不让 GPU 保留
Xu 数组指针；GPU fence 等待规则不变。快照仍是独立、不可变的副本。批量 API 的收益
应以包含数组准备、分配、上传、提交等待、读回和消费结果的应用总成本评估，不能只用
GPU 内核时间判断是否优于 CPU。
