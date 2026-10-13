# SQLite 驱动

通过 `require:` 引入的 `#database/sql/driver` 适配器。每个连接持有一个原生 SQLite 数据库句柄，SQL 操作共享同一个数据库会话；`:memory:`、提交和回滚因此具有跨操作语义。

## 运行依赖与迁移

`2.0.0` 通过 Xu 的 `xu-c-1` 原生模块直接封装官方 SQLite C API，静态链接 SQLite 3.53.4。运行时不需要 Python、sqlite3 CLI 或数据库子进程。包内提供 macOS arm64/x64 两个原生切片；当前 Xu 一步原生链接支持这两个目标，其他平台需要目标工具链和外部链接流程，尚未验证。

原生资源通过内部整数句柄管理，文本和二进制结果先复制再交给 Xu。必须调用 `close` 释放数据库；它可重复调用。数据库打不开时，`ping`、`exec`、`query` 返回错误，连接无效。

从 1.0.0 迁移仍使用相同的 `database/sql/driver` 方法。内部会话结构和部署依赖发生变化，因此模块升级为 2.0.0。重新生成原生库：`python3 sqlite/native/build.py`；Python 仅用于现有原生打包工具，应用执行 SQL 不调用 Python。源码和下载校验值在 `vendor/source.json`。

`SqliteTx` 新增事务标识与错误状态。应用通过 `begin`/`begin_tx` 获得事务，不应手工构造事务结构。

## 引入

```xun
module: my_project
require:
  "github.com/xulang-org/packages": file:../path/to/packages
```

## 使用

```xu
@main
sqlite #github.com/xulang-org/packages/sqlite
#database/sql/driver

fn main() int {
    conn := sqlite.open(":memory:")
    created := conn.exec("CREATE TABLE users (name TEXT)", []driver.NamedValue{})
    if created.err.len > 0 { println(created.err)
        conn.close()
        << 1 }
    tx := conn.begin()
    args := [driver.NamedValue{name: "", ordinal: 1, value: "Alice"}]
    inserted := conn.exec("INSERT INTO users VALUES (?)", args)
    if inserted.err.len > 0 { tx.rollback()
        conn.close()
        << 2 }
    if tx.commit().len > 0 { conn.close()
        << 3 }
    conn.close()
    << 0
}
```

标准库入口是 `sql.open(sqlite.driver(), "app.db")`，不是按驱动名称注册。`prepare` 保存 SQL，执行时委托 SQLite 校验；`num_input()` 返回 `-1`，表示参数数量由后端确定。

## 数据与事务

- 参数由 SQLite 绑定，支持 `?` 和命名参数；命名参数使用不含前缀的 `NamedValue.name`，不能混用命名与位置参数。
- `NamedValue.value` 是一次性 `temp`：每个实参只转换一次，SQL 中重复名称或不同前缀的同名参数复用转换结果；调用方不应再次使用已消费的参数盒子。
- 整数、浮点、布尔、字符串、字节数组按类型传输；时间按 RFC 3339 文本存储。结果沿用 driver 的 `i:`/`f:`/`s:`/`y:`/`n:` 编码，保留空格、换行、NULL 与二进制。空结果仍带列名。
- `begin_tx` 支持默认隔离级别 `0` 和 Serializable `6`，其他级别明确报错；只读事务使用 `PRAGMA query_only`。
- 失败的 begin 不允许后续操作假装在事务中执行；结束失败事务或 `reset_session` 后可恢复。已结束的事务不能再次提交。重置连接会回滚未提交事务并恢复写入权限。
- 单个输入/结果缓冲区和编码后的结果快照限制为 16 MiB。一次调用只允许一条 SQL 语句，可附加空语句和注释；拒绝内嵌 NUL 的 SQL/路径，文本参数可含 NUL。结果是内存快照。
- 原生桥接当前通过一个资源锁串行执行 SQLite C API，保证关闭与访问安全；独立连接暂不能并行执行 C API。SQLite 文件锁的忙等待设为 5 秒；桥接资源锁的等待暂时没有超时。Context 方法当前不提供中途取消正在执行的 SQLite 操作。

在仓库根运行 `make packages-test` 验证行为，无需外部数据库服务。
