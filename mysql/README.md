# xu-mysql (MySQL / MariaDB Driver for Xu)

`packages/mysql` 是为 Xu 语言生态打造的纯 Xu 实现的 MySQL / MariaDB 协议驱动包，符合标准库 `#database/sql/driver` 规范，通过 `xu.xun` 外部包机制引入。

## 特性

- **零依赖纯原生**：纯 Xu 原生实现 MySQL wire text protocol 与握手，无需 CGO 或 libmysqlclient；
- **标准数据库接口**：接入标准库 `#database/sql` 抽象体系（`sql.open(mysql.driver(), dsn)`）；
- **动态驱动参数处理**：驱动按 `[]driver.NamedValue` 遍历参数；`database/sql` 高层调用的参数数量仍受其自身实现限制；
- **认证支持**：完整支持 `mysql_native_password` 密码握手认证；
- **数据库操作**：支持 CRUD、事务（`begin`/`commit`/`rollback`）、预编译语句（`prepare`）和连接探测（`ping`）；
- **丰富 DSN 格式**：支持 Go 风格 `user:pass@tcp(host:port)/db` 及标准 URL `mysql://user:pass@host:port/db`。

## 安装与使用

在你的项目 `xu.xun` 中声明：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

在源文件中使用：

```xu
@main

#database/sql
mysql #github.com/xulang-org/packages/mysql

fn main() int {
    dsn := "root:123456@tcp(127.0.0.1:3306)/my_db"
    mut db := sql.open(mysql.driver(), dsn)
    db.ping() or (err) {
        println("Ping failed: " + err.msg())
        << 1
    }

    db.exec("INSERT INTO users (name, age) VALUES (?, ?)", "Ada", 25) or (err) {
        println(err.msg())
        << 1
    }

    mut rows := db.query("SELECT id, name FROM users WHERE age >= ?", 18) or (err) {
        println(err.msg())
        << 1
    }
    while rows.next() {
        vals := rows.take() or (err) {
            println(err.msg())
            << 1
        }
        println(vals[1] as string)
    }
    rows.close()
    db.close() or (err) {
        println(err.msg())
        << 1
    }
    << 0
}
```

## TLS

DSN 的 `tls=true`、`tls=required` 或 `tls=verify` 会在 MySQL 初始握手后发送 `CLIENT_SSL` 请求，并在同一 TCP 连接上升级到 TLS；服务端不声明 TLS、握手失败、证书链或主机名校验失败都会使连接打开失败，不会回退到明文。`tlsrootcert=/absolute/path/ca.pem` 可指定自签名根证书。

## 模块边界与迁移

数据库协议、认证、DSN 和 SQL 驱动实现统一位于本模块。通用连接池与事务管理由 `#database/sql` 提供，驱动契约由 `#database/sql/driver` 提供。标准库不再维护另一份 `mysql` 驱动。

从旧标准库路径 `#database/sql/mysql` 迁移时，在项目 `xu.xun` 声明上面的依赖，并将导入改为 `#github.com/xulang-org/packages/mysql`；`sql.open(mysql.driver(), dsn)` 的调用方式保持不变。

驱动使用根 packages 模块的版本，与 Xu 产品版本独立。
