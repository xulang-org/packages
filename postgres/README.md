# xu-postgres (PostgreSQL Driver for Xu)

`packages/postgres` 是为 Xu 语言构建的纯 Xu 实现的 PostgreSQL 协议驱动，独立于标准库存在，通过 `xu.xun` 的 `require:` 引入。

## 特性

- **纯 Xu 协议栈**：实现 PostgreSQL Wire Protocol 3.0 的查询和认证路径；
- **广泛认证支持**：内置 Trust、Cleartext、MD5 以及安全标准 SCRAM-SHA-256；
- **防 SQL 注入**：Extended Query Protocol（Parse / Bind / Execute / Sync），参数不拼装进 SQL；
- **标准驱动**：适配 `#database/sql/driver`，配合 `#database/sql` 连接池与事务；
- **直接连接**：`postgres.open(dsn)` 返回 `driver.Conn`，接收 `[]driver.NamedValue` 参数。

## 安装与引入

在你的项目 `xu.xun` 中声明：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

## 使用示例

### 配合 `#database/sql` 标准库使用

```xu
@main

#database/sql
postgres #github.com/xulang-org/packages/postgres

fn main() int {
    dsn := "postgres://postgres:postgres@127.0.0.1:5432/xu_demo?sslmode=disable"
    mut db := sql.open(postgres.driver(), dsn)
    db.ping() or (err) {
        println(err.msg())
        << 1
    }

    db.exec("CREATE TABLE IF NOT EXISTS demo (id INT, name TEXT)") or (err) {
        println(err.msg())
        << 1
    }
    db.exec("INSERT INTO demo (id, name) VALUES ($$1, $$2)", 1, "Xu") or (err) {
        println(err.msg())
        << 1
    }

    mut rows := db.query("SELECT id, name FROM demo") or (err) {
        println(err.msg())
        << 1
    }
    while rows.next() {
        cells := rows.take() or (err) {
            println(err.msg())
            << 1
        }
        println(str(cells[0] as int) + ": " + (cells[1] as string))
    }
    rows.close()
    db.close() or (err) {
        println(err.msg())
        << 1
    }
    << 0
}
```

## 模块边界与迁移

数据库协议、认证、DSN 和 SQL 驱动实现统一位于本模块。通用连接池与事务管理由 `#database/sql` 提供，驱动契约由 `#database/sql/driver` 提供。标准库不再维护另一份 `postgres` 驱动。

Xu 1.0.0 移除了 `#database/sql/postgres`。迁移时，在项目 `xu.xun` 声明上面的依赖，并将导入改为 `#github.com/xulang-org/packages/postgres`；`sql.open(postgres.driver(), dsn)` 的调用方式保持不变。首版驱动不提供 TLS。
