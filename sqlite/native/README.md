# SQLite 原生桥接

`bridge.c` 通过官方 SQLite C API 管理数据库、语句和结果。`native.json` 声明 `xu-c-1` 导出，`sqlite.xulib` 提供 macOS arm64/x64 静态切片。

跨 ABI 只传递整数资源令牌；不会将 Xu 字符串布局或裸 C 指针交给另一侧。输入采用 `SQLITE_TRANSIENT`，结果复制到独立缓冲区；关闭数据库会清理它的语句。资源令牌不重复分配，失效令牌不能访问后来创建的资源。

当前 Xu 原生浮点寄存器适配尚不完整，浮点经不受 locale 影响的十进制文本缓冲区传输，再调用 `sqlite3_bind_double`/`sqlite3_column_double`。结果使用 17 位有效数字，数据库仍保存 SQLite REAL。

在 packages 仓库根运行（通过 `XU_ROOT` 指定编译器仓库） `python3 sqlite/native/build.py` 重建；Python 只参与打包。SQLite 的 C API 资源访问当前使用全局锁串行执行。应用的限制和使用方式见 [包说明](../README.md)。
