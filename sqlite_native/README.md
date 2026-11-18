# sqlite_native (Xu FFI)

Pure-Xu SQLite driver: `dlopen` of system `libsqlite3` and integer handle
tokens. This replaces `sqlite/native/bridge.c` + `sqlite.xulib` as the default
path (`xu.xun` no longer maps `sqlite_native` under `native:`).

Resolve via `XUPATH` at the packages root, or compile from this module tree.
The legacy `sqlite/native/` tree remains only for optional xulib rebuilds.
