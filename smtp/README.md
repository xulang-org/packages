# packages/smtp

Xu 语言原生 SMTP 邮件传输协议客户端驱动，严格遵循 RFC 5321 规范。

## 特性

- **标准握手与问候**：支持 220 欢迎响应解析与 EHLO 多行响应（`-` 连字符多行解析）；
- **身份认证**：支持 `AUTH LOGIN` 与 Base64 凭据安全认证；
- **邮件事务与信封**：支持 `MAIL FROM`、`RCPT TO`、`DATA` 传输与 MIME 头部封装（From, To, Subject, Content-Type）；
- **高并发连接池**：内置 `SmtpPool` 线程安全借还复用与优雅关闭。

## 快速使用

```xu
smtp #github.com/xulang-org/packages/smtp

mut p := smtp.pool("smtp://127.0.0.1:1025", 5)

// 发送邮件
msg := smtp.Message{
    from: "noreply@xu.lang",
    to: "developer@example.com",
    subject: "Hello from Xu SMTP Driver",
    body: "This is a native SMTP test from Xu programming language."
}
p.send_mail(msg) or panic

// 释放连接池
p.close()
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
