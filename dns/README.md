# packages/dns

Xu 语言原生 DNS 解析器驱动包，严格遵循 RFC 1035 标准。

## 特性

- **原生二进制报文解析**：支持 Question 区域域名序列化（标号长度前缀）与 12 字节头部打包；
- **Answer 记录解压与解析**：支持指针压缩（`0xC0` 压缩指针偏移计算），精准提取 Type A IPv4 地址；
- **TCP DNS 传输与连接池**：支持高并发长连接借还与连接复用。

## 快速使用

```xu
dns #github.com/xulang-org/packages/dns

mut p := dns.pool("8.8.8.8:53", 5)
ips := p.resolve_a("google.com") or panic
for ip in ips {
    println("Resolved IP: " + ip)
}
p.close()
```
