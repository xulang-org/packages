# xu-mqtt (MQTT 3.1.1 IoT & Messaging Driver for Xu)

`packages/mqtt` 是为 Xu 语言构建的高性能、纯二进制报文格式的 MQTT 3.1.1 协议驱动库，适用于物联网边缘设备、移动端即时推送与消息总线，通过 `xu.xun` 外部包机制引入。

## 特性

- **零依赖纯二进制实现**：纯 Xu 实现 MQTT 变长剩余长度（Variable Length Quantity）与固定/可变报头封包；
- **全会话生命周期**：支持 `CONNECT` 握手与 `CONNACK` 返回码校验、`DISCONNECT` 优雅断开；
- **消息发布**：支持 `PUBLISH` QoS 0 与 QoS 1（带 `PUBACK` 确认）；
- **主题订阅**：支持 `SUBSCRIBE` 与 `SUBACK` 确认；
- **心跳保活**：全自动 `PINGREQ` / `PINGRESP` 链路存活探测。

## 安装与使用

在项目 `xu.xun` 中声明：

```yaml
module: my_project
require:
  "github.com/xulang-org/packages": file:../packages
```

在源文件中使用：

```xu
@main

mqtt #github.com/xulang-org/packages/mqtt

fn main() int {
    // 1. 连接 MQTT Broker
    $client := mqtt.open("127.0.0.1:1883", "xu_iot_node_1") or >>

    // 2. 心跳探测
    if client.ping() {
        println("Broker online")
    }

    // 3. 订阅主题
    client.subscribe("sensor/temperature", 0) or >>

    // 4. 发布遥测数据 (QoS 0)
    client.publish("sensor/temperature", "{\"temp\": 23.5, \"unit\": \"C\"}", 0) or >>

    // 5. 发布重要控制指令 (QoS 1，带 PUBACK 确认)
    client.publish("device/command", "{\"action\": \"reboot\"}", 1) or >>

    // 6. 优雅断开
    client.disconnect()
    << 0
}
```


## TLS

使用 `tls_connect(endpoint, default_port)` 建立带系统根证书验证和 SNI 的 TLS 连接；需要自定义 CA 时使用 `tls_connect_config(endpoint, default_port, cfg)`；返回的 `TlsClient` 提供 `read_exact`、`write_all` 和 `close`，协议帧可在同一会话上收发。TLS 握手、证书错误和超时都会直接返回错误，不会回退到明文。
