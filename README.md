# 🚀 MPTCP感知SDN控制器仿真系统

基于 Mininet 的 MPTCP + SDN + 深度学习LSTM 网络仿真系统

## 🌟 系统特点

- **真实网络仿真**: 使用 Mininet 进行真实的网络环境仿真
- **MPTCP支持**: 多路径TCP连接管理和优化
- **SDN控制**: 软件定义网络架构，灵活的路径控制
- **LSTM预测**: 深度学习模型进行网络性能预测和拥塞检测
- **实时监控**: 网络状态实时监控和可视化
- **智能路径选择**: 基于AI的最优路径选择算法

## 📋 系统要求

### 操作系统

- Ubuntu 18.04+ / Debian 10+
- 需要 root 权限

### 软件依赖

- Python 3.8+
- Mininet 2.3+
- PyTorch 1.8+
- 支持MPTCP的Linux内核

## 🛠️ 安装步骤

### 1. 安装 Mininet

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install mininet

# 或者从源码安装
git clone https://github.com/mininet/mininet
cd mininet
sudo ./util/install.sh -a
```

### 2. 安装 Python 依赖

```bash
pip install torch torchvision
pip install matplotlib numpy
```

### 3. 启用 MPTCP 支持

```bash
# 检查内核是否支持MPTCP
sysctl net.mptcp.mptcp_enabled

# 如果不支持，需要安装支持MPTCP的内核
sudo apt-get install linux-image-generic-hwe-20.04
```

### 4. 下载项目代码

```bash
git clone <此项目地址>
cd SDN+MPTCP+dl
```

## 🚀 使用方法

### 启动仿真系统
```bash

# 需要root权限
sudo python3 mptcp_sdn_mininet.py
```

### 系统菜单功能

1. **🌐 显示网络拓扑信息** - 查看网络结构和状态
2. **📊 更新并显示网络统计** - 获取实时网络性能数据
3. **🔗 创建MPTCP连接** - 建立多路径TCP连接
4. **🧠 运行LSTM路径预测** - AI驱动的路径性能预测
5. **📈 实时网络监控** - 30秒实时网络状态监控
6. **🔄 智能负载均衡演示** - 动态负载分配演示
7. **🖥️ 进入Mininet CLI** - 原生Mininet命令行界面
8. **🚦 网络拥塞模拟** - 拥塞检测和处理演示

### 使用 interactive_demo.py (模拟模式)
```bash
# 不需要root权限的模拟版本
python3 interactive_demo.py
```

## 🧠 LSTM模型

### 预训练模型
系统会自动尝试加载预训练的LSTM模型：
- `trained_models/performance_model.pth` - 性能预测模型
- `trained_models/path_selection_model.pth` - 路径选择模型
- `trained_models/congestion_model.pth` - 拥塞预测模型

### 训练新模型
```bash
python3 lstm_training.py
```

## 📊 网络拓扑

系统使用如下多路径网络拓扑：

```
    h1 ────── s1 ────── s2 ────── h2
              │          │
              │          │
              s3 ────── s4
              │          │
              │          │
    h3 ────── └─────────┘ ────── h4
```

- **路径1**: h1-s1-s2-h2 (高带宽，低延迟)
- **路径2**: h1-s1-s3-s4-h2 (中等带宽，中等延迟)  
- **路径3**: h3-s3-s4-h4 (低带宽，高延迟)
- **交叉连接**: s2-s4 提供更多路径选择

## 🔧 核心功能

### 1. 网络监控

- 实时带宽测量
- 延迟检测 (ping)
- 丢包率统计
- 拥塞状态分析

### 2. LSTM预测

- 基于历史数据的性能预测
- 8维特征向量输入
- 路径质量评分输出
- 在线学习能力

### 3. SDN控制

- 流表管理
- 路径选择决策
- 负载均衡控制
- QoS保障

### 4. MPTCP管理

- 子流创建和管理
- 多路径调度
- 拥塞窗口控制
- 故障切换

## 📝 日志文件

- `mptcp_sdn_mininet.log` - Mininet仿真日志
- `mptcp_sdn_demo.log` - 模拟模式日志

## 🐛 故障排除

### 1. Mininet安装问题

```bash
# 重新安装Mininet
sudo apt-get remove mininet
sudo apt-get install mininet
```

### 2. 权限问题

```bash
# 确保以root权限运行
sudo python3 mptcp_sdn_mininet.py
```

### 3. MPTCP不支持

```bash
# 检查内核版本
uname -r
# 需要Linux 5.6+版本才完整支持MPTCP
```

### 4. PyTorch GPU支持

```bash
# 如果需要GPU加速
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 项目文件说明

- `mptcp_sdn_mininet.py` - 基于Mininet的仿真主程序
- `interactive_demo.py` - 交互式演示程序(模拟模式)
- `lstm_training.py` - LSTM模型训练程序  
- `mptcp_sdn_lstm.py` - 核心LSTM模型定义
- `拥塞度判断机制说明.md` - 拥塞检测机制详细说明
- `trained_models/` - 预训练模型存储目录

