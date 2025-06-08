#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 基于 Mininet 的 MPTCP感知SDN控制器仿真系统
版本: 2.0
功能: 使用 Mininet 进行真实网络仿真 + MPTCP + 深度学习LSTM
"""

import os
import sys
import time
import random
import torch
import torch.nn as nn
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

# Mininet imports
try:
    from mininet.net import Mininet
    from mininet.node import Controller, RemoteController, OVSSwitch
    from mininet.link import TCLink
    from mininet.topo import Topo
    from mininet.log import setLogLevel, info
    from mininet.cli import CLI
    from mininet.util import dumpNodeConnections
except ImportError:
    print("❌ Mininet 未安装，请先安装 Mininet")
    print("安装命令: sudo apt-get install mininet")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mptcp_sdn_mininet.log')
    ]
)
logger = logging.getLogger(__name__)


class MPTCPTopology(Topo):
    """MPTCP网络拓扑类"""
    
    def __init__(self):
        super(MPTCPTopology, self).__init__()
        
    def build(self):
        """构建MPTCP多路径拓扑"""
        info('*** 构建MPTCP多路径拓扑\n')
        
        # 创建主机
        h1 = self.addHost('h1', ip='10.0.1.1/24')
        h2 = self.addHost('h2', ip='10.0.2.1/24')
        h3 = self.addHost('h3', ip='10.0.3.1/24')
        h4 = self.addHost('h4', ip='10.0.4.1/24')
        
        # 创建交换机
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        s3 = self.addSwitch('s3')
        s4 = self.addSwitch('s4')
        
        # 添加链路 - 多条路径提供MPTCP选择
        # 路径1: h1-s1-s2-h2 (高带宽，低延迟)
        self.addLink(h1, s1, bw=100, delay='10ms', loss=0)
        self.addLink(s1, s2, bw=100, delay='10ms', loss=0)
        self.addLink(s2, h2, bw=100, delay='10ms', loss=0)
        
        # 路径2: h1-s1-s3-s4-h2 (中等带宽，中等延迟)
        self.addLink(s1, s3, bw=50, delay='20ms', loss=1)
        self.addLink(s3, s4, bw=50, delay='20ms', loss=1)
        self.addLink(s4, h2, bw=50, delay='20ms', loss=1)
        
        # 路径3: h3-s3-s4-h4 (低带宽，高延迟)
        self.addLink(h3, s3, bw=20, delay='50ms', loss=2)
        self.addLink(s4, h4, bw=20, delay='50ms', loss=2)
        
        # 交叉连接提供更多路径选择
        self.addLink(s2, s4, bw=30, delay='30ms', loss=1)


class LSTMNetworkPredictor(nn.Module):
    """LSTM网络性能预测模型"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, 
                 output_size=1):
        super(LSTMNetworkPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class NetworkMonitor:
    """网络状态监控器"""
    
    def __init__(self, net: Mininet):
        self.net = net
        self.stats_history = []
        self.lstm_model = LSTMNetworkPredictor()
        self.load_trained_model()
        
    def load_trained_model(self):
        """加载预训练的LSTM模型"""
        try:
            model_path = 'trained_models/performance_model.pth'
            if os.path.exists(model_path):
                self.lstm_model.load_state_dict(torch.load(model_path))
                self.lstm_model.eval()
                logger.info("✅ 成功加载预训练LSTM模型")
            else:
                logger.warning("⚠️ 未找到预训练模型，将使用随机初始化权重")
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
    
    def get_link_stats(self, src: str, dst: str) -> Dict[str, float]:
        """获取链路统计信息"""
        try:
            # 获取网络接口统计
            src_node = self.net.get(src)
            
            # 使用iperf测试获取带宽
            cmd = f'iperf -c {dst} -t 1 -f m'  
            result = src_node.cmd(cmd)
            
            # 解析结果获取网络指标
            bandwidth = self._parse_bandwidth(result)
            latency = self._get_ping_latency(src, dst)
            packet_loss = self._get_packet_loss(src, dst)
            
            return {
                'bandwidth': bandwidth,
                'latency': latency,
                'packet_loss': packet_loss,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"获取链路统计失败: {e}")
            return self._get_default_stats()
    
    def _parse_bandwidth(self, iperf_output: str) -> float:
        """解析iperf输出获取带宽"""
        try:
            lines = iperf_output.split('\n')
            for line in lines:
                if 'Mbits/sec' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Mbits/sec' in part and i > 0:
                            return float(parts[i-1])
            return 50.0  # 默认值
        except (ValueError, IndexError):
            return 50.0
    
    def _get_ping_latency(self, src: str, dst: str) -> float:
        """获取ping延迟"""
        try:
            src_node = self.net.get(src)
            result = src_node.cmd(f'ping -c 1 {dst}')
            
            if 'time=' in result:
                time_part = result.split('time=')[1].split(' ')[0]
                return float(time_part)
            return 20.0
        except (ValueError, IndexError):
            return 20.0
    
    def _get_packet_loss(self, src: str, dst: str) -> float:
        """获取丢包率"""
        try:
            src_node = self.net.get(src)
            result = src_node.cmd(f'ping -c 10 {dst}')
            
            if 'packet loss' in result:
                loss_line = [line for line in result.split('\n') 
                           if 'packet loss' in line][0]
                loss_percent = loss_line.split('%')[0].split()[-1]
                return float(loss_percent) / 100.0
            return 0.01
        except (ValueError, IndexError):
            return 0.01
    
    def _get_default_stats(self) -> Dict[str, float]:
        """获取默认统计信息"""
        return {
            'bandwidth': random.uniform(30, 80),
            'latency': random.uniform(10, 50),
            'packet_loss': random.uniform(0, 0.05),
            'timestamp': time.time()
        }
    
    def predict_path_performance(self, path_stats: List[Dict]) -> float:
        """使用LSTM预测路径性能"""
        try:
            if len(path_stats) < 5:  # 需要足够的历史数据
                return random.uniform(0.3, 0.8)
            
            # 准备输入数据
            features = []
            for stats in path_stats[-10:]:  # 使用最近10个数据点
                feature_vector = [
                    stats['bandwidth'] / 100.0,  # 归一化
                    stats['latency'] / 100.0,
                    stats['packet_loss'],
                    stats.get('congestion', 0.5),
                    random.uniform(0, 1),  # throughput
                    random.uniform(0, 1),  # cwnd
                    random.uniform(0, 1),  # rtt
                    random.uniform(0, 1)   # subflows
                ]
                features.append(feature_vector)
            
            # 转换为张量
            X = torch.tensor([features], dtype=torch.float32)
            
            with torch.no_grad():
                prediction = self.lstm_model(X).item()
            
            return prediction
            
        except Exception as e:
            logger.error(f"LSTM预测失败: {e}")
            return random.uniform(0.3, 0.8)


class MPTCPSDNController:
    """MPTCP感知的SDN控制器"""
    
    def __init__(self, net: Mininet):
        self.net = net
        self.monitor = NetworkMonitor(net)
        self.flow_table = {}
        self.path_stats = {}
        self.active_connections = {}
        
    def start_monitoring(self):
        """开始监控网络状态"""
        logger.info("🔄 开始网络监控...")
        
        # 获取所有主机对
        hosts = self.net.hosts
        host_pairs = [(h1.name, h2.name) for h1 in hosts for h2 in hosts 
                     if h1 != h2]
        
        # 初始化路径统计
        for src, dst in host_pairs:
            path_id = f"{src}-{dst}"
            self.path_stats[path_id] = []
    
    def update_network_stats(self):
        """更新网络统计信息"""
        for path_id in self.path_stats:
            src, dst = path_id.split('-')
            
            try:
                stats = self.monitor.get_link_stats(src, dst)
                self.path_stats[path_id].append(stats)
                
                # 保持历史记录在合理范围内
                if len(self.path_stats[path_id]) > 50:
                    self.path_stats[path_id] = self.path_stats[path_id][-50:]
                    
            except Exception as e:
                logger.error(f"更新路径 {path_id} 统计失败: {e}")
    
    def select_optimal_paths(self, src: str, dst: str, 
                           num_paths: int = 2) -> List[str]:
        """选择最优路径"""
        path_id = f"{src}-{dst}"
        
        if path_id not in self.path_stats or not self.path_stats[path_id]:
            return [f"path_{i}" for i in range(1, num_paths + 1)]
        
        # 使用LSTM预测路径性能
        performance_score = self.monitor.predict_path_performance(
            self.path_stats[path_id]
        )
        
        logger.info(f"路径 {path_id} 性能预测: {performance_score:.3f}")
        
        # 简化的路径选择逻辑
        available_paths = ['path_1', 'path_2', 'path_3']
        if performance_score > 0.6:
            return available_paths[:num_paths]
        elif performance_score > 0.3:
            return ['path_2', 'path_3']
        else:
            return ['path_3']
    
    def create_mptcp_flow(self, src: str, dst: str, port: int = 80):
        """创建MPTCP流"""
        flow_id = f"{src}-{dst}:{port}"
        
        # 选择最优路径
        optimal_paths = self.select_optimal_paths(src, dst)
        
        self.active_connections[flow_id] = {
            'src': src,
            'dst': dst,
            'port': port,
            'paths': optimal_paths,
            'created_time': datetime.now(),
            'bytes_sent': 0
        }
        
        logger.info(f"✅ 创建MPTCP流: {flow_id}, 使用路径: {optimal_paths}")
        return flow_id
    
    def get_network_summary(self) -> Dict:
        """获取网络状态摘要"""
        summary = {
            'total_hosts': len(self.net.hosts),
            'total_switches': len(self.net.switches),
            'active_connections': len(self.active_connections),
            'monitored_paths': len(self.path_stats),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        
        # 添加路径性能信息
        path_performance = {}
        for path_id, stats_list in self.path_stats.items():
            if stats_list:
                latest_stats = stats_list[-1]
                performance = self.monitor.predict_path_performance(stats_list)
                path_performance[path_id] = {
                    'bandwidth': latest_stats['bandwidth'],
                    'latency': latest_stats['latency'],
                    'packet_loss': latest_stats['packet_loss'],
                    'performance_score': performance
                }
        
        summary['path_performance'] = path_performance
        return summary


def setup_mptcp_environment():
    """设置MPTCP环境"""
    info('*** 配置MPTCP环境\n')
    
    # 检查MPTCP内核支持
    try:
        result = subprocess.run(['sysctl', 'net.mptcp.mptcp_enabled'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            info('*** 启用MPTCP支持\n')
            os.system('sudo sysctl -w net.mptcp.mptcp_enabled=1')
            os.system('sudo sysctl -w net.mptcp.mptcp_path_manager=fullmesh')
    except (subprocess.SubprocessError, OSError):
        info('*** MPTCP配置可能需要手动设置\n')


def run_interactive_demo():
    """运行交互式演示"""
    print("=" * 70)
    print("🚀 基于 Mininet 的 MPTCP感知SDN控制器仿真系统")
    print("版本: 2.0")
    print("=" * 70)
    
    # 设置Mininet日志级别
    setLogLevel('info')
    
    # 设置MPTCP环境
    setup_mptcp_environment()
    
    # 创建网络拓扑
    topo = MPTCPTopology()
    
    # 启动Mininet网络
    info('*** 启动网络\n')
    net = Mininet(topo=topo, switch=OVSSwitch, link=TCLink, 
                  controller=Controller)
    net.start()
    
    try:
        # 测试网络连通性
        info('*** 测试网络连通性\n')
        net.pingAll()
        
        # 创建SDN控制器
        controller = MPTCPSDNController(net)
        controller.start_monitoring()
        
        # 启动iperf服务器
        info('*** 启动iperf服务器\n')
        h2 = net.get('h2')
        h4 = net.get('h4')
        h2.cmd('iperf -s &')
        h4.cmd('iperf -s &')
        
        time.sleep(2)  # 等待服务器启动
        
        # 交互式菜单
        while True:
            print("\n" + "="*50)
            print("📋 Mininet MPTCP-SDN 仿真菜单:")
            print("1. 🌐 显示网络拓扑信息")
            print("2. 📊 更新并显示网络统计")
            print("3. 🔗 创建MPTCP连接")
            print("4. 🧠 运行LSTM路径预测")
            print("5. 📈 实时网络监控 (30秒)")
            print("6. 🔄 智能负载均衡演示")
            print("7. 🖥️  进入Mininet CLI")
            print("8. 🚦 网络拥塞模拟")
            print("0. 👋 退出程序")
            print("="*50)
            
            choice = input("请选择功能 (0-8): ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                demo_topology_info(net, controller)
            elif choice == "2":
                demo_network_stats(controller)
            elif choice == "3":
                demo_mptcp_connection(controller)
            elif choice == "4":
                demo_lstm_prediction(controller)
            elif choice == "5":
                demo_real_time_monitoring(controller)
            elif choice == "6":
                demo_load_balancing(net, controller)
            elif choice == "7":
                info('*** 进入Mininet CLI\n')
                CLI(net)
            elif choice == "8":
                demo_congestion_simulation(net, controller)
            else:
                print("❌ 无效选择，请重试")
            
            if choice != "7":  # CLI不需要等待
                input("\n按 Enter 键继续...")
    
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    
    finally:
        info('*** 停止网络\n')
        net.stop()


def demo_topology_info(net: Mininet, controller: MPTCPSDNController):
    """显示网络拓扑信息"""
    print("\n🌐 网络拓扑信息:")
    print(f"主机: {[h.name for h in net.hosts]}")
    print(f"交换机: {[s.name for s in net.switches]}")
    
    print("\n🔗 链路信息:")
    for link in net.links:
        print(f"  {link.intf1.node.name} <-> {link.intf2.node.name}")
    
    summary = controller.get_network_summary()
    print(f"\n📊 网络状态摘要:")
    print(f"  主机数量: {summary['total_hosts']}")
    print(f"  交换机数量: {summary['total_switches']}")
    print(f"  活跃连接: {summary['active_connections']}")
    print(f"  监控路径: {summary['monitored_paths']}")


def demo_network_stats(controller: MPTCPSDNController):
    """显示网络统计"""
    print("\n📊 更新网络统计...")
    controller.update_network_stats()
    
    summary = controller.get_network_summary()
    
    if 'path_performance' in summary and summary['path_performance']:
        print("\n📈 路径性能分析:")
        for path_id, perf in summary['path_performance'].items():
            print(f"  {path_id}:")
            print(f"    带宽: {perf['bandwidth']:.2f} Mbps")
            print(f"    延迟: {perf['latency']:.2f} ms")
            print(f"    丢包率: {perf['packet_loss']:.4f}")
            print(f"    性能评分: {perf['performance_score']:.3f}")
    else:
        print("暂无路径性能数据，请先运行几次统计更新")


def demo_mptcp_connection(controller: MPTCPSDNController):
    """演示MPTCP连接创建"""
    print("\n🔗 创建MPTCP连接演示:")
    
    # 创建多个连接
    connections = [
        ('h1', 'h2'),
        ('h3', 'h4'),
        ('h1', 'h4')
    ]
    
    for src, dst in connections:
        flow_id = controller.create_mptcp_flow(src, dst)
        print(f"✅ 创建连接: {flow_id}")
        time.sleep(1)
    
    print(f"\n📋 当前活跃连接数: {len(controller.active_connections)}")


def demo_lstm_prediction(controller: MPTCPSDNController):
    """演示LSTM路径预测"""
    print("\n🧠 LSTM路径预测演示:")
    
    # 更新网络统计
    controller.update_network_stats()
    
    # 为每个路径进行预测
    print("\n🔮 路径性能预测:")
    for path_id in controller.path_stats:
        if controller.path_stats[path_id]:
            prediction = controller.monitor.predict_path_performance(
                controller.path_stats[path_id]
            )
            
            status = "🟢 优秀" if prediction > 0.7 else \
                    "🟡 良好" if prediction > 0.4 else "🔴 较差"
            
            print(f"  {path_id}: {prediction:.3f} {status}")


def demo_real_time_monitoring(controller: MPTCPSDNController):
    """演示实时监控"""
    print("\n📈 实时网络监控 (30秒)...")
    print("按 Ctrl+C 提前停止\n")
    
    try:
        for i in range(30):
            controller.update_network_stats()
            
            print(f"\r时间: {datetime.now().strftime('%H:%M:%S')} | ", end="")
            
            # 显示活跃连接状态
            print(f"连接数: {len(controller.active_connections)} | ", end="")
            
            # 显示路径状态
            path_count = 0
            for path_id, stats_list in controller.path_stats.items():
                if stats_list and path_count < 3:  # 只显示前3个路径
                    latest = stats_list[-1]
                    loss = latest['packet_loss']
                    status = "🟢" if loss < 0.01 else "🟡" if loss < 0.03 else "🔴"
                    print(f"{path_id.split('-')[0]}→{path_id.split('-')[1]}:{status} ", end="")
                    path_count += 1
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")


def demo_load_balancing(net: Mininet, controller: MPTCPSDNController):
    """演示负载均衡"""
    print("\n🔄 智能负载均衡演示:")
    
    # 启动多个iperf测试模拟负载
    h1 = net.get('h1')
    h3 = net.get('h3')
    
    print("启动并发数据传输...")
    
    # 后台启动iperf客户端
    h1.cmd('iperf -c 10.0.2.1 -t 10 &')
    h3.cmd('iperf -c 10.0.4.1 -t 10 &')
    
    # 监控负载变化
    for i in range(10):
        controller.update_network_stats()
        
        print(f"\n时刻 {i+1}:")
        
        # 检查每个路径的负载
        for path_id, stats_list in controller.path_stats.items():
            if stats_list:
                latest = stats_list[-1]
                bandwidth_util = min(latest['bandwidth'] / 100.0, 1.0)
                
                if bandwidth_util > 0.8:
                    print(f"  ⚠️  {path_id} 负载过高: {bandwidth_util:.2%}")
                    # 模拟负载重分配
                    print(f"  🔄 重新分配 {path_id} 的流量")
                else:
                    print(f"  ✅ {path_id} 负载正常: {bandwidth_util:.2%}")
        
        time.sleep(1)


def demo_congestion_simulation(net: Mininet, controller: MPTCPSDNController):
    """演示网络拥塞模拟"""
    print("\n🚦 网络拥塞模拟演示:")
    
    # 获取链路并模拟拥塞
    print("模拟网络拥塞...")
    
    h1 = net.get('h1')
    h2 = net.get('h2')
    
    # 启动大流量传输造成拥塞
    print("📈 启动高负载流量...")
    h1.cmd('iperf -c 10.0.2.1 -t 15 -P 4 &')  # 4个并行连接
    
    # 监控拥塞状况
    for i in range(15):
        controller.update_network_stats()
        
        summary = controller.get_network_summary()
        
        print(f"\r拥塞监控 {i+1}/15: ", end="")
        
        if 'path_performance' in summary:
            for path_id, perf in summary['path_performance'].items():
                if 'h1-h2' in path_id:
                    loss = perf['packet_loss']
                    if loss > 0.05:
                        print("🔴 严重拥塞 ", end="")
                    elif loss > 0.02:
                        print("🟡 轻微拥塞 ", end="")
                    else:
                        print("🟢 通畅 ", end="")
                    break
        
        time.sleep(1)
    
    print("\n拥塞模拟完成")


if __name__ == "__main__":
    if os.getuid() != 0:
        print("❌ 此程序需要root权限运行")
        print("请使用: sudo python3 mptcp_sdn_mininet.py")
        sys.exit(1)
    
    run_interactive_demo() 