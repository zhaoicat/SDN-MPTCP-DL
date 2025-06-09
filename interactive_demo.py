#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 MPTCP感知SDN控制器 - 增强版交互式体验程序
版本: 2.0
功能: SDN控制器 + MPTCP + 深度学习LSTM + 多种网络拓扑 + 实时微调
"""

import os
import time
import random
import torch
import torch.nn as nn
import logging
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['font.family'] = 'sans-serif'
from typing import Dict, List, Tuple

# Mininet imports (可选)
try:
    from mininet.net import Mininet
    from mininet.node import Controller, RemoteController, OVSSwitch
    from mininet.link import TCLink
    from mininet.topo import Topo
    from mininet.log import setLogLevel, info
    from mininet.cli import CLI
    from mininet.util import dumpNodeConnections
    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False
    print("⚠️ Mininet 未安装，将使用模拟模式")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mptcp_sdn_demo.log')
    ]
)
logger = logging.getLogger(__name__)

# 新增拓扑类
class SimpleTopology(Topo):
    """简单6节点网络拓扑"""
    def build(self):
        # 4个主机 + 2个交换机 = 6个节点
        h1 = self.addHost('h1', ip='10.0.1.1/24')
        h2 = self.addHost('h2', ip='10.0.1.2/24')
        h3 = self.addHost('h3', ip='10.0.2.1/24')
        h4 = self.addHost('h4', ip='10.0.2.2/24')
        
        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')
        
        # 简单星形拓扑
        self.addLink(h1, s1, bw=100, delay='10ms')
        self.addLink(h2, s1, bw=100, delay='10ms')
        self.addLink(h3, s2, bw=100, delay='10ms')
        self.addLink(h4, s2, bw=100, delay='10ms')
        self.addLink(s1, s2, bw=50, delay='20ms')

class MediumTopology(Topo):
    """中等12节点网络拓扑"""
    def build(self):
        # 8个主机 + 4个交换机 = 12个节点
        hosts = []
        switches = []
        
        # 创建主机
        for i in range(8):
            host = self.addHost(f'h{i+1}', ip=f'10.0.{i+1}.1/24')
            hosts.append(host)
        
        # 创建交换机
        for i in range(4):
            switch = self.addSwitch(f's{i+1}')
            switches.append(switch)
        
        # 每个交换机连接2个主机
        for i in range(4):
            self.addLink(hosts[i*2], switches[i], bw=100, delay='10ms')
            self.addLink(hosts[i*2+1], switches[i], bw=100, delay='10ms')
        
        # 交换机之间形成环形拓扑
        for i in range(4):
            next_switch = (i + 1) % 4
            self.addLink(switches[i], switches[next_switch], 
                        bw=50, delay='15ms', loss=1)

class ComplexTopology(Topo):
    """复杂32节点网络拓扑"""
    def build(self):
        # 24个主机 + 8个交换机 = 32个节点
        hosts = []
        switches = []
        
        # 创建主机
        for i in range(24):
            host = self.addHost(f'h{i+1}', ip=f'10.0.{i//8+1}.{i%8+1}/24')
            hosts.append(host)
        
        # 创建核心交换机
        for i in range(4):
            core_switch = self.addSwitch(f'core{i+1}')
            switches.append(core_switch)
        
        # 创建边缘交换机
        for i in range(4):
            edge_switch = self.addSwitch(f'edge{i+1}')
            switches.append(edge_switch)
        
        # 每个边缘交换机连接6个主机
        for i in range(4):
            for j in range(6):
                host_idx = i * 6 + j
                self.addLink(hosts[host_idx], switches[4+i], 
                           bw=100, delay='5ms')
        
        # 边缘交换机连接到核心交换机（全互联）
        for edge_idx in range(4):
            for core_idx in range(4):
                self.addLink(switches[4+edge_idx], switches[core_idx],
                           bw=1000, delay='2ms')

class NetworkState:
    """增强版网络状态类"""
    def __init__(self, topology_type='simple'):
        self.topology_type = topology_type
        self.paths = self._init_paths()
        self.bandwidth = {path: random.uniform(50, 100) for path in self.paths}
        self.latency = {path: random.uniform(10, 50) for path in self.paths}
        self.packet_loss = {path: random.uniform(0, 0.05) for path in self.paths}
        self.congestion = {path: random.uniform(0, 1) for path in self.paths}
        self.history = []
    
    def _init_paths(self):
        """根据拓扑类型初始化路径"""
        if self.topology_type == 'simple':
            return ['path_1', 'path_2']
        elif self.topology_type == 'medium':
            return [f'path_{i+1}' for i in range(6)]
        elif self.topology_type == 'complex':
            return [f'path_{i+1}' for i in range(16)]
        else:
            return ['path_1', 'path_2', 'path_3', 'path_4']
    
    def update(self, network_change_factor=1.0):
        """更新网络状态，支持网络变化因子"""
        change_magnitude = 5 * network_change_factor
        
        for path in self.paths:
            self.bandwidth[path] += random.uniform(-change_magnitude, change_magnitude)
            self.bandwidth[path] = max(10, min(150, self.bandwidth[path]))
            
            self.latency[path] += random.uniform(-2*network_change_factor, 2*network_change_factor)
            self.latency[path] = max(5, min(100, self.latency[path]))
            
            self.packet_loss[path] = max(0, min(0.1, 
                self.packet_loss[path] + random.uniform(-0.01*network_change_factor, 
                                                       0.01*network_change_factor)))
            
            self.congestion[path] = max(0, min(1, 
                self.congestion[path] + random.uniform(-0.1*network_change_factor, 
                                                      0.1*network_change_factor)))
        
        # 记录历史数据
        self.history.append({
            'timestamp': datetime.now(),
            'bandwidth': self.bandwidth.copy(),
            'latency': self.latency.copy(),
            'packet_loss': self.packet_loss.copy(),
            'congestion': self.congestion.copy()
        })
        
        # 保持历史记录不超过1000条
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
    
    def get_features(self, path: str) -> List[float]:
        """获取路径特征（增强版）"""
        if path not in self.paths:
            return [0.0] * 8
            
        # 基础特征
        features = [
            self.bandwidth[path],
            self.latency[path], 
            self.packet_loss[path],
            self.congestion[path]
        ]
        
        # 历史趋势特征
        if len(self.history) >= 3:
            recent_bw = [h['bandwidth'][path] for h in self.history[-3:]]
            recent_lat = [h['latency'][path] for h in self.history[-3:]]
            
            bw_trend = (recent_bw[-1] - recent_bw[0]) / 3
            lat_trend = (recent_lat[-1] - recent_lat[0]) / 3
            
            features.extend([
                bw_trend,  # 带宽趋势
                lat_trend,  # 延迟趋势
                np.std(recent_bw),  # 带宽稳定性
                np.std(recent_lat)  # 延迟稳定性
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features

class LSTMNetworkPredictor(nn.Module):
    """增强版LSTM网络性能预测模型"""
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMNetworkPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM层
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 注意力机制
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)
        
        # 全连接层
        out = self.fc1(attn_out[:, -1, :])
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return self.sigmoid(out)

class MPTCPConnection:
    """MPTCP连接类"""
    def __init__(self, src_ip: str, src_port: int, dst_ip: str, dst_port: int):
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.connection_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
        self.subflows = []
        self.active_paths = []
        self.total_bytes = 0
        self.created_time = datetime.now()
        
    def add_subflow(self, path: str):
        """添加子流"""
        if path not in self.subflows:
            self.subflows.append(path)
            self.active_paths.append(path)
            logger.info(f"为连接 {self.connection_id} 添加子流: {path}")
    
    def get_status(self) -> Dict:
        """获取连接状态"""
        return {
            'connection_id': self.connection_id,
            'subflows': len(self.subflows),
            'active_paths': self.active_paths,
            'total_bytes': self.total_bytes,
            'duration': (datetime.now() - self.created_time).seconds
        }

class SDNController:
    """增强版SDN控制器"""
    def __init__(self, topology_type='simple', use_mininet=False):
        self.topology_type = topology_type
        self.use_mininet = use_mininet and MININET_AVAILABLE
        self.network_state = NetworkState(topology_type)
        self.lstm_model = LSTMNetworkPredictor()
        self.optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.connections = {}
        self.flow_table = {}
        self.topology = self._init_topology()
        self.mininet_net = None
        self.training_history = []
        self.performance_metrics = []
        
        # 实时微调相关
        self.online_learning_rate = 0.0001
        self.online_optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self.online_learning_rate)
        self.adaptation_buffer = []
        self.adaptation_threshold = 10
        
        if self.use_mininet:
            self._setup_mininet()
        
    def _init_topology(self):
        """初始化网络拓扑"""
        if self.topology_type == 'simple':
            return {
                'switches': ['s1', 's2'],
                'hosts': ['h1', 'h2', 'h3', 'h4'],
                'links': [
                    ('h1', 's1', 'path_1'),
                    ('s1', 's2', 'path_2')
                ],
                'node_count': 6
            }
        elif self.topology_type == 'medium':
            return {
                'switches': [f's{i+1}' for i in range(4)],
                'hosts': [f'h{i+1}' for i in range(8)],
                'links': [(f's{i+1}', f's{(i+1)%4+1}', f'path_{i+1}') for i in range(4)],
                'node_count': 12
            }
        elif self.topology_type == 'complex':
            return {
                'switches': [f'core{i+1}' for i in range(4)] + [f'edge{i+1}' for i in range(4)],
                'hosts': [f'h{i+1}' for i in range(24)],
                'links': [(f'core{i+1}', f'edge{j+1}', f'path_{i*4+j+1}') 
                         for i in range(4) for j in range(4)],
                'node_count': 32
            }
        else:
            return {
                'switches': ['s1', 's2', 's3', 's4'],
                'hosts': ['h1', 'h2', 'h3', 'h4'],
                'links': [
                    ('s1', 's2', 'path_1'),
                    ('s1', 's3', 'path_2'), 
                    ('s2', 's4', 'path_3'),
                    ('s3', 's4', 'path_4')
                ],
                'node_count': 8
            }
    
    def _setup_mininet(self):
        """设置Mininet网络"""
        try:
            setLogLevel('info')
            
            if self.topology_type == 'simple':
                topo = SimpleTopology()
            elif self.topology_type == 'medium':
                topo = MediumTopology()
            elif self.topology_type == 'complex':
                topo = ComplexTopology()
            else:
                topo = SimpleTopology()
            
            self.mininet_net = Mininet(topo=topo, switch=OVSSwitch, 
                                     link=TCLink, controller=Controller)
            self.mininet_net.start()
            logger.info(f"✅ Mininet网络启动成功 - {self.topology_type}拓扑")
            
        except Exception as e:
            logger.error(f"❌ Mininet启动失败: {e}")
            self.use_mininet = False
    
    def create_mptcp_connection(self, src_ip: str, dst_ip: str) -> MPTCPConnection:
        """创建MPTCP连接"""
        src_port = random.randint(1024, 65535)
        dst_port = random.choice([80, 443, 8080, 3306])
        
        connection = MPTCPConnection(src_ip, src_port, dst_ip, dst_port)
        self.connections[connection.connection_id] = connection
        
        # 使用LSTM选择最优路径
        best_paths = self.select_best_paths(num_paths=2)
        for path in best_paths:
            connection.add_subflow(path)
            
        logger.info(f"创建MPTCP连接: {connection.connection_id}")
        return connection
    
    def select_best_paths(self, num_paths: int = 2) -> List[str]:
        """使用LSTM模型选择最优路径"""
        path_scores = {}
        
        for path in self.network_state.paths:
            features = torch.tensor([self.network_state.get_features(path)], dtype=torch.float32)
            features = features.unsqueeze(0)  # 添加序列维度
            
            with torch.no_grad():
                score = self.lstm_model(features).item()
                path_scores[path] = score
        
        # 选择得分最高的路径
        sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)
        return [path for path, _ in sorted_paths[:num_paths]]
    
    def predict_congestion(self, path: str) -> float:
        """预测路径拥塞"""
        features = torch.tensor([self.network_state.get_features(path)], dtype=torch.float32)
        features = features.unsqueeze(0)
        
        with torch.no_grad():
            congestion_prob = self.lstm_model(features).item()
        
        return congestion_prob
    
    def train_lstm_model(self, num_epochs: int = 10) -> Dict[str, float]:
        """训练LSTM模型"""
        logger.info("开始训练LSTM模型...")
        
        # 生成训练数据
        train_data = []
        train_labels = []
        
        for _ in range(200):
            self.network_state.update()
            for path in self.network_state.paths:
                features = self.network_state.get_features(path)
                # 增强的性能分数计算
                bw_score = min(features[0] / 100.0, 1.0)
                latency_score = max(0, 1 - features[1] / 100.0)
                loss_score = 1 - features[2]
                congestion_score = 1 - features[3]
                
                # 考虑趋势和稳定性
                if len(features) == 8:  # 包含历史特征
                    trend_penalty = abs(features[4]) + abs(features[5])  # 变化趋势
                    stability_bonus = 1 / (1 + features[6] + features[7])  # 稳定性
                else:
                    trend_penalty = 0
                    stability_bonus = 1
                
                performance_score = (bw_score * 0.3 + latency_score * 0.3 + 
                                   loss_score * 0.2 + congestion_score * 0.2) * stability_bonus - trend_penalty * 0.1
                performance_score = max(0, min(1, performance_score))
                
                train_data.append(features)
                train_labels.append([performance_score])
        
        # 转换为张量
        X = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(train_labels, dtype=torch.float32)
        
        # 训练循环
        losses = []
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.lstm_model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(losses) / len(losses)
        
        # 记录训练历史
        training_record = {
            'timestamp': datetime.now(),
            'topology_type': self.topology_type,
            'num_epochs': num_epochs,
            'average_loss': avg_loss,
            'final_loss': losses[-1],
            'losses': losses
        }
        self.training_history.append(training_record)
        
        logger.info(f"训练完成 - 平均损失: {avg_loss:.4f}")
        
        return {'average_loss': avg_loss, 'final_loss': losses[-1], 'training_record': training_record}
    
    def online_finetune(self, path: str, actual_performance: float):
        """实时在线微调LSTM模型"""
        features = self.network_state.get_features(path)
        
        # 添加到适应缓冲区
        self.adaptation_buffer.append({
            'features': features,
            'performance': actual_performance,
            'timestamp': datetime.now()
        })
        
        # 当缓冲区达到阈值时进行微调
        if len(self.adaptation_buffer) >= self.adaptation_threshold:
            self._perform_online_update()
            self.adaptation_buffer = []  # 清空缓冲区
    
    def _perform_online_update(self):
        """执行在线更新"""
        if not self.adaptation_buffer:
            return
        
        # 准备微调数据
        adapt_data = []
        adapt_labels = []
        
        for item in self.adaptation_buffer:
            adapt_data.append(item['features'])
            adapt_labels.append([item['performance']])
        
        X = torch.tensor(adapt_data, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(adapt_labels, dtype=torch.float32)
        
        # 在线微调（只进行几次迭代）
        for _ in range(3):
            self.online_optimizer.zero_grad()
            outputs = self.lstm_model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.online_optimizer.step()
            
        logger.info(f"✅ 在线微调完成 - 使用 {len(self.adaptation_buffer)} 个样本")
    
    def simulate_network_change(self, change_type='congestion'):
        """模拟网络变化"""
        change_factor = 1.0
        
        if change_type == 'congestion':
            # 模拟拥塞
            change_factor = 2.0
            logger.info("🚨 模拟网络拥塞")
        elif change_type == 'failure':
            # 模拟链路故障
            failed_paths = random.sample(self.network_state.paths, 
                                       max(1, len(self.network_state.paths) // 4))
            for path in failed_paths:
                self.network_state.bandwidth[path] *= 0.1
                self.network_state.packet_loss[path] = min(0.5, self.network_state.packet_loss[path] * 10)
            logger.info(f"🔥 模拟链路故障: {failed_paths}")
        elif change_type == 'improvement':
            # 模拟网络改善
            change_factor = 0.5
            logger.info("📈 模拟网络改善")
        
        self.network_state.update(change_factor)
        
        # 触发实时微调
        for path in self.network_state.paths:
            # 模拟实际性能测量
            actual_perf = self.predict_congestion(path) + random.uniform(-0.1, 0.1)
            actual_perf = max(0, min(1, actual_perf))
            self.online_finetune(path, actual_perf)
    
    def get_performance_comparison(self) -> Dict:
        """获取性能对比数据"""
        current_time = datetime.now()
        
        # 收集当前性能指标
        current_metrics = {}
        for path in self.network_state.paths:
            prediction = self.predict_congestion(path)
            current_metrics[path] = {
                'bandwidth': self.network_state.bandwidth[path],
                'latency': self.network_state.latency[path],
                'packet_loss': self.network_state.packet_loss[path],
                'congestion': self.network_state.congestion[path],
                'prediction': prediction
            }
        
        performance_data = {
            'timestamp': current_time,
            'topology_type': self.topology_type,
            'node_count': self.topology['node_count'],
            'path_metrics': current_metrics,
            'training_history_count': len(self.training_history),
            'online_updates_count': len(self.adaptation_buffer),
            'use_mininet': self.use_mininet
        }
        
        self.performance_metrics.append(performance_data)
        
        # 保持性能指标历史不超过100条
        if len(self.performance_metrics) > 100:
            self.performance_metrics = self.performance_metrics[-100:]
            
        return performance_data
    
    def cleanup(self):
        """清理资源"""
        if self.use_mininet and self.mininet_net:
            try:
                self.mininet_net.stop()
                logger.info("🛑 Mininet网络已停止")
            except Exception as e:
                logger.error(f"❌ 停止Mininet时出错: {e}")

def generate_performance_plots(controller: SDNController):
    """生成性能对比图表"""
    if not controller.performance_metrics:
        print("⚠️ 没有性能数据可以绘制")
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MPTCP-SDN性能分析 - {controller.topology_type.upper()}拓扑 ({controller.topology["node_count"]}节点)', 
                 fontsize=16, fontweight='bold')
    
    # 提取时间序列数据
    timestamps = [m['timestamp'] for m in controller.performance_metrics]
    
    # 1. 带宽对比图
    ax1 = axes[0, 0]
    for path in controller.network_state.paths[:4]:  # 最多显示4条路径
        bandwidths = []
        for m in controller.performance_metrics:
            if path in m['path_metrics']:
                bandwidths.append(m['path_metrics'][path]['bandwidth'])
            else:
                bandwidths.append(0)
        ax1.plot(range(len(bandwidths)), bandwidths, label=path, linewidth=2)
    ax1.set_title('带宽变化趋势')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('带宽 (Mbps)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 延迟对比图
    ax2 = axes[0, 1]
    for path in controller.network_state.paths[:4]:
        latencies = []
        for m in controller.performance_metrics:
            if path in m['path_metrics']:
                latencies.append(m['path_metrics'][path]['latency'])
            else:
                latencies.append(0)
        ax2.plot(range(len(latencies)), latencies, label=path, linewidth=2)
    ax2.set_title('延迟变化趋势')
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('延迟 (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测准确性
    ax3 = axes[1, 0]
    if controller.training_history:
        training_losses = []
        for record in controller.training_history:
            training_losses.extend(record['losses'])
        ax3.plot(training_losses, color='red', linewidth=2)
        ax3.set_title('LSTM训练损失')
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('损失值')
        ax3.grid(True, alpha=0.3)
    
    # 4. 路径性能预测对比
    ax4 = axes[1, 1]
    if controller.performance_metrics:
        latest_metrics = controller.performance_metrics[-1]['path_metrics']
        paths = list(latest_metrics.keys())[:6]  # 最多显示6条路径
        predictions = [latest_metrics[p]['prediction'] for p in paths]
        actual_scores = []
        
        for p in paths:
            # 计算实际性能分数
            bw = latest_metrics[p]['bandwidth']
            lat = latest_metrics[p]['latency']
            loss = latest_metrics[p]['packet_loss']  
            cong = latest_metrics[p]['congestion']
            
            actual_score = (min(bw/100, 1) * 0.4 + 
                          max(0, 1-lat/100) * 0.3 + 
                          (1-loss) * 0.2 + 
                          (1-cong) * 0.1)
            actual_scores.append(actual_score)
        
        x = np.arange(len(paths))
        width = 0.35
        
        ax4.bar(x - width/2, predictions, width, label='LSTM预测', alpha=0.8)
        ax4.bar(x + width/2, actual_scores, width, label='实际性能', alpha=0.8)
        ax4.set_title('预测vs实际性能')
        ax4.set_xlabel('路径')
        ax4.set_ylabel('性能分数')
        ax4.set_xticks(x)
        ax4.set_xticklabels(paths, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'performance_analysis_{controller.topology_type}_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 性能分析图表已保存: {filename}")
    return filename

def save_experiment_results(controller: SDNController, experiment_name: str):
    """保存实验结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'experiment_info': {
            'name': experiment_name,
            'timestamp': timestamp,
            'topology_type': controller.topology_type,
            'node_count': controller.topology['node_count'],
            'use_mininet': controller.use_mininet,
            'mininet_available': MININET_AVAILABLE
        },
        'network_topology': controller.topology,
        'training_history': [
            {
                'timestamp': record['timestamp'].isoformat(),
                'topology_type': record['topology_type'],
                'num_epochs': record['num_epochs'],
                'average_loss': record['average_loss'],
                'final_loss': record['final_loss']
            }
            for record in controller.training_history
        ],
        'performance_metrics': [
            {
                'timestamp': m['timestamp'].isoformat(),
                'topology_type': m['topology_type'],
                'node_count': m['node_count'],
                'path_metrics': m['path_metrics'],
                'training_history_count': m['training_history_count'],
                'online_updates_count': m['online_updates_count'],
                'use_mininet': m['use_mininet']
            }
            for m in controller.performance_metrics
        ],
        'summary': {
            'total_training_sessions': len(controller.training_history),
            'total_online_updates': sum(len(controller.adaptation_buffer) for _ in range(1)),
            'paths_monitored': len(controller.network_state.paths),
            'experiment_duration': len(controller.performance_metrics)
        }
    }
    
    filename = f'experiment_results_{experiment_name}_{controller.topology_type}_{timestamp}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 实验结果已保存: {filename}")
    return filename

def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    """打印程序头部"""
    print("=" * 70)
    print("🚀 MPTCP感知SDN控制器 - 交互式体验程序")
    print("版本: 1.0")
    print("功能: SDN控制器 + MPTCP + 深度学习LSTM")
    print("=" * 70)

def print_menu():
    """打印主菜单"""
    print("\n📋 增强版主菜单:")
    print("1. 🏗️  选择网络拓扑 (简单/中等/复杂)")
    print("2. 🌐 查看当前网络拓扑和状态")
    print("3. 🔗 创建MPTCP连接")
    print("4. 🧠 训练LSTM模型")
    print("5. 📊 路径性能预测")
    print("6. 🚨 网络变化模拟")
    print("7. 🔄 实时LSTM微调演示")
    print("8. 📈 实时网络监控")
    print("9. 📋 查看所有连接状态")
    print("10. 🌐 生成网络拓扑图")
    print("11. 📊 生成性能对比图")
    print("12. 💾 保存实验结果")
    print("13. 🎯 完整实验流程")
    print("0. 👋 退出程序")
    print("-" * 60)

def select_topology():
    """选择网络拓扑"""
    print("\n🏗️ 选择网络拓扑:")
    print("1. 📱 简单拓扑 (6个节点)")
    print("2. 🏢 中等拓扑 (12个节点)")
    print("3. 🌐 复杂拓扑 (32个节点)")
    print("4. 🔄 使用Mininet (如果可用)")
    
    choice = input("请选择拓扑类型 (1-4): ").strip()
    use_mininet = False
    
    if choice == "1":
        topology_type = "simple"
    elif choice == "2":
        topology_type = "medium"
    elif choice == "3":
        topology_type = "complex"
    elif choice == "4":
        if MININET_AVAILABLE:
            print("选择Mininet拓扑:")
            print("1. 简单 (6节点)")
            print("2. 中等 (12节点)")
            print("3. 复杂 (32节点)")
            topo_choice = input("请选择 (1-3): ").strip()
            
            if topo_choice == "1":
                topology_type = "simple"
            elif topo_choice == "2":
                topology_type = "medium"
            elif topo_choice == "3":
                topology_type = "complex"
            else:
                topology_type = "simple"
            
            use_mininet = True
        else:
            print("❌ Mininet不可用，使用模拟模式")
            topology_type = "simple"
    else:
        topology_type = "simple"
    
    print(f"\n✅ 已选择: {topology_type.upper()}拓扑" + (" (Mininet模式)" if use_mininet else " (模拟模式)"))
    return topology_type, use_mininet

def demo_network_topology(controller: SDNController):
    """演示网络拓扑"""
    print(f"\n🌐 网络拓扑信息 - {controller.topology_type.upper()}:")
    print(f"📊 节点总数: {controller.topology['node_count']}")
    print(f"🔧 交换机: {controller.topology['switches']}")
    print(f"💻 主机: {controller.topology['hosts']}")
    print(f"🔗 链路: {controller.topology['links']}")
    print(f"⚙️  模式: {'Mininet真实仿真' if controller.use_mininet else '模拟模式'}")
    
    print(f"\n📊 当前网络状态 ({len(controller.network_state.paths)} 条路径):")
    for i, path in enumerate(controller.network_state.paths):
        state = controller.network_state
        print(f"  {path}:")
        print(f"    带宽: {state.bandwidth[path]:.2f} Mbps")
        print(f"    延迟: {state.latency[path]:.2f} ms")
        print(f"    丢包率: {state.packet_loss[path]:.4f}")
        print(f"    拥塞度: {state.congestion[path]:.4f}")
        
        # 只显示前8条路径以避免输出过长
        if i >= 7:
            remaining = len(controller.network_state.paths) - 8
            if remaining > 0:
                print(f"  ... 还有 {remaining} 条路径")
            break

def demo_mptcp_connection(controller: SDNController):
    """演示MPTCP连接创建"""
    print("\n🔗 创建MPTCP连接演示:")
    
    src_ip = f"192.168.1.{random.randint(1, 100)}"
    dst_ip = f"192.168.2.{random.randint(1, 100)}"
    
    print(f"源地址: {src_ip}")
    print(f"目标地址: {dst_ip}")
    
    connection = controller.create_mptcp_connection(src_ip, dst_ip)
    
    print(f"\n✅ 连接创建成功!")
    print(f"连接ID: {connection.connection_id}")
    print(f"子流数量: {len(connection.subflows)}")
    print(f"使用路径: {connection.active_paths}")

def demo_lstm_training(controller: SDNController):
    """演示LSTM模型训练"""
    print("\n🧠 LSTM模型训练演示:")
    print("正在准备训练数据...")
    
    # 显示训练前的预测
    print("\n训练前预测示例:")
    for path in controller.network_state.paths[:2]:
        score = controller.predict_congestion(path)
        print(f"  {path} 性能分数: {score:.4f}")
    
    # 开始训练
    print("\n开始训练...")
    results = controller.train_lstm_model(num_epochs=20)
    
    print(f"\n✅ 训练完成!")
    print(f"平均损失: {results['average_loss']:.4f}")
    print(f"最终损失: {results['final_loss']:.4f}")
    
    # 显示训练后的预测
    print("\n训练后预测示例:")
    for path in controller.network_state.paths[:2]:
        score = controller.predict_congestion(path)
        print(f"  {path} 性能分数: {score:.4f}")

def demo_path_prediction(controller: SDNController):
    """演示路径性能预测"""
    print("\n📊 路径性能预测演示:")
    
    print("正在分析所有路径...")
    time.sleep(1)
    
    predictions = {}
    for path in controller.network_state.paths:
        score = controller.predict_congestion(path)
        predictions[path] = score
    
    print("\n预测结果:")
    sorted_paths = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for i, (path, score) in enumerate(sorted_paths, 1):
        print(f"  {i}. {path}: {score:.4f} {'🟢' if score > 0.6 else '🟡' if score > 0.3 else '🔴'}")
    
    best_paths = controller.select_best_paths(2)
    print(f"\n🎯 推荐使用路径: {best_paths}")

def demo_congestion_prediction(controller: SDNController):
    """演示拥塞预测"""
    print("\n🚨 拥塞预测演示:")
    
    print("模拟网络负载变化...")
    for i in range(5):
        controller.network_state.update()
        print(f"\n时刻 {i+1}:")
        
        for path in controller.network_state.paths:
            congestion_prob = controller.predict_congestion(path)
            status = "🔴 高拥塞" if congestion_prob < 0.3 else "🟡 中等拥塞" if congestion_prob < 0.6 else "🟢 通畅"
            print(f"  {path}: {status} (预测值: {congestion_prob:.3f})")
        
        time.sleep(2)

def demo_real_time_monitoring(controller: SDNController):
    """演示实时网络监控"""
    print("\n📈 实时网络监控演示:")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        for i in range(20):
            controller.network_state.update()
            
            print(f"\r时间: {datetime.now().strftime('%H:%M:%S')} | ", end="")
            
            for path in controller.network_state.paths:
                congestion = controller.network_state.congestion[path]
                if congestion < 0.3:
                    status = "🟢"
                elif congestion < 0.6:
                    status = "🟡"
                else:
                    status = "🔴"
                print(f"{path}:{status} ", end="")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def demo_intelligent_path_switching(controller: SDNController):
    """演示智能路径切换"""
    print("\n🔄 智能路径切换演示:")
    
    # 创建一个连接
    connection = controller.create_mptcp_connection("192.168.1.10", "192.168.2.20")
    print(f"初始路径: {connection.active_paths}")
    
    print("\n模拟网络状况变化...")
    for round_num in range(3):
        print(f"\n=== 第 {round_num + 1} 轮 ===")
        
        # 模拟网络变化
        controller.network_state.update()
        
        # 重新选择最优路径
        new_paths = controller.select_best_paths(2)
        
        if set(new_paths) != set(connection.active_paths):
            print(f"检测到更优路径: {new_paths}")
            print(f"从 {connection.active_paths} 切换到 {new_paths}")
            connection.active_paths = new_paths
            print("✅ 路径切换完成")
        else:
            print("当前路径仍为最优，无需切换")
        
        time.sleep(2)

def demo_topology_visualization(controller: SDNController):
    """演示网络拓扑可视化"""
    print("\n🌐 生成网络拓扑图:")
    
    try:
        # 导入网络可视化模块
        from network_visualizer import NetworkVisualizer
        
        visualizer = NetworkVisualizer()
        
        print(f"正在生成 {controller.topology_type} 拓扑图...")
        
        if controller.topology_type == 'simple':
            fig, stats = visualizer.create_simple_topology_graph()
            filename = f"topology_{controller.topology_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        elif controller.topology_type == 'medium':
            fig, stats = visualizer.create_medium_topology_graph()
            filename = f"topology_{controller.topology_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        elif controller.topology_type == 'complex':
            fig, stats = visualizer.create_complex_topology_graph()
            filename = f"topology_{controller.topology_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        else:
            print("❌ 不支持的拓扑类型")
            return
        
        fig.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✅ 拓扑图已保存: {filename}")
        print(f"📊 拓扑统计:")
        print(f"  节点总数: {stats['total_nodes']}")
        print(f"  连接总数: {stats['total_edges']}")
        if stats['hosts']:
            print(f"  主机数量: {stats['hosts']}")
        if stats['switches']:
            print(f"  交换机数量: {stats['switches']}")
        if stats['cores']:
            print(f"  核心交换机: {stats['cores']}")
        if stats['edges']:
            print(f"  边缘交换机: {stats['edges']}")
            
    except ImportError:
        print("❌ 网络可视化模块未找到，请确保 network_visualizer.py 文件存在")
    except Exception as e:
        print(f"❌ 生成拓扑图时发生错误: {e}")

def demo_connection_status(controller: SDNController):
    """显示所有连接状态"""
    print("\n📋 连接状态总览:")
    
    if not controller.connections:
        print("当前没有活跃连接")
        return
    
    for conn_id, connection in controller.connections.items():
        status = connection.get_status()
        print(f"\n连接: {status['connection_id']}")
        print(f"  子流数量: {status['subflows']}")
        print(f"  活跃路径: {status['active_paths']}")
        print(f"  持续时间: {status['duration']} 秒")
        print(f"  传输字节: {status['total_bytes']}")

def demo_network_change_simulation(controller: SDNController):
    """网络变化模拟演示"""
    print("\n🚨 网络变化模拟演示:")
    print("1. 拥塞模拟")
    print("2. 链路故障模拟") 
    print("3. 网络改善模拟")
    
    choice = input("请选择模拟类型 (1-3): ").strip()
    
    print("\n📊 变化前网络状态:")
    for path in controller.network_state.paths[:4]:
        pred = controller.predict_congestion(path)
        print(f"  {path}: 预测性能 {pred:.3f}")
    
    # 记录变化前的性能
    controller.get_performance_comparison()
    
    if choice == "1":
        controller.simulate_network_change('congestion')
    elif choice == "2":
        controller.simulate_network_change('failure')
    elif choice == "3":
        controller.simulate_network_change('improvement')
    else:
        controller.simulate_network_change('congestion')
    
    print("\n📊 变化后网络状态:")
    for path in controller.network_state.paths[:4]:
        pred = controller.predict_congestion(path)
        print(f"  {path}: 预测性能 {pred:.3f}")
    
    # 记录变化后的性能
    controller.get_performance_comparison()

def demo_online_finetune(controller: SDNController):
    """实时LSTM微调演示"""
    print("\n🔄 实时LSTM微调演示:")
    print("模拟网络变化并进行实时模型适应...")
    
    print("\n初始预测性能:")
    initial_predictions = {}
    for path in controller.network_state.paths[:4]:
        pred = controller.predict_congestion(path)
        initial_predictions[path] = pred
        print(f"  {path}: {pred:.4f}")
    
    print("\n开始模拟网络变化和实时微调...")
    for round_num in range(5):
        print(f"\n=== 第 {round_num + 1} 轮 ===")
        
        # 模拟网络变化
        controller.network_state.update(random.uniform(1.0, 2.0))
        
        # 为每条路径模拟实际性能测量并进行微调
        for path in controller.network_state.paths[:4]:
            # 模拟实际测量值
            actual_performance = controller.predict_congestion(path) + random.uniform(-0.15, 0.15)
            actual_performance = max(0, min(1, actual_performance))
            
            # 进行在线微调
            controller.online_finetune(path, actual_performance)
            
            print(f"  {path}: 实际性能 {actual_performance:.4f}")
        
        time.sleep(1)
    
    print("\n微调后预测性能:")
    for path in controller.network_state.paths[:4]:
        pred = controller.predict_congestion(path)
        initial = initial_predictions.get(path, 0)
        change = pred - initial
        print(f"  {path}: {pred:.4f} (变化: {change:+.4f})")
    
    print(f"\n✅ 完成 {len(controller.adaptation_buffer)} 次在线微调")

def demo_complete_experiment(controller: SDNController):
    """完整实验流程"""
    print("\n🎯 完整实验流程:")
    
    experiment_name = f"complete_experiment_{controller.topology_type}"
    
    print("\n1️⃣ 初始LSTM模型训练...")
    controller.train_lstm_model(15)
    
    print("\n2️⃣ 创建多个MPTCP连接...")
    connections = []
    for i in range(min(3, len(controller.network_state.paths) // 2)):
        src_ip = f"192.168.1.{i+1}"
        dst_ip = f"192.168.2.{i+1}"
        conn = controller.create_mptcp_connection(src_ip, dst_ip)
        connections.append(conn)
        controller.get_performance_comparison()
        time.sleep(0.5)
    
    print("\n3️⃣ 网络变化模拟...")
    change_types = ['congestion', 'failure', 'improvement']
    for change_type in change_types:
        print(f"\n模拟 {change_type}...")
        controller.simulate_network_change(change_type)
        controller.get_performance_comparison()
        time.sleep(1)
    
    print("\n4️⃣ 实时监控和适应...")
    for i in range(8):
        controller.network_state.update()
        
        # 每个连接的路径优化
        for j, conn in enumerate(connections):
            best_paths = controller.select_best_paths(min(2, len(controller.network_state.paths)))
            if set(best_paths) != set(conn.active_paths):
                print(f"  连接{j+1}: 路径从 {conn.active_paths} 切换到 {best_paths}")
                conn.active_paths = best_paths
        
        controller.get_performance_comparison()
        time.sleep(0.5)
    
    print("\n5️⃣ 生成结果...")
    
    # 生成图表
    plot_file = generate_performance_plots(controller)
    
    # 保存实验结果
    result_file = save_experiment_results(controller, experiment_name)
    
    print(f"\n✅ 完整实验完成!")
    print(f"📊 图表文件: {plot_file}")
    print(f"💾 结果文件: {result_file}")
    print(f"📈 训练次数: {len(controller.training_history)}")
    print(f"📊 性能记录: {len(controller.performance_metrics)}")
    print(f"🔄 在线更新: {len(controller.adaptation_buffer)}")

def main():
    """主函数"""
    print_header()
    print("🚀 欢迎使用MPTCP感知SDN控制器增强版!")
    
    # 初始选择拓扑
    topology_type, use_mininet = select_topology()
    controller = SDNController(topology_type, use_mininet)
    
    try:
        while True:
            clear_screen()
            print_header()
            print(f"当前拓扑: {controller.topology_type.upper()} ({controller.topology['node_count']}节点)")
            print(f"运行模式: {'Mininet真实仿真' if controller.use_mininet else '模拟模式'}")
            print_menu()
            
            try:
                choice = input("请选择功能 (0-13): ").strip()
                
                if choice == "0":
                    print("\n👋 感谢使用，再见!")
                    break
                elif choice == "1":
                    # 重新选择拓扑
                    controller.cleanup()
                    topology_type, use_mininet = select_topology()
                    controller = SDNController(topology_type, use_mininet)
                elif choice == "2":
                    demo_network_topology(controller)
                elif choice == "3":
                    demo_mptcp_connection(controller)
                elif choice == "4":
                    demo_lstm_training(controller)
                elif choice == "5":
                    demo_path_prediction(controller)
                elif choice == "6":
                    demo_network_change_simulation(controller)
                elif choice == "7":
                    demo_online_finetune(controller)
                elif choice == "8":
                    demo_real_time_monitoring(controller)
                elif choice == "9":
                    demo_connection_status(controller)
                elif choice == "10":
                    demo_topology_visualization(controller)
                elif choice == "11":
                    if controller.performance_metrics:
                        generate_performance_plots(controller)
                    else:
                        print("⚠️ 没有性能数据，请先运行其他功能收集数据")
                elif choice == "12":
                    experiment_name = input("请输入实验名称: ").strip() or "manual_experiment"
                    save_experiment_results(controller, experiment_name)
                elif choice == "13":
                    demo_complete_experiment(controller)
                else:
                    print("\n❌ 无效选择，请重试")
                
                input("\n按 Enter 键继续...")
                
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                logger.error(f"程序错误: {e}", exc_info=True)
                input("按 Enter 键继续...")
    
    finally:
        # 确保清理资源
        try:
            controller.cleanup()
        except:
            pass

if __name__ == "__main__":
    main() 