#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPTCP-aware SDN Controller with LSTM-based Performance Optimization
基于LSTM深度学习的MPTCP感知SDN控制器

作者: AI Assistant
版本: 1.0
描述: 实现SDN控制器 + MPTCP + 深度学习LSTM的完整解决方案
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """网络性能指标"""
    bandwidth_utilization: float
    latency: float
    packet_loss_rate: float
    throughput: float
    congestion_window_size: int
    rtt: float
    subflow_count: int
    path_diversity: float

@dataclass
class MPTCPConnection:
    """MPTCP连接信息"""
    connection_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    token: str
    subflows: List[Dict]
    primary_path: str
    backup_paths: List[str]
    total_bytes: int
    active: bool

@dataclass
class PathInfo:
    """路径信息"""
    path_id: str
    switches: List[str]
    links: List[Tuple[str, str]]
    bandwidth: float
    delay: float
    loss_rate: float
    utilization: float
    available: bool

class LSTMNetworkPredictor(nn.Module):
    """LSTM网络性能预测模型"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=4):
        super(LSTMNetworkPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最后时间步的输出
        out = self.fc1(lstm_out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return self.sigmoid(out)

class PathSelectionLSTM(nn.Module):
    """路径选择LSTM模型"""
    
    def __init__(self, input_size=8, hidden_size=32, num_layers=1, num_paths=8):
        super(PathSelectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_paths)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        path_scores = self.fc(lstm_out[:, -1, :])
        return self.softmax(path_scores)

class CongestionPredictionLSTM(nn.Module):
    """拥塞预测LSTM模型"""
    
    def __init__(self, input_size=8, hidden_size=24, num_layers=1):
        super(CongestionPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        congestion_prob = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(congestion_prob)

class NetworkDataCollector:
    """网络数据收集器"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.path_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.connection_metrics = defaultdict(lambda: deque(maxlen=max_history))
        
    def add_network_metrics(self, metrics: NetworkMetrics):
        """添加网络性能指标"""
        timestamp = time.time()
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
    def add_path_metrics(self, path_id: str, metrics: Dict):
        """添加路径性能指标"""
        timestamp = time.time()
        self.path_metrics[path_id].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
    def get_feature_vector(self, window_size=10) -> np.ndarray:
        """获取特征向量用于LSTM输入"""
        if len(self.metrics_history) < window_size:
            return None
            
        features = []
        for i in range(-window_size, 0):
            metrics = self.metrics_history[i]['metrics']
            feature = [
                metrics.bandwidth_utilization,
                metrics.latency / 100.0,  # 归一化
                metrics.packet_loss_rate,
                metrics.throughput / 1000.0,  # 归一化
                metrics.congestion_window_size / 65535.0,  # 归一化
                metrics.rtt / 200.0,  # 归一化
                metrics.subflow_count / 8.0,  # 归一化
                metrics.path_diversity
            ]
            features.append(feature)
            
        return np.array(features, dtype=np.float32)

class MPTCPConnectionManager:
    """MPTCP连接管理器"""
    
    def __init__(self):
        self.connections = {}
        self.active_connections = set()
        self.connection_paths = defaultdict(list)
        
    def create_connection(self, src_ip: str, dst_ip: str, 
                         src_port: int, dst_port: int) -> str:
        """创建MPTCP连接"""
        connection_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
        
        connection = MPTCPConnection(
            connection_id=connection_id,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            token=self._generate_token(),
            subflows=[],
            primary_path="",
            backup_paths=[],
            total_bytes=0,
            active=True
        )
        
        self.connections[connection_id] = connection
        self.active_connections.add(connection_id)
        
        logger.info(f"创建MPTCP连接: {connection_id}")
        return connection_id
    
    def add_subflow(self, connection_id: str, subflow_info: Dict):
        """添加子流"""
        if connection_id in self.connections:
            self.connections[connection_id].subflows.append(subflow_info)
            logger.info(f"添加子流到连接 {connection_id}: {subflow_info}")
    
    def _generate_token(self) -> str:
        """生成MPTCP token"""
        import hashlib
        import random
        data = f"{time.time()}{random.random()}".encode()
        return hashlib.sha256(data).hexdigest()[:8]

class IntelligentPathManager:
    """智能路径管理器"""
    
    def __init__(self):
        self.paths = {}
        self.path_selection_model = PathSelectionLSTM()
        self.congestion_model = CongestionPredictionLSTM()
        self.performance_model = LSTMNetworkPredictor()
        self.data_collector = NetworkDataCollector()
        
        # 优化器
        self.path_optimizer = optim.Adam(self.path_selection_model.parameters(), lr=0.001)
        self.congestion_optimizer = optim.Adam(self.congestion_model.parameters(), lr=0.001)
        self.performance_optimizer = optim.Adam(self.performance_model.parameters(), lr=0.001)
        
        # 加载预训练模型（如果存在）
        self._load_models()
        
    def _load_models(self):
        """加载预训练模型"""
        try:
            self.path_selection_model.load_state_dict(
                torch.load('models/path_selection_lstm.pth', map_location='cpu'))
            self.congestion_model.load_state_dict(
                torch.load('models/congestion_prediction_lstm.pth', map_location='cpu'))
            self.performance_model.load_state_dict(
                torch.load('models/performance_lstm.pth', map_location='cpu'))
            logger.info("成功加载预训练模型")
        except FileNotFoundError:
            logger.warning("未找到预训练模型，使用随机初始化")
    
    def save_models(self):
        """保存训练好的模型"""
        import os
        os.makedirs('models', exist_ok=True)
        
        torch.save(self.path_selection_model.state_dict(), 
                  'models/path_selection_lstm.pth')
        torch.save(self.congestion_model.state_dict(), 
                  'models/congestion_prediction_lstm.pth')
        torch.save(self.performance_model.state_dict(), 
                  'models/performance_lstm.pth')
        logger.info("模型已保存")
    
    def add_path(self, path_info: PathInfo):
        """添加路径信息"""
        self.paths[path_info.path_id] = path_info
        logger.info(f"添加路径: {path_info.path_id}")
    
    def select_optimal_path(self, connection_id: str, 
                          available_paths: List[str]) -> str:
        """使用LSTM模型选择最优路径"""
        if not available_paths:
            return None
            
        # 获取网络特征
        features = self.data_collector.get_feature_vector()
        if features is None:
            # 如果没有足够的历史数据，使用简单策略
            return available_paths[0]
        
        # 准备LSTM输入
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # 预测路径选择概率
        with torch.no_grad():
            self.path_selection_model.eval()
            path_probs = self.path_selection_model(features_tensor)
            
        # 选择概率最高的可用路径
        best_path_idx = 0
        best_prob = 0
        
        for i, path_id in enumerate(available_paths):
            if i < len(path_probs[0]) and path_probs[0][i] > best_prob:
                best_prob = path_probs[0][i]
                best_path_idx = i
        
        selected_path = available_paths[best_path_idx]
        logger.info(f"为连接 {connection_id} 选择路径: {selected_path} (概率: {best_prob:.3f})")
        
        return selected_path
    
    def predict_congestion(self, path_id: str) -> float:
        """预测路径拥塞概率"""
        features = self.data_collector.get_feature_vector()
        if features is None:
            return 0.0
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            self.congestion_model.eval()
            congestion_prob = self.congestion_model(features_tensor)
            
        return congestion_prob.item()
    
    def optimize_network_performance(self) -> Dict:
        """使用LSTM优化网络性能"""
        features = self.data_collector.get_feature_vector()
        if features is None:
            return {}
            
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            self.performance_model.eval()
            predictions = self.performance_model(features_tensor)
            
        # 解析预测结果
        optimization_suggestions = {
            'bandwidth_optimization': predictions[0][0].item(),
            'latency_optimization': predictions[0][1].item(),
            'throughput_optimization': predictions[0][2].item(),
            'load_balancing': predictions[0][3].item()
        }
        
        return optimization_suggestions
    
    def train_models(self, X: torch.Tensor, y_performance: torch.Tensor, 
                    y_congestion: torch.Tensor, y_paths: torch.Tensor):
        """训练所有LSTM模型"""
        # 训练性能预测模型
        self.performance_model.train()
        self.performance_optimizer.zero_grad()
        pred_performance = self.performance_model(X)
        loss_performance = nn.MSELoss()(pred_performance, y_performance)
        loss_performance.backward()
        self.performance_optimizer.step()
        
        # 训练拥塞预测模型
        self.congestion_model.train()
        self.congestion_optimizer.zero_grad()
        pred_congestion = self.congestion_model(X)
        loss_congestion = nn.BCELoss()(pred_congestion, y_congestion)
        loss_congestion.backward()
        self.congestion_optimizer.step()
        
        # 训练路径选择模型
        self.path_selection_model.train()
        self.path_optimizer.zero_grad()
        pred_paths = self.path_selection_model(X)
        loss_paths = nn.CrossEntropyLoss()(pred_paths, y_paths)
        loss_paths.backward()
        self.path_optimizer.step()
        
        return {
            'performance_loss': loss_performance.item(),
            'congestion_loss': loss_congestion.item(),
            'path_loss': loss_paths.item()
        }

class SimpleSDNController:
    """简化的SDN控制器（不依赖Ryu）"""
    
    def __init__(self):
        self.connection_manager = MPTCPConnectionManager()
        self.path_manager = IntelligentPathManager()
        self.data_collector = NetworkDataCollector()
        
        # 网络拓扑信息
        self.switches = {}
        self.links = {}
        self.hosts = {}
        self.flow_table = {}
        
        # 性能监控线程
        self.monitor_thread = None
        self.training_thread = None
        self.running = False
        
        logger.info("简化SDN控制器初始化完成")
    
    def start(self):
        """启动控制器"""
        self.running = True
        
        # 启动性能监控
        self.monitor_thread = threading.Thread(target=self._performance_monitor, daemon=True)
        self.monitor_thread.start()
        
        # 启动模型训练
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("SDN控制器已启动")
    
    def stop(self):
        """停止控制器"""
        self.running = False
        logger.info("SDN控制器已停止")
    
    def handle_packet(self, packet_info: Dict):
        """处理数据包"""
        src_ip = packet_info['src_ip']
        dst_ip = packet_info['dst_ip']
        src_port = packet_info['src_port']
        dst_port = packet_info['dst_port']
        is_mptcp = packet_info.get('is_mptcp', False)
        
        if is_mptcp:
            return self._handle_mptcp_packet(src_ip, dst_ip, src_port, dst_port)
        else:
            return self._handle_regular_packet(src_ip, dst_ip, src_port, dst_port)
    
    def _handle_mptcp_packet(self, src_ip: str, dst_ip: str, 
                           src_port: int, dst_port: int) -> Dict:
        """处理MPTCP数据包"""
        connection_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
        
        # 检查是否为新连接
        if connection_id not in self.connection_manager.connections:
            self.connection_manager.create_connection(src_ip, dst_ip, src_port, dst_port)
        
        # 获取可用路径
        available_paths = self._get_available_paths(src_ip, dst_ip)
        
        # 使用LSTM选择最优路径
        optimal_path = self.path_manager.select_optimal_path(connection_id, available_paths)
        
        # 安装流表项
        flow_rule = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'path': optimal_path,
            'action': 'forward'
        }
        
        self.flow_table[connection_id] = flow_rule
        
        # 收集性能数据
        self._collect_performance_data(connection_id)
        
        return flow_rule
    
    def _handle_regular_packet(self, src_ip: str, dst_ip: str, 
                             src_port: int, dst_port: int) -> Dict:
        """处理常规数据包"""
        # 简单转发逻辑
        flow_rule = {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'src_port': src_port,
            'dst_port': dst_port,
            'path': 'default_path',
            'action': 'forward'
        }
        
        return flow_rule
    
    def _get_available_paths(self, src_ip: str, dst_ip: str) -> List[str]:
        """获取可用路径"""
        # 模拟路径发现
        return [f"path_{i}" for i in range(1, 5)]
    
    def _collect_performance_data(self, connection_id: str):
        """收集性能数据"""
        # 模拟性能数据收集
        metrics = NetworkMetrics(
            bandwidth_utilization=np.random.uniform(0.3, 0.9),
            latency=np.random.uniform(1, 50),
            packet_loss_rate=np.random.uniform(0, 0.05),
            throughput=np.random.uniform(100, 1000),
            congestion_window_size=np.random.randint(1000, 65535),
            rtt=np.random.uniform(10, 200),
            subflow_count=np.random.randint(1, 8),
            path_diversity=np.random.uniform(0.5, 1.0)
        )
        
        self.data_collector.add_network_metrics(metrics)
    
    def _performance_monitor(self):
        """性能监控循环"""
        while self.running:
            try:
                # 获取网络优化建议
                suggestions = self.path_manager.optimize_network_performance()
                if suggestions:
                    logger.info(f"网络优化建议: {suggestions}")
                
                # 检查拥塞状态
                for path_id in [f"path_{i}" for i in range(1, 5)]:
                    congestion_prob = self.path_manager.predict_congestion(path_id)
                    if congestion_prob > 0.7:
                        logger.warning(f"路径 {path_id} 拥塞概率较高: {congestion_prob:.3f}")
                
                time.sleep(10)  # 每10秒监控一次
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
    
    def _training_loop(self):
        """模型训练循环"""
        while self.running:
            try:
                time.sleep(60)  # 每分钟训练一次
                
                if len(self.data_collector.metrics_history) < 50:
                    continue  # 数据不足，跳过训练
                
                # 准备训练数据
                self._train_models()
                
            except Exception as e:
                logger.error(f"模型训练错误: {e}")
    
    def _train_models(self):
        """训练LSTM模型"""
        logger.info("开始训练LSTM模型...")
        
        # 准备训练数据
        training_data = self._prepare_training_data()
        
        if training_data is None:
            logger.warning("训练数据不足")
            return
        
        X, y_performance, y_congestion, y_paths = training_data
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X)
        y_performance_tensor = torch.FloatTensor(y_performance)
        y_congestion_tensor = torch.FloatTensor(y_congestion)
        y_paths_tensor = torch.LongTensor(y_paths)
        
        # 训练模型
        losses = self.path_manager.train_models(
            X_tensor, y_performance_tensor, y_congestion_tensor, y_paths_tensor
        )
        
        logger.info(f"训练完成 - 损失: {losses}")
        
        # 定期保存模型
        if len(self.data_collector.metrics_history) % 100 == 0:
            self.path_manager.save_models()
    
    def _prepare_training_data(self):
        """准备训练数据"""
        if len(self.data_collector.metrics_history) < 20:
            return None
        
        X, y_performance, y_congestion, y_paths = [], [], [], []
        window_size = 10
        
        for i in range(window_size, len(self.data_collector.metrics_history)):
            # 输入特征窗口
            features = []
            for j in range(i - window_size, i):
                metrics = self.data_collector.metrics_history[j]['metrics']
                feature = [
                    metrics.bandwidth_utilization,
                    metrics.latency / 100.0,
                    metrics.packet_loss_rate,
                    metrics.throughput / 1000.0,
                    metrics.congestion_window_size / 65535.0,
                    metrics.rtt / 200.0,
                    metrics.subflow_count / 8.0,
                    metrics.path_diversity
                ]
                features.append(feature)
            
            # 目标值
            next_metrics = self.data_collector.metrics_history[i]['metrics']
            
            # 性能预测目标
            performance_target = [
                next_metrics.bandwidth_utilization,
                next_metrics.latency / 100.0,
                next_metrics.throughput / 1000.0,
                1 - next_metrics.packet_loss_rate
            ]
            
            # 拥塞预测目标
            congestion_target = [1.0 if next_metrics.packet_loss_rate > 0.02 else 0.0]
            
            # 路径选择目标（简化为随机）
            path_target = np.random.randint(0, 4)
            
            X.append(features)
            y_performance.append(performance_target)
            y_congestion.append(congestion_target)
            y_paths.append(path_target)
        
        return (np.array(X), np.array(y_performance), 
                np.array(y_congestion), np.array(y_paths))

def simulate_network_traffic(controller: SimpleSDNController, duration: int = 300):
    """模拟网络流量"""
    logger.info(f"开始模拟网络流量，持续 {duration} 秒")
    
    start_time = time.time()
    packet_count = 0
    
    while time.time() - start_time < duration:
        # 生成随机数据包
        packet_info = {
            'src_ip': f"192.168.1.{np.random.randint(1, 100)}",
            'dst_ip': f"192.168.2.{np.random.randint(1, 100)}",
            'src_port': np.random.randint(1024, 65535),
            'dst_port': np.random.choice([80, 443, 8080, 3306]),
            'is_mptcp': np.random.random() > 0.3  # 70%概率为MPTCP流量
        }
        
        # 处理数据包
        flow_rule = controller.handle_packet(packet_info)
        packet_count += 1
        
        if packet_count % 100 == 0:
            logger.info(f"已处理 {packet_count} 个数据包")
        
        # 随机间隔
        time.sleep(np.random.uniform(0.01, 0.1))
    
    logger.info(f"流量模拟完成，共处理 {packet_count} 个数据包")

def main():
    """主函数"""
    print("=" * 70)
    print("🚀 MPTCP感知SDN控制器 with LSTM深度学习优化")
    print("版本: 1.0")
    print("功能: SDN控制器 + MPTCP + 深度学习LSTM")
    print("=" * 70)
    
    # 创建控制器
    controller = SimpleSDNController()
    
    print("\n🔧 初始化组件...")
    print("✅ 数据收集器初始化完成")
    print("✅ LSTM模型初始化完成")
    print("✅ MPTCP连接管理器初始化完成")
    print("✅ 智能路径管理器初始化完成")
    
    # 启动控制器
    print("\n🚀 启动SDN控制器...")
    controller.start()
    
    # 模拟一些初始数据
    print("\n📊 生成初始网络数据...")
    for i in range(50):
        metrics = NetworkMetrics(
            bandwidth_utilization=np.random.uniform(0.3, 0.9),
            latency=np.random.uniform(1, 50),
            packet_loss_rate=np.random.uniform(0, 0.05),
            throughput=np.random.uniform(100, 1000),
            congestion_window_size=np.random.randint(1000, 65535),
            rtt=np.random.uniform(10, 200),
            subflow_count=np.random.randint(1, 8),
            path_diversity=np.random.uniform(0.5, 1.0)
        )
        controller.data_collector.add_network_metrics(metrics)
        
        if (i + 1) % 10 == 0:
            print(f"✅ 已生成 {i + 1} 条网络数据")
    
    print("\n🌐 演示功能...")
    
    # 演示MPTCP连接创建
    print("\n1. 创建MPTCP连接:")
    connection_id = controller.connection_manager.create_connection(
        "192.168.1.10", "192.168.2.20", 5000, 80
    )
    print(f"   ✅ 连接ID: {connection_id}")
    
    # 演示路径选择
    print("\n2. 智能路径选择:")
    available_paths = ["path_1", "path_2", "path_3", "path_4"]
    optimal_path = controller.path_manager.select_optimal_path(connection_id, available_paths)
    print(f"   ✅ 选择的最优路径: {optimal_path}")
    
    # 演示拥塞预测
    print("\n3. 拥塞预测:")
    for path in available_paths:
        congestion_prob = controller.path_manager.predict_congestion(path)
        print(f"   📈 {path} 拥塞概率: {congestion_prob:.3f}")
    
    # 演示性能优化
    print("\n4. 网络性能优化建议:")
    optimization = controller.path_manager.optimize_network_performance()
    for key, value in optimization.items():
        print(f"   🎯 {key}: {value:.3f}")
    
    # 演示数据包处理
    print("\n5. 数据包处理演示:")
    test_packets = [
        {'src_ip': '192.168.1.1', 'dst_ip': '192.168.2.1', 
         'src_port': 5001, 'dst_port': 80, 'is_mptcp': True},
        {'src_ip': '192.168.1.2', 'dst_ip': '192.168.2.2', 
         'src_port': 5002, 'dst_port': 443, 'is_mptcp': True},
        {'src_ip': '192.168.1.3', 'dst_ip': '192.168.2.3', 
         'src_port': 5003, 'dst_port': 8080, 'is_mptcp': False}
    ]
    
    for packet in test_packets:
        flow_rule = controller.handle_packet(packet)
        packet_type = "MPTCP" if packet['is_mptcp'] else "TCP"
        print(f"   📦 {packet_type} 包: {packet['src_ip']}:{packet['src_port']} → "
              f"{packet['dst_ip']}:{packet['dst_port']} via {flow_rule['path']}")
    
    print("\n" + "=" * 70)
    print("✅ 所有核心功能演示完成!")
    print("\n📚 使用说明:")
    print("   • 控制器支持MPTCP感知的智能路径选择")
    print("   • LSTM模型实现网络性能预测和优化")
    print("   • 自动进行拥塞预测和负载均衡")
    print("   • 支持实时模型训练和更新")
    
    print("\n🔄 要运行长时间仿真，请取消下面的注释:")
    print("   # simulate_network_traffic(controller, duration=300)")
    
    # 可选：运行流量仿真
    user_input = input("\n是否运行网络流量仿真? (y/n): ")
    if user_input.lower() == 'y':
        simulate_network_traffic(controller, duration=60)
    
    # 停止控制器
    controller.stop()
    print("\n👋 程序结束")

if __name__ == '__main__':
    main() 