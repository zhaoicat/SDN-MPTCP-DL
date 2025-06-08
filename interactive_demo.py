#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 MPTCP感知SDN控制器 - 交互式体验程序
版本: 1.0
功能: 逐步体验 SDN控制器 + MPTCP + 深度学习LSTM
"""

import os
import time
import random
import torch
import torch.nn as nn
import logging
from datetime import datetime
from typing import Dict, List

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

class NetworkState:
    """网络状态类"""
    def __init__(self):
        self.paths = ['path_1', 'path_2', 'path_3', 'path_4']
        self.bandwidth = {path: random.uniform(50, 100) for path in self.paths}
        self.latency = {path: random.uniform(10, 50) for path in self.paths}
        self.packet_loss = {path: random.uniform(0, 0.05) for path in self.paths}
        self.congestion = {path: random.uniform(0, 1) for path in self.paths}
    
    def update(self):
        """更新网络状态"""
        for path in self.paths:
            self.bandwidth[path] += random.uniform(-5, 5)
            self.latency[path] += random.uniform(-2, 2)
            self.packet_loss[path] = max(0, self.packet_loss[path] + random.uniform(-0.01, 0.01))
            self.congestion[path] = max(0, min(1, self.congestion[path] + random.uniform(-0.1, 0.1)))
    
    def get_features(self, path: str) -> List[float]:
        """获取路径特征"""
        return [
            self.bandwidth[path],
            self.latency[path], 
            self.packet_loss[path],
            self.congestion[path]
        ]

class LSTMNetworkPredictor(nn.Module):
    """LSTM网络性能预测模型"""
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMNetworkPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
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
    """SDN控制器"""
    def __init__(self):
        self.network_state = NetworkState()
        self.lstm_model = LSTMNetworkPredictor()
        self.optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.connections = {}
        self.flow_table = {}
        self.topology = self._init_topology()
        
    def _init_topology(self):
        """初始化网络拓扑"""
        return {
            'switches': ['s1', 's2', 's3', 's4'],
            'hosts': ['h1', 'h2', 'h3', 'h4'],
            'links': [
                ('s1', 's2', 'path_1'),
                ('s1', 's3', 'path_2'), 
                ('s2', 's4', 'path_3'),
                ('s3', 's4', 'path_4')
            ]
        }
    
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
        
        for _ in range(100):
            self.network_state.update()
            for path in self.network_state.paths:
                features = self.network_state.get_features(path)
                # 计算性能分数 (简化的标签生成)
                performance_score = (features[0] / 100.0) * (1 - features[2]) * (1 - features[3])
                train_data.append(features)
                train_labels.append([performance_score])
        
        # 转换为张量
        X = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # 添加序列维度
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
        logger.info(f"训练完成 - 平均损失: {avg_loss:.4f}")
        
        return {'average_loss': avg_loss, 'final_loss': losses[-1]}

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
    print("\n📋 主菜单:")
    print("1. 🌐 查看网络拓扑和状态")
    print("2. 🔗 创建MPTCP连接")
    print("3. 🧠 训练LSTM模型")
    print("4. 📊 路径性能预测")
    print("5. 🚨 拥塞预测演示")
    print("6. 📈 实时网络监控")
    print("7. 🔄 智能路径切换演示")
    print("8. 📋 查看所有连接状态")
    print("9. 🎯 完整功能演示")
    print("0. 👋 退出程序")
    print("-" * 50)

def demo_network_topology(controller: SDNController):
    """演示网络拓扑"""
    print("\n🌐 网络拓扑信息:")
    print(f"交换机: {controller.topology['switches']}")
    print(f"主机: {controller.topology['hosts']}")
    print(f"链路: {controller.topology['links']}")
    
    print("\n📊 当前网络状态:")
    for path in controller.network_state.paths:
        state = controller.network_state
        print(f"  {path}:")
        print(f"    带宽: {state.bandwidth[path]:.2f} Mbps")
        print(f"    延迟: {state.latency[path]:.2f} ms")
        print(f"    丢包率: {state.packet_loss[path]:.4f}")
        print(f"    拥塞度: {state.congestion[path]:.4f}")

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

def demo_complete_workflow(controller: SDNController):
    """完整功能演示"""
    print("\n🎯 完整功能演示:")
    
    print("\n1️⃣ 训练LSTM模型...")
    controller.train_lstm_model(10)
    
    print("\n2️⃣ 创建多个MPTCP连接...")
    connections = []
    for i in range(3):
        src_ip = f"192.168.1.{i+1}"
        dst_ip = f"192.168.2.{i+1}"
        conn = controller.create_mptcp_connection(src_ip, dst_ip)
        connections.append(conn)
        time.sleep(1)
    
    print("\n3️⃣ 实时性能监控...")
    for i in range(5):
        controller.network_state.update()
        print(f"\n时刻 {i+1}:")
        
        # 显示每个连接的最优路径
        for j, conn in enumerate(connections):
            best_paths = controller.select_best_paths(2)
            print(f"  连接{j+1}: 推荐路径 {best_paths}")
        
        time.sleep(2)
    
    print("\n✅ 完整演示完成!")

def main():
    """主函数"""
    controller = SDNController()
    
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        try:
            choice = input("请选择功能 (0-9): ").strip()
            
            if choice == "0":
                print("\n👋 感谢使用，再见!")
                break
            elif choice == "1":
                demo_network_topology(controller)
            elif choice == "2":
                demo_mptcp_connection(controller)
            elif choice == "3":
                demo_lstm_training(controller)
            elif choice == "4":
                demo_path_prediction(controller)
            elif choice == "5":
                demo_congestion_prediction(controller)
            elif choice == "6":
                demo_real_time_monitoring(controller)
            elif choice == "7":
                demo_intelligent_path_switching(controller)
            elif choice == "8":
                demo_connection_status(controller)
            elif choice == "9":
                demo_complete_workflow(controller)
            else:
                print("\n❌ 无效选择，请重试")
            
            input("\n按 Enter 键继续...")
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被用户中断")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            input("按 Enter 键继续...")

if __name__ == "__main__":
    main() 