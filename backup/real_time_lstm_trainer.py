#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔄 实时LSTM模型训练器
支持在线学习和增量训练，能够根据实时网络数据动态更新模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from collections import deque
from datetime import datetime
import json
import threading
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OnlineLSTMPredictor(nn.Module):
    """在线LSTM预测模型"""
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, output_size=1):
        super(OnlineLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class RealTimeTrainer:
    """实时LSTM训练器"""
    
    def __init__(self, model_path=None, learning_rate=0.001, 
                 buffer_size=1000, batch_size=32, update_frequency=10):
        self.model = OnlineLSTMPredictor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 在线学习参数
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        self.label_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.training_stats = {
            'total_updates': 0,
            'last_loss': 0.0,
            'avg_loss': 0.0,
            'learning_rate': learning_rate,
            'samples_processed': 0
        }
        
        # 加载预训练模型
        if model_path and self._load_model(model_path):
            logger.info("✅ 成功加载预训练模型")
        else:
            logger.info("🔄 使用随机初始化模型")
        
        # 线程安全队列
        self.data_queue = queue.Queue()
        self.is_training = False
        self.training_thread = None
    
    def _load_model(self, model_path):
        """加载预训练模型"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def save_model(self, path='real_time_model.pth'):
        """保存当前模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"模型已保存到 {path}")
    
    def add_training_sample(self, network_data, performance_target):
        """添加训练样本到缓冲区"""
        # 归一化网络数据
        normalized_data = self._normalize_network_data(network_data)
        
        self.data_buffer.append(normalized_data)
        self.label_buffer.append(performance_target)
        self.training_stats['samples_processed'] += 1
        
        # 检查是否需要触发训练
        if len(self.data_buffer) >= self.batch_size:
            if self.training_stats['samples_processed'] % self.update_frequency == 0:
                self._trigger_incremental_training()
    
    def _normalize_network_data(self, data):
        """归一化网络数据"""
        # 假设输入格式: [bandwidth, latency, packet_loss, throughput, cwnd, rtt, subflows, diversity]
        normalized = [
            min(1.0, max(0.0, data[0] / 100.0)),  # bandwidth (Mbps)
            min(1.0, data[1] / 100.0),            # latency (ms)
            min(1.0, data[2]),                    # packet_loss
            min(1.0, data[3] / 1000.0),           # throughput (Mbps)
            min(1.0, max(0.0, data[4] / 100.0)), # congestion window
            min(1.0, data[5] / 200.0),            # rtt (ms)
            min(1.0, data[6] / 8.0),              # subflows
            min(1.0, max(0.0, data[7]))          # path diversity
        ]
        return normalized
    
    def _trigger_incremental_training(self):
        """触发增量训练"""
        if not self.is_training and len(self.data_buffer) >= self.batch_size:
            # 准备训练数据
            train_data = list(self.data_buffer)[-self.batch_size:]
            train_labels = list(self.label_buffer)[-self.batch_size:]
            
            # 在单独线程中进行训练，避免阻塞主程序
            self.training_thread = threading.Thread(
                target=self._incremental_train,
                args=(train_data, train_labels)
            )
            self.training_thread.start()
    
    def _incremental_train(self, train_data, train_labels):
        """执行增量训练"""
        self.is_training = True
        
        try:
            # 转换为张量
            X = torch.FloatTensor(train_data).unsqueeze(0)  # 添加序列维度
            y = torch.FloatTensor([[label] for label in train_labels])
            
            # 训练步骤
            self.model.train()
            self.optimizer.zero_grad()
            
            predictions = self.model(X)
            loss = self.criterion(predictions, y)
            
            loss.backward()
            self.optimizer.step()
            
            # 更新统计
            self.training_stats['total_updates'] += 1
            self.training_stats['last_loss'] = loss.item()
            
            # 计算平均损失
            if self.training_stats['total_updates'] == 1:
                self.training_stats['avg_loss'] = loss.item()
            else:
                alpha = 0.1  # 指数移动平均
                self.training_stats['avg_loss'] = (
                    alpha * loss.item() + 
                    (1 - alpha) * self.training_stats['avg_loss']
                )
            
            logger.info(f"🔄 增量训练完成 - 损失: {loss.item():.4f}, "
                       f"平均损失: {self.training_stats['avg_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"增量训练失败: {e}")
        
        finally:
            self.is_training = False
    
    def predict(self, network_data):
        """进行预测"""
        self.model.eval()
        
        normalized_data = self._normalize_network_data(network_data)
        
        with torch.no_grad():
            # 将数据转换为序列格式
            X = torch.FloatTensor([normalized_data]).unsqueeze(0)
            prediction = self.model(X).item()
        
        return prediction
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return self.training_stats.copy()
    
    def reset_buffer(self):
        """重置数据缓冲区"""
        self.data_buffer.clear()
        self.label_buffer.clear()
        logger.info("数据缓冲区已重置")

class NetworkSimulator:
    """网络状态模拟器"""
    
    def __init__(self):
        self.time_step = 0
        
    def generate_network_sample(self):
        """生成网络样本"""
        self.time_step += 1
        
        # 模拟时间变化的网络状态
        base_load = 0.5 + 0.3 * np.sin(2 * np.pi * self.time_step / 100)
        noise = np.random.normal(0, 0.1)
        
        network_data = [
            max(0, min(100, 50 + base_load * 50 + noise * 10)),  # bandwidth
            max(1, 20 + np.random.exponential(10)),              # latency
            max(0, np.random.exponential(0.01)),                 # packet_loss
            max(0, min(1000, 500 + base_load * 300)),           # throughput
            max(0, min(100, 50 + np.random.normal(0, 10))),     # cwnd
            max(10, 50 + np.random.exponential(20)),            # rtt
            np.random.randint(1, 8),                             # subflows
            np.random.uniform(0.5, 1.0)                         # diversity
        ]
        
        # 基于当前状态计算性能目标
        performance = (
            (network_data[0] / 100.0) *      # 带宽利用率
            (1 - network_data[2]) *          # 低丢包率
            (network_data[3] / 1000.0) *     # 高吞吐量
            (1 - network_data[1] / 100.0)    # 低延迟
        )
        
        return network_data, performance

def run_real_time_training_demo():
    """运行实时训练演示"""
    print("🔄 实时LSTM训练演示")
    print("="*50)
    
    # 初始化训练器
    trainer = RealTimeTrainer(
        learning_rate=0.001,
        buffer_size=500,
        batch_size=16,
        update_frequency=5
    )
    
    # 初始化网络模拟器
    simulator = NetworkSimulator()
    
    print("🚀 开始实时训练...")
    print("按 Ctrl+C 停止训练\n")
    
    try:
        for step in range(200):  # 模拟200个时间步
            # 生成网络数据
            network_data, true_performance = simulator.generate_network_sample()
            
            # 使用当前模型进行预测
            predicted_performance = trainer.predict(network_data)
            
            # 添加训练样本
            trainer.add_training_sample(network_data, true_performance)
            
            # 每10步显示一次状态
            if step % 10 == 0:
                stats = trainer.get_training_stats()
                print(f"步骤 {step:3d}: "
                      f"真实值={true_performance:.3f}, "
                      f"预测值={predicted_performance:.3f}, "
                      f"误差={abs(true_performance - predicted_performance):.3f}")
                print(f"         "
                      f"训练更新={stats['total_updates']}, "
                      f"平均损失={stats['avg_loss']:.4f}, "
                      f"处理样本={stats['samples_processed']}")
                print()
            
            # 模拟实时处理延迟
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n👋 训练被用户停止")
    
    # 等待训练线程完成
    if trainer.training_thread and trainer.training_thread.is_alive():
        trainer.training_thread.join()
    
    # 保存模型
    trainer.save_model('real_time_trained_model.pth')
    
    # 生成报告
    final_stats = trainer.get_training_stats()
    
    report = {
        'training_mode': 'real_time_online',
        'timestamp': datetime.now().isoformat(),
        'final_stats': final_stats,
        'configuration': {
            'buffer_size': trainer.buffer_size,
            'batch_size': trainer.batch_size,
            'update_frequency': trainer.update_frequency
        }
    }
    
    with open('real_time_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📊 训练完成!")
    print(f"   总训练更新: {final_stats['total_updates']}")
    print(f"   处理样本数: {final_stats['samples_processed']}")
    print(f"   最终损失: {final_stats['last_loss']:.4f}")
    print(f"   平均损失: {final_stats['avg_loss']:.4f}")
    print("   📄 报告已保存到: real_time_training_report.json")
    print("   💾 模型已保存到: real_time_trained_model.pth")

if __name__ == "__main__":
    run_real_time_training_demo() 