#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPTCP SDN LSTM模型训练脚本
专门用于训练LSTM神经网络模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from dataclasses import dataclass
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

class NetworkDataGenerator:
    """网络数据生成器"""
    
    def __init__(self):
        self.rng = np.random.RandomState(42)  # 固定随机种子确保可重现
        
    def generate_synthetic_data(self, num_samples=1000, sequence_length=10):
        """生成合成网络数据"""
        logger.info(f"生成 {num_samples} 个样本，序列长度: {sequence_length}")
        
        X, y_performance, y_congestion, y_paths = [], [], [], []
        
        for i in range(num_samples):
            # 生成一个序列的网络状态
            sequence = []
            for t in range(sequence_length):
                # 模拟时间相关的网络状态变化
                base_utilization = 0.5 + 0.3 * np.sin(2 * np.pi * t / 10)
                noise = self.rng.normal(0, 0.1)
                
                metrics = [
                    min(1.0, max(0.0, base_utilization + noise)),  # bandwidth_utilization
                    self.rng.uniform(5, 50),  # latency
                    self.rng.exponential(0.01),  # packet_loss_rate
                    self.rng.uniform(100, 1000),  # throughput
                    self.rng.uniform(0.1, 1.0),  # congestion_window_size (normalized)
                    self.rng.uniform(10, 200),  # rtt
                    self.rng.randint(1, 9),  # subflow_count
                    self.rng.uniform(0.5, 1.0)  # path_diversity
                ]
                
                # 归一化处理
                normalized_metrics = [
                    metrics[0],  # bandwidth_utilization 已归一化
                    metrics[1] / 100.0,  # latency 归一化
                    min(1.0, metrics[2]),  # packet_loss_rate
                    metrics[3] / 1000.0,  # throughput 归一化
                    metrics[4],  # congestion_window_size 已归一化
                    metrics[5] / 200.0,  # rtt 归一化
                    metrics[6] / 8.0,  # subflow_count 归一化
                    metrics[7]  # path_diversity 已归一化
                ]
                
                sequence.append(normalized_metrics)
            
            X.append(sequence)
            
            # 生成目标标签
            last_metrics = sequence[-1]
            
            # 性能预测目标: [带宽利用率, 延迟, 吞吐量, 质量分数]
            quality_score = (last_metrics[0] * last_metrics[3] * 
                            (1 - last_metrics[2]) * (1 - last_metrics[1]))
            performance_target = [
                last_metrics[0],  # 带宽利用率
                last_metrics[1],  # 延迟
                last_metrics[3],  # 吞吐量
                quality_score     # 整体质量分数
            ]
            
            # 拥塞预测目标
            congestion_prob = 1.0 if last_metrics[2] > 0.02 else 0.0
            
            # 路径选择目标 (基于性能选择最优路径)
            path_target = min(7, int(quality_score * 8))
            
            y_performance.append(performance_target)
            y_congestion.append([congestion_prob])
            y_paths.append(path_target)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{num_samples} 个样本")
        
        return (np.array(X, dtype=np.float32), 
                np.array(y_performance, dtype=np.float32),
                np.array(y_congestion, dtype=np.float32), 
                np.array(y_paths, dtype=np.int64))

class LSTMTrainer:
    """LSTM模型训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        logger.info(f"使用设备: {device}")
        
        # 初始化模型
        self.performance_model = LSTMNetworkPredictor().to(device)
        self.path_model = PathSelectionLSTM().to(device)
        self.congestion_model = CongestionPredictionLSTM().to(device)
        
        # 初始化优化器
        self.performance_optimizer = optim.Adam(self.performance_model.parameters(), lr=0.001)
        self.path_optimizer = optim.Adam(self.path_model.parameters(), lr=0.001)
        self.congestion_optimizer = optim.Adam(self.congestion_model.parameters(), lr=0.001)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_history = {
            'performance_loss': [],
            'path_loss': [],
            'congestion_loss': []
        }
    
    def train_models(self, X, y_performance, y_congestion, y_paths, 
                    epochs=100, batch_size=32, validation_split=0.2):
        """训练所有LSTM模型"""
        logger.info("开始训练LSTM模型...")
        
        # 数据转换为张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_perf_tensor = torch.FloatTensor(y_performance).to(self.device)
        y_cong_tensor = torch.FloatTensor(y_congestion).to(self.device)
        y_path_tensor = torch.LongTensor(y_paths).to(self.device)
        
        # 划分训练集和验证集
        num_samples = len(X)
        val_size = int(num_samples * validation_split)
        train_size = num_samples - val_size
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 训练数据
        X_train = X_tensor[train_indices]
        y_perf_train = y_perf_tensor[train_indices]
        y_cong_train = y_cong_tensor[train_indices]
        y_path_train = y_path_tensor[train_indices]
        
        # 验证数据
        X_val = X_tensor[val_indices]
        y_perf_val = y_perf_tensor[val_indices]
        y_cong_val = y_cong_tensor[val_indices]
        y_path_val = y_path_tensor[val_indices]
        
        logger.info(f"训练集大小: {train_size}, 验证集大小: {val_size}")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            train_losses = self._train_epoch(
                X_train, y_perf_train, y_cong_train, y_path_train, batch_size
            )
            
            # 验证阶段
            val_losses = self._validate_epoch(
                X_val, y_perf_val, y_cong_val, y_path_val
            )
            
            # 记录损失
            for key in train_losses:
                self.train_history[key].append(train_losses[key])
            
            # 早停检查
            current_val_loss = sum(val_losses.values())
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_models('best_models')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  训练损失 - 性能: {train_losses['performance_loss']:.4f}, "
                          f"路径: {train_losses['path_loss']:.4f}, "
                          f"拥塞: {train_losses['congestion_loss']:.4f}")
                logger.info(f"  验证损失 - 性能: {val_losses['performance_loss']:.4f}, "
                          f"路径: {val_losses['path_loss']:.4f}, "
                          f"拥塞: {val_losses['congestion_loss']:.4f}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停于第 {epoch+1} 轮，验证损失未改善")
                break
        
        logger.info("训练完成!")
        return self.train_history
    
    def _train_epoch(self, X, y_perf, y_cong, y_path, batch_size):
        """训练一个epoch"""
        self.performance_model.train()
        self.path_model.train()
        self.congestion_model.train()
        
        total_losses = {'performance_loss': 0, 'path_loss': 0, 'congestion_loss': 0}
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        for i in range(0, len(X), batch_size):
            end_idx = min(i + batch_size, len(X))
            batch_X = X[i:end_idx]
            batch_y_perf = y_perf[i:end_idx]
            batch_y_cong = y_cong[i:end_idx]
            batch_y_path = y_path[i:end_idx]
            
            # 性能预测模型训练
            self.performance_optimizer.zero_grad()
            perf_pred = self.performance_model(batch_X)
            perf_loss = self.mse_loss(perf_pred, batch_y_perf)
            perf_loss.backward()
            self.performance_optimizer.step()
            
            # 拥塞预测模型训练
            self.congestion_optimizer.zero_grad()
            cong_pred = self.congestion_model(batch_X)
            cong_loss = self.bce_loss(cong_pred, batch_y_cong)
            cong_loss.backward()
            self.congestion_optimizer.step()
            
            # 路径选择模型训练
            self.path_optimizer.zero_grad()
            path_pred = self.path_model(batch_X)
            path_loss = self.ce_loss(path_pred, batch_y_path)
            path_loss.backward()
            self.path_optimizer.step()
            
            total_losses['performance_loss'] += perf_loss.item()
            total_losses['path_loss'] += path_loss.item()
            total_losses['congestion_loss'] += cong_loss.item()
        
        # 计算平均损失
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def _validate_epoch(self, X, y_perf, y_cong, y_path):
        """验证一个epoch"""
        self.performance_model.eval()
        self.path_model.eval()
        self.congestion_model.eval()
        
        with torch.no_grad():
            perf_pred = self.performance_model(X)
            cong_pred = self.congestion_model(X)
            path_pred = self.path_model(X)
            
            perf_loss = self.mse_loss(perf_pred, y_perf).item()
            cong_loss = self.bce_loss(cong_pred, y_cong).item()
            path_loss = self.ce_loss(path_pred, y_path).item()
        
        return {
            'performance_loss': perf_loss,
            'path_loss': path_loss,
            'congestion_loss': cong_loss
        }
    
    def save_models(self, save_dir='trained_models'):
        """保存训练好的模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.performance_model.state_dict(), 
                  os.path.join(save_dir, 'performance_lstm.pth'))
        torch.save(self.path_model.state_dict(), 
                  os.path.join(save_dir, 'path_selection_lstm.pth'))
        torch.save(self.congestion_model.state_dict(), 
                  os.path.join(save_dir, 'congestion_prediction_lstm.pth'))
        
        # 保存训练历史
        with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.train_history, f)
        
        logger.info(f"模型已保存到 {save_dir}")
    
    def load_models(self, load_dir='trained_models'):
        """加载训练好的模型"""
        try:
            self.performance_model.load_state_dict(
                torch.load(os.path.join(load_dir, 'performance_lstm.pth'), 
                          map_location=self.device))
            self.path_model.load_state_dict(
                torch.load(os.path.join(load_dir, 'path_selection_lstm.pth'), 
                          map_location=self.device))
            self.congestion_model.load_state_dict(
                torch.load(os.path.join(load_dir, 'congestion_prediction_lstm.pth'), 
                          map_location=self.device))
            
            logger.info(f"模型已从 {load_dir} 加载")
            return True
        except FileNotFoundError:
            logger.warning(f"未找到预训练模型在 {load_dir}")
            return False
    
    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 性能损失
        axes[0].plot(self.train_history['performance_loss'])
        axes[0].set_title('性能预测损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].grid(True)
        
        # 路径选择损失
        axes[1].plot(self.train_history['path_loss'])
        axes[1].set_title('路径选择损失')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('CrossEntropy Loss')
        axes[1].grid(True)
        
        # 拥塞预测损失
        axes[2].plot(self.train_history['congestion_loss'])
        axes[2].set_title('拥塞预测损失')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('BCE Loss')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        logger.info(f"训练历史图表已保存到 {save_path}")

def main():
    """主训练函数"""
    print("=" * 70)
    print("🧠 MPTCP SDN LSTM模型训练")
    print("版本: 1.0")
    print("=" * 70)
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建数据生成器
    data_generator = NetworkDataGenerator()
    
    # 生成训练数据
    print("\n📊 生成训练数据...")
    X, y_performance, y_congestion, y_paths = data_generator.generate_synthetic_data(
        num_samples=2000, sequence_length=15
    )
    
    print(f"✅ 数据生成完成:")
    print(f"   输入形状: {X.shape}")
    print(f"   性能目标形状: {y_performance.shape}")
    print(f"   拥塞目标形状: {y_congestion.shape}")
    print(f"   路径目标形状: {y_paths.shape}")
    
    # 创建训练器
    trainer = LSTMTrainer(device=device)
    
    # 开始训练
    print("\n🚀 开始训练LSTM模型...")
    start_time = time.time()
    
    training_history = trainer.train_models(
        X, y_performance, y_congestion, y_paths,
        epochs=50, batch_size=64, validation_split=0.2
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ 训练完成! 耗时: {training_time:.2f} 秒")
    
    # 保存模型
    trainer.save_models('trained_models')
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 模型评估
    print("\n📈 模型评估:")
    final_losses = {
        'performance_loss': training_history['performance_loss'][-1],
        'path_loss': training_history['path_loss'][-1],
        'congestion_loss': training_history['congestion_loss'][-1]
    }
    
    for loss_name, loss_value in final_losses.items():
        print(f"   {loss_name}: {loss_value:.4f}")
    
    print("\n🎯 训练总结:")
    print(f"   总训练时间: {training_time:.2f} 秒")
    print(f"   训练样本数: {len(X)}")
    print(f"   模型已保存到: trained_models/")
    print("   可以使用训练好的模型进行网络优化预测")

if __name__ == "__main__":
    main() 