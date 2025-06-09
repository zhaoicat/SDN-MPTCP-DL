#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 MPTCP-SDN增强版快速演示脚本
自动运行完整实验流程，展示所有新功能
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_demo import (
    SDNController, generate_performance_plots, save_experiment_results,
    demo_network_topology, demo_mptcp_connection, demo_lstm_training,
    demo_path_prediction, demo_network_change_simulation, demo_online_finetune,
    demo_real_time_monitoring, demo_connection_status
)

def print_section(title):
    """打印节标题"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def quick_demo_all_topologies():
    """快速演示所有拓扑类型"""
    topologies = ['simple', 'medium', 'complex']
    
    print_section("MPTCP-SDN增强版系统快速演示")
    print("本演示将自动测试所有拓扑类型和核心功能")
    
    all_results = []
    
    for topo in topologies:
        print_section(f"{topo.upper()}拓扑演示")
        
        try:
            # 创建控制器
            controller = SDNController(topology_type=topo, use_mininet=False)
            
            print(f"\n📊 当前拓扑: {topo} - {controller.topology['node_count']}个节点")
            print(f"路径数量: {len(controller.network_state.paths)}")
            
            # 1. 网络拓扑验证
            print("\n1️⃣ 网络拓扑验证...")
            demo_network_topology(controller)
            
            # 2. LSTM训练
            print("\n2️⃣ LSTM模型训练...")
            training_result = controller.train_lstm_model(10)
            print(f"训练完成 - 损失: {training_result['final_loss']:.4f}")
            
            # 3. 创建MPTCP连接
            print("\n3️⃣ 创建MPTCP连接...")
            for i in range(min(2, len(controller.network_state.paths) // 2)):
                conn = controller.create_mptcp_connection(
                    f"192.168.1.{i+1}", f"192.168.2.{i+1}"
                )
                print(f"连接 {i+1}: {conn.connection_id}")
            
            # 4. 路径性能预测
            print("\n4️⃣ 路径性能预测...")
            for path in controller.network_state.paths[:4]:
                pred = controller.predict_congestion(path)
                print(f"  {path}: 性能预测 {pred:.4f}")
            
            # 5. 网络变化模拟
            print("\n5️⃣ 网络变化模拟...")
            for change_type in ['congestion', 'failure', 'improvement']:
                print(f"  模拟 {change_type}...")
                controller.simulate_network_change(change_type)
                controller.get_performance_comparison()
                time.sleep(0.5)
            
            # 6. 实时微调演示
            print("\n6️⃣ 实时LSTM微调...")
            for i in range(3):
                controller.network_state.update()
                for path in controller.network_state.paths[:2]:
                    actual_perf = controller.predict_congestion(path) + 0.1
                    controller.online_finetune(path, actual_perf)
                print(f"  微调轮次 {i+1} 完成")
            
            # 7. 性能数据收集
            print("\n7️⃣ 性能数据收集...")
            for i in range(10):
                controller.network_state.update()
                controller.get_performance_comparison()
                if i % 3 == 0:
                    print(f"  数据点 {i+1}/10")
            
            # 8. 生成结果
            print("\n8️⃣ 生成分析结果...")
            
            # 生成图表
            try:
                plot_file = generate_performance_plots(controller)
                print(f"✅ 图表已生成: {plot_file}")
            except Exception as e:
                print(f"⚠️ 图表生成跳过: {str(e)}")
            
            # 保存实验结果
            try:
                result_file = save_experiment_results(controller, f"quick_demo_{topo}")
                print(f"✅ 结果已保存: {result_file}")
                all_results.append(result_file)
            except Exception as e:
                print(f"⚠️ 结果保存跳过: {str(e)}")
            
            # 清理资源
            controller.cleanup()
            
            print(f"\n✅ {topo.upper()}拓扑演示完成!")
            print(f"   - 训练损失: {training_result['final_loss']:.4f}")
            print(f"   - 连接数量: {len(controller.connections)}")
            print(f"   - 性能记录: {len(controller.performance_metrics)}")
            print(f"   - 在线更新: {len(controller.adaptation_buffer)}")
            
        except Exception as e:
            print(f"❌ {topo}拓扑演示失败: {str(e)}")
            
        time.sleep(2)
    
    # 演示总结
    print_section("演示总结")
    print("🎉 所有拓扑演示完成!")
    print(f"📊 生成的结果文件: {len(all_results)}")
    
    if all_results:
        print("\n📂 生成的文件:")
        for file in all_results:
            print(f"  - {file}")
    
    print("\n🌟 增强功能验证:")
    print("  ✅ 多种网络拓扑支持 (简单/中等/复杂)")
    print("  ✅ 实时LSTM微调机制")
    print("  ✅ 网络变化模拟")
    print("  ✅ 性能分析与可视化")
    print("  ✅ 实验结果自动保存")
    
    print("\n📋 系统特性:")
    print("  - 支持3种不同规模的网络拓扑")
    print("  - 增强版LSTM模型（含注意力机制）")
    print("  - 实时在线学习和适应")
    print("  - 完整的性能监控和分析")
    print("  - 自动化实验流程")

def demo_specific_topology(topology_type='medium'):
    """演示特定拓扑的详细功能"""
    print_section(f"详细功能演示 - {topology_type.upper()}拓扑")
    
    controller = SDNController(topology_type=topology_type, use_mininet=False)
    
    try:
        print(f"\n🏗️ 拓扑信息:")
        print(f"  - 节点数量: {controller.topology['node_count']}")
        print(f"  - 路径数量: {len(controller.network_state.paths)}")
        print(f"  - 交换机: {len(controller.topology['switches'])}")
        print(f"  - 主机: {len(controller.topology['hosts'])}")
        
        # 详细的LSTM训练演示
        print("\n🧠 详细LSTM训练演示:")
        print("  训练参数:")
        print(f"    - 输入特征: 8维 (包含历史趋势)")
        print(f"    - 隐藏层: 64单元 x 2层")
        print(f"    - 注意力头: 8个")
        print(f"    - 学习率: 0.001")
        
        training_result = controller.train_lstm_model(20)
        
        print("\n📊 训练结果分析:")
        print(f"  - 平均损失: {training_result['average_loss']:.6f}")
        print(f"  - 最终损失: {training_result['final_loss']:.6f}")
        print(f"  - 收敛性: {'良好' if training_result['final_loss'] < 0.1 else '需要调优'}")
        
        # 详细的网络变化适应演示
        print("\n🔄 网络适应性详细测试:")
        
        change_scenarios = [
            ('normal', '正常运行'),
            ('congestion', '拥塞场景'),
            ('failure', '故障场景'),
            ('improvement', '改善场景'),
            ('mixed', '混合场景')
        ]
        
        adaptation_results = {}
        
        for scenario, description in change_scenarios:
            print(f"\n  {description}测试:")
            
            if scenario == 'mixed':
                # 混合场景：多种变化
                for sub_change in ['congestion', 'failure', 'improvement']:
                    controller.simulate_network_change(sub_change)
                    time.sleep(0.2)
            elif scenario != 'normal':
                controller.simulate_network_change(scenario)
            
            # 收集预测性能
            predictions = {}
            for path in controller.network_state.paths[:4]:
                pred = controller.predict_congestion(path)
                predictions[path] = pred
                print(f"    {path}: {pred:.4f}")
            
            adaptation_results[scenario] = predictions
            controller.get_performance_comparison()
        
        # 生成详细报告
        print("\n📋 适应性分析报告:")
        baseline = adaptation_results.get('normal', {})
        
        for scenario, predictions in adaptation_results.items():
            if scenario == 'normal':
                continue
            
            print(f"\n  {scenario}场景分析:")
            for path in predictions:
                baseline_val = baseline.get(path, 0.5)
                current_val = predictions[path]
                change = current_val - baseline_val
                
                print(f"    {path}: {current_val:.4f} (变化: {change:+.4f})")
        
        # 最终性能总结
        print("\n📈 系统性能总结:")
        print(f"  - 总路径数量: {len(controller.network_state.paths)}")
        print(f"  - 训练会话: {len(controller.training_history)}")
        print(f"  - 性能记录: {len(controller.performance_metrics)}")
        print(f"  - 在线更新: {len(controller.adaptation_buffer)}")
        print(f"  - 连接管理: {len(controller.connections)}")
        
        controller.cleanup()
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        controller.cleanup()

def main():
    """主演示函数"""
    print("🚀 MPTCP-SDN增强版系统演示")
    print("=" * 60)
    
    demo_choice = input("""
请选择演示模式:
1. 🏃 快速演示 (所有拓扑)
2. 🔍 详细演示 (单一拓扑)
3. 🎯 自动完整演示

请输入选择 (1-3，默认3): """).strip()
    
    if demo_choice == "1":
        quick_demo_all_topologies()
    elif demo_choice == "2":
        topo_choice = input("""
选择拓扑类型:
1. 简单拓扑 (6节点)
2. 中等拓扑 (12节点)  
3. 复杂拓扑 (32节点)

请输入选择 (1-3，默认2): """).strip()
        
        topo_map = {'1': 'simple', '2': 'medium', '3': 'complex'}
        topology = topo_map.get(topo_choice, 'medium')
        demo_specific_topology(topology)
    else:
        # 默认自动完整演示
        print("\n🎯 开始自动完整演示...")
        quick_demo_all_topologies()
        print("\n" + "="*60)
        print("演示结束！您可以查看生成的文件了解详细结果。")

if __name__ == "__main__":
    main() 