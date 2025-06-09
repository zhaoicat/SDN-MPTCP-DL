#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 测试正在运行的Mininet网络
"""

import subprocess
import time
import json
from datetime import datetime

def run_command(cmd):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "命令超时", 1

def test_network_connectivity():
    """测试网络连通性"""
    print("🌐 测试网络连通性...")
    
    # 测试主机间的连通性
    test_pairs = [
        ("h1", "h2"),
        ("h1", "h3"), 
        ("h1", "h4"),
        ("h2", "h3"),
        ("h2", "h4"),
        ("h3", "h4")
    ]
    
    connectivity_results = {}
    
    for src, dst in test_pairs:
        print(f"  测试 {src} → {dst}...")
        
        # 使用mnexec在Mininet命名空间中执行ping
        cmd = f"mnexec -a {src} ping -c 3 {dst}"
        stdout, stderr, returncode = run_command(cmd)
        
        if returncode == 0 and "3 received" in stdout:
            # 解析延迟
            if "avg" in stdout:
                try:
                    avg_line = [line for line in stdout.split('\n') if 'avg' in line][0]
                    avg_latency = float(avg_line.split('/')[4])
                    connectivity_results[f"{src}-{dst}"] = {
                        "status": "success",
                        "latency": avg_latency,
                        "packet_loss": 0
                    }
                    print(f"    ✅ 成功 (延迟: {avg_latency:.2f}ms)")
                except:
                    connectivity_results[f"{src}-{dst}"] = {
                        "status": "success",
                        "latency": "unknown",
                        "packet_loss": 0
                    }
                    print(f"    ✅ 成功")
            else:
                connectivity_results[f"{src}-{dst}"] = {
                    "status": "success",
                    "latency": "unknown", 
                    "packet_loss": 0
                }
                print(f"    ✅ 成功")
        else:
            connectivity_results[f"{src}-{dst}"] = {
                "status": "failed",
                "error": stderr or "连接失败"
            }
            print(f"    ❌ 失败")
    
    return connectivity_results

def test_bandwidth():
    """测试带宽"""
    print("\n📊 测试网络带宽...")
    
    bandwidth_results = {}
    
    # 启动iperf服务器
    print("  启动iperf服务器...")
    server_cmd = "mnexec -a h2 iperf -s -D"
    run_command(server_cmd)
    
    time.sleep(2)  # 等待服务器启动
    
    # 测试h1到h2的带宽
    print("  测试 h1 → h2 带宽...")
    client_cmd = "mnexec -a h1 iperf -c h2 -t 5 -f m"
    stdout, stderr, returncode = run_command(client_cmd)
    
    if returncode == 0 and "Mbits/sec" in stdout:
        try:
            # 解析带宽
            lines = stdout.split('\n')
            for line in lines:
                if 'Mbits/sec' in line and 'sec' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Mbits/sec' in part and i > 0:
                            bandwidth = float(parts[i-1])
                            bandwidth_results["h1-h2"] = {
                                "bandwidth": bandwidth,
                                "unit": "Mbps"
                            }
                            print(f"    📈 带宽: {bandwidth:.2f} Mbps")
                            break
                    break
        except:
            print(f"    ⚠️ 无法解析带宽数据")
    else:
        print(f"    ❌ 带宽测试失败")
    
    # 停止iperf服务器
    run_command("pkill iperf")
    
    return bandwidth_results

def test_switch_status():
    """测试交换机状态"""
    print("\n🔀 检查交换机状态...")
    
    switch_results = {}
    
    # 检查OVS交换机
    cmd = "ovs-vsctl list-br"
    stdout, stderr, returncode = run_command(cmd)
    
    if returncode == 0:
        bridges = stdout.strip().split('\n')
        print(f"  发现 {len(bridges)} 个交换机:")
        
        for bridge in bridges:
            if bridge.strip():
                print(f"    🔀 {bridge}")
                
                # 检查端口
                port_cmd = f"ovs-vsctl list-ports {bridge}"
                port_stdout, _, port_returncode = run_command(port_cmd)
                
                if port_returncode == 0:
                    ports = port_stdout.strip().split('\n')
                    switch_results[bridge] = {
                        "status": "active",
                        "ports": [p for p in ports if p.strip()]
                    }
                    print(f"      端口: {len([p for p in ports if p.strip()])} 个")
    
    return switch_results

def generate_test_report(connectivity, bandwidth, switches):
    """生成测试报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "mininet_network_test",
        "connectivity_test": connectivity,
        "bandwidth_test": bandwidth,
        "switch_status": switches,
        "summary": {
            "total_connectivity_tests": len(connectivity),
            "successful_connections": len([r for r in connectivity.values() if r.get("status") == "success"]),
            "active_switches": len(switches),
            "test_duration": "completed"
        }
    }
    
    # 保存报告
    with open('mininet_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def main():
    """主函数"""
    print("🧪 Mininet网络测试开始")
    print("=" * 50)
    
    # 执行测试
    connectivity_results = test_network_connectivity()
    bandwidth_results = test_bandwidth()
    switch_results = test_switch_status()
    
    # 生成报告
    report = generate_test_report(connectivity_results, bandwidth_results, switch_results)
    
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    print(f"  🌐 连通性测试: {report['summary']['successful_connections']}/{report['summary']['total_connectivity_tests']} 成功")
    print(f"  🔀 活跃交换机: {report['summary']['active_switches']} 个")
    print(f"  📈 带宽测试: {'完成' if bandwidth_results else '未完成'}")
    print(f"  📄 报告文件: mininet_test_report.json")
    
    if report['summary']['successful_connections'] > 0:
        print("  ✅ Mininet网络运行正常!")
    else:
        print("  ⚠️ Mininet网络可能存在问题")

if __name__ == "__main__":
    main() 