#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 网络拓扑可视化工具
用于生成MPTCP-SDN网络结构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class NetworkVisualizer:
    """网络拓扑可视化器"""
    
    def __init__(self):
        self.colors = {
            'host': '#FF6B6B',      # 红色 - 主机
            'switch': '#4ECDC4',    # 青色 - 交换机  
            'core': '#45B7D1',      # 蓝色 - 核心交换机
            'edge': '#96CEB4',      # 绿色 - 边缘交换机
            'link': '#95A5A6'       # 灰色 - 连接线
        }
    
    def create_simple_topology_graph(self) -> Tuple[plt.Figure, Dict]:
        """创建简单6节点拓扑图"""
        G = nx.Graph()
        
        # 添加节点
        G.add_node('h1', type='host', pos=(0, 1))
        G.add_node('h2', type='host', pos=(0, -1))
        G.add_node('h3', type='host', pos=(4, 1))
        G.add_node('h4', type='host', pos=(4, -1))
        G.add_node('s1', type='switch', pos=(1, 0))
        G.add_node('s2', type='switch', pos=(3, 0))
        
        # 添加边
        G.add_edge('h1', 's1', bandwidth='100Mbps', delay='10ms')
        G.add_edge('h2', 's1', bandwidth='100Mbps', delay='10ms')
        G.add_edge('h3', 's2', bandwidth='100Mbps', delay='10ms')
        G.add_edge('h4', 's2', bandwidth='100Mbps', delay='10ms')
        G.add_edge('s1', 's2', bandwidth='50Mbps', delay='20ms')
        
        return self._create_topology_plot(G, "Simple Network Topology (6 nodes)", (10, 6))
    
    def create_medium_topology_graph(self) -> Tuple[plt.Figure, Dict]:
        """创建中等12节点拓扑图"""
        G = nx.Graph()
        
        # 添加主机节点（8个）
        host_positions = [
            (0, 2), (0, 1), (2, 3), (2, 2),  # 连接到s1, s2
            (4, 3), (4, 2), (6, 2), (6, 1)   # 连接到s3, s4
        ]
        
        for i, pos in enumerate(host_positions):
            G.add_node(f'h{i+1}', type='host', pos=pos)
        
        # 添加交换机节点（4个）
        switch_positions = [(1, 1.5), (3, 2.5), (5, 2.5), (7, 1.5)]
        for i, pos in enumerate(switch_positions):
            G.add_node(f's{i+1}', type='switch', pos=pos)
        
        # 主机到交换机的连接
        host_to_switch = [
            ('h1', 's1'), ('h2', 's1'),
            ('h3', 's2'), ('h4', 's2'),
            ('h5', 's3'), ('h6', 's3'),
            ('h7', 's4'), ('h8', 's4')
        ]
        
        for host, switch in host_to_switch:
            G.add_edge(host, switch, bandwidth='100Mbps', delay='10ms')
        
        # 交换机之间的环形连接
        switch_connections = [('s1', 's2'), ('s2', 's3'), ('s3', 's4'), ('s4', 's1')]
        for s1, s2 in switch_connections:
            G.add_edge(s1, s2, bandwidth='50Mbps', delay='15ms')
        
        return self._create_topology_plot(G, "Medium Network Topology (12 nodes)", (12, 8))
    
    def create_complex_topology_graph(self) -> Tuple[plt.Figure, Dict]:
        """创建复杂32节点拓扑图"""
        G = nx.Graph()
        
        # 添加24个主机节点（分为4组，每组6个）
        host_groups = [
            [(0, 4), (0, 3), (0, 2), (1, 4), (1, 3), (1, 2)],      # 组1
            [(0, 0), (0, -1), (0, -2), (1, 0), (1, -1), (1, -2)],   # 组2
            [(6, 4), (6, 3), (6, 2), (7, 4), (7, 3), (7, 2)],      # 组3
            [(6, 0), (6, -1), (6, -2), (7, 0), (7, -1), (7, -2)]   # 组4
        ]
        
        host_idx = 1
        for group in host_groups:
            for pos in group:
                G.add_node(f'h{host_idx}', type='host', pos=pos)
                host_idx += 1
        
        # 添加4个边缘交换机
        edge_switches = [
            ('edge1', (2, 3)), ('edge2', (2, -1)), 
            ('edge3', (5, 3)), ('edge4', (5, -1))
        ]
        for name, pos in edge_switches:
            G.add_node(name, type='edge', pos=pos)
        
        # 添加4个核心交换机
        core_switches = [
            ('core1', (3, 2)), ('core2', (4, 2)),
            ('core3', (3, 0)), ('core4', (4, 0))
        ]
        for name, pos in core_switches:
            G.add_node(name, type='core', pos=pos)
        
        # 主机到边缘交换机的连接
        host_to_edge = [
            # 组1到edge1
            ('h1', 'edge1'), ('h2', 'edge1'), ('h3', 'edge1'),
            ('h4', 'edge1'), ('h5', 'edge1'), ('h6', 'edge1'),
            # 组2到edge2
            ('h7', 'edge2'), ('h8', 'edge2'), ('h9', 'edge2'),
            ('h10', 'edge2'), ('h11', 'edge2'), ('h12', 'edge2'),
            # 组3到edge3
            ('h13', 'edge3'), ('h14', 'edge3'), ('h15', 'edge3'),
            ('h16', 'edge3'), ('h17', 'edge3'), ('h18', 'edge3'),
            # 组4到edge4
            ('h19', 'edge4'), ('h20', 'edge4'), ('h21', 'edge4'),
            ('h22', 'edge4'), ('h23', 'edge4'), ('h24', 'edge4')
        ]
        
        for host, edge in host_to_edge:
            G.add_edge(host, edge, bandwidth='100Mbps', delay='5ms')
        
        # 边缘交换机到核心交换机的全互联
        edges = ['edge1', 'edge2', 'edge3', 'edge4']
        cores = ['core1', 'core2', 'core3', 'core4']
        
        for edge in edges:
            for core in cores:
                G.add_edge(edge, core, bandwidth='1000Mbps', delay='2ms')
        
        return self._create_topology_plot(G, "Complex Network Topology (32 nodes)", (15, 10))
    
    def _create_topology_plot(self, G: nx.Graph, title: str, figsize: Tuple[int, int]) -> Tuple[plt.Figure, Dict]:
        """创建拓扑图"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 获取节点位置
        pos = nx.get_node_attributes(G, 'pos')
        
        # 分类节点
        hosts = [n for n in G.nodes() if G.nodes[n].get('type') == 'host']
        switches = [n for n in G.nodes() if G.nodes[n].get('type') == 'switch']
        cores = [n for n in G.nodes() if G.nodes[n].get('type') == 'core']
        edges = [n for n in G.nodes() if G.nodes[n].get('type') == 'edge']
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color=self.colors['link'], 
                              width=2, alpha=0.6, ax=ax)
        
        # 绘制不同类型的节点
        if hosts:
            nx.draw_networkx_nodes(G, pos, nodelist=hosts, 
                                  node_color=self.colors['host'],
                                  node_size=800, node_shape='o', ax=ax)
        
        if switches:
            nx.draw_networkx_nodes(G, pos, nodelist=switches,
                                  node_color=self.colors['switch'],
                                  node_size=1200, node_shape='s', ax=ax)
        
        if cores:
            nx.draw_networkx_nodes(G, pos, nodelist=cores,
                                  node_color=self.colors['core'],
                                  node_size=1500, node_shape='s', ax=ax)
        
        if edges:
            nx.draw_networkx_nodes(G, pos, nodelist=edges,
                                  node_color=self.colors['edge'],
                                  node_size=1200, node_shape='s', ax=ax)
        
        # 添加节点标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # 添加边标签（带宽和延迟信息）
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if 'bandwidth' in data and 'delay' in data:
                edge_labels[(u, v)] = f"{data['bandwidth']}\n{data['delay']}"
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        # 设置标题和图例
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 创建图例
        legend_elements = []
        if hosts:
            legend_elements.append(mpatches.Patch(color=self.colors['host'], label='Host'))
        if switches:
            legend_elements.append(mpatches.Patch(color=self.colors['switch'], label='Switch'))
        if cores:
            legend_elements.append(mpatches.Patch(color=self.colors['core'], label='Core Switch'))
        if edges:
            legend_elements.append(mpatches.Patch(color=self.colors['edge'], label='Edge Switch'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        # 返回统计信息
        stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'hosts': len(hosts),
            'switches': len(switches),
            'cores': len(cores),
            'edges': len(edges)
        }
        
        return fig, stats
    
    def generate_all_topologies(self, save_path: str = "network_topologies"):
        """生成所有网络拓扑图"""
        topologies = [
            ('simple', self.create_simple_topology_graph),
            ('medium', self.create_medium_topology_graph),
            ('complex', self.create_complex_topology_graph)
        ]
        
        results = {}
        
        for topo_name, create_func in topologies:
            print(f"🌐 生成{topo_name}拓扑图...")
            fig, stats = create_func()
            
            # 保存图片
            filename = f"{save_path}_{topo_name}_topology.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            results[topo_name] = {
                'filename': filename,
                'stats': stats
            }
            
            print(f"  ✅ 已保存: {filename}")
            print(f"  📊 统计: {stats['total_nodes']}节点, {stats['total_edges']}连接")
        
        return results

def main():
    """主函数"""
    print("🚀 网络拓扑可视化工具")
    print("=" * 50)
    
    visualizer = NetworkVisualizer()
    
    print("\n🌐 生成网络拓扑图...")
    results = visualizer.generate_all_topologies()
    
    print(f"\n📊 生成完成! 共生成{len(results)}个拓扑图:")
    for topo_name, info in results.items():
        print(f"  • {topo_name}: {info['filename']}")
        stats = info['stats']
        print(f"    节点: {stats['total_nodes']}, 连接: {stats['total_edges']}")
        if stats['hosts']:
            print(f"    主机: {stats['hosts']}, 交换机: {stats['switches'] + stats['cores'] + stats['edges']}")
    
    print("\n✨ 可视化完成!")

if __name__ == "__main__":
    main() 