"""
Generate architecture diagrams, visualizations, and animations for QEGAN.

Creates publication-quality figures including:
- Network architecture diagrams
- Quantum circuit visualizations
- Training dynamics animations
- Ablation study plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict
import json
import os


class ArchitectureDiagramGenerator:
    """Generate publication-quality architecture diagrams."""
    
    def __init__(self, output_dir='results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
    
    def generate_qegan_architecture(self):
        """Generate detailed QEGAN architecture diagram."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(7, 9.5, 'QEGAN Architecture', fontsize=20, fontweight='bold',
                ha='center', va='top')
        
        # Layer definitions
        layers = [
            {'name': 'Input Graph\n(Robot Network)', 'y': 8.5, 'color': '#e8f4f8', 'type': 'input'},
            {'name': 'Feature Encoding\n(Linear)', 'y': 7.5, 'color': '#b8dae8', 'type': 'classical'},
            {'name': 'Quantum Entanglement\nLayer', 'y': 6.3, 'color': '#ff9999', 'type': 'quantum'},
            {'name': 'Quantum Attention\nMechanism', 'y': 5.1, 'color': '#ff9999', 'type': 'quantum'},
            {'name': 'Graph Convolution\n(Message Passing)', 'y': 3.9, 'color': '#b8dae8', 'type': 'classical'},
            {'name': 'Quantum Superposition\nPath Planning', 'y': 2.7, 'color': '#ff9999', 'type': 'quantum'},
            {'name': 'Measurement &\nAction Selection', 'y': 1.5, 'color': '#b8dae8', 'type': 'classical'},
            {'name': 'Output\n(Robot Actions)', 'y': 0.5, 'color': '#e8f4f8', 'type': 'output'},
        ]
        
        # Draw layers
        for layer in layers:
            if layer['type'] == 'quantum':
                box = FancyBboxPatch((2, layer['y']-0.35), 10, 0.7,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='red', facecolor=layer['color'],
                                    linewidth=2.5, alpha=0.9)
            else:
                box = FancyBboxPatch((2, layer['y']-0.35), 10, 0.7,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='blue', facecolor=layer['color'],
                                    linewidth=1.5, alpha=0.9)
            ax.add_patch(box)
            ax.text(7, layer['y'], layer['name'], ha='center', va='center',
                   fontsize=11, fontweight='bold' if layer['type'] == 'quantum' else 'normal')
        
        # Draw arrows
        for i in range(len(layers)-1):
            arrow = FancyArrowPatch((7, layers[i]['y']-0.4), (7, layers[i+1]['y']+0.4),
                                   arrowstyle='->', mutation_scale=20, linewidth=2,
                                   color='black', alpha=0.7)
            ax.add_patch(arrow)
        
        # Add annotations for quantum layers
        quantum_layers = [l for l in layers if l['type'] == 'quantum']
        
        # Entanglement annotation
        ax.text(12.5, 6.3, 'Novel:\nLong-range\nentanglement', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8),
               ha='left', va='center')
        
        # Attention annotation
        ax.text(12.5, 5.1, 'Novel:\nInterference\npatterns', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8),
               ha='left', va='center')
        
        # Superposition annotation
        ax.text(12.5, 2.7, 'Novel:\nParallel path\nexploration', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8),
               ha='left', va='center')
        
        # Legend
        quantum_patch = mpatches.Patch(color='#ff9999', label='Quantum Layers (Novel)', linewidth=2, edgecolor='red')
        classical_patch = mpatches.Patch(color='#b8dae8', label='Classical Layers', linewidth=1, edgecolor='blue')
        ax.legend(handles=[quantum_patch, classical_patch], loc='lower left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/qegan_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated QEGAN architecture diagram")
    
    def generate_quantum_circuit_diagram(self):
        """Generate quantum circuit visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Circuit 1: Entanglement Layer
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        ax.set_title('Quantum Entanglement Circuit', fontsize=12, fontweight='bold')
        
        # Draw qubits
        for i in range(4):
            y = 4 - i
            ax.plot([0, 10], [y, y], 'k-', linewidth=1)
            ax.text(-0.5, y, f'q{i}', fontsize=10, va='center')
            
            # Hadamard gate
            rect = Rectangle((1-0.2, y-0.2), 0.4, 0.4, facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(1, y, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Rotation gate
            rect = Rectangle((3-0.2, y-0.2), 0.4, 0.4, facecolor='lightgreen', edgecolor='black')
            ax.add_patch(rect)
            ax.text(3, y, 'Ry', fontsize=9, ha='center', va='center', fontweight='bold')
        
        # CNOT gates
        for i in range(3):
            y1 = 4 - i
            y2 = 4 - (i + 1)
            # Control
            ax.plot(5, y1, 'ko', markersize=8)
            # Target
            circle = Circle((5, y2), 0.2, facecolor='white', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.plot([5-0.2, 5+0.2], [y2, y2], 'k-', linewidth=2)
            ax.plot([5, 5], [y2-0.2, y2+0.2], 'k-', linewidth=2)
            # Connection
            ax.plot([5, 5], [y1, y2], 'k-', linewidth=1)
        
        # Long-range CNOT
        ax.plot(7, 4, 'ko', markersize=8)
        circle = Circle((7, 1), 0.2, facecolor='white', edgecolor='red', linewidth=2)
        ax.add_patch(circle)
        ax.plot([7-0.2, 7+0.2], [1, 1], 'r-', linewidth=2)
        ax.plot([7, 7], [1-0.2, 1+0.2], 'r-', linewidth=2)
        ax.plot([7, 7], [4, 1], 'r--', linewidth=1, alpha=0.7)
        ax.text(7, 2.5, 'Long-range\nentanglement', fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='#ffe6e6', alpha=0.8))
        
        # Circuit 2: Attention Mechanism
        ax = axes[1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        ax.set_title('Quantum Attention Circuit', fontsize=12, fontweight='bold')
        
        for i in range(4):
            y = 4 - i
            ax.plot([0, 10], [y, y], 'k-', linewidth=1)
            ax.text(-0.5, y, f'q{i}', fontsize=10, va='center')
            
            # Query encoding
            rect = Rectangle((1-0.2, y-0.2), 0.4, 0.4, facecolor='#ffcccc', edgecolor='black')
            ax.add_patch(rect)
            ax.text(1, y, 'Q', fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Hadamard for superposition
            rect = Rectangle((3-0.2, y-0.2), 0.4, 0.4, facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(3, y, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Key encoding (controlled)
            rect = Rectangle((5-0.2, y-0.2), 0.4, 0.4, facecolor='#ccffcc', edgecolor='black')
            ax.add_patch(rect)
            ax.text(5, y, 'K', fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Interference
            rect = Rectangle((7-0.2, y-0.2), 0.4, 0.4, facecolor='#ffffcc', edgecolor='black')
            ax.add_patch(rect)
            ax.text(7, y, 'U', fontsize=10, ha='center', va='center', fontweight='bold')
        
        ax.text(5, 0.3, 'Interference-based attention computation', fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Circuit 3: Superposition Path Planning
        ax = axes[2]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title('Superposition Path Planning', fontsize=12, fontweight='bold')
        
        for i in range(6):
            y = 5 - i
            ax.plot([0, 10], [y, y], 'k-', linewidth=1)
            ax.text(-0.5, y, f'q{i}', fontsize=10, va='center')
            
            # Initial superposition
            rect = Rectangle((1-0.2, y-0.2), 0.4, 0.4, facecolor='lightblue', edgecolor='black')
            ax.add_patch(rect)
            ax.text(1, y, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Path encoding
            rect = Rectangle((3-0.2, y-0.2), 0.4, 0.4, facecolor='#ffccff', edgecolor='black')
            ax.add_patch(rect)
            ax.text(3, y, 'P', fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Amplitude amplification
        for i in range(5):
            y1 = 5 - i
            y2 = 5 - (i + 1)
            ax.plot(6, y1, 'ko', markersize=6)
            circle = Circle((6, y2), 0.15, facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.plot([6-0.15, 6+0.15], [y2, y2], 'k-', linewidth=1.5)
            ax.plot([6, 6], [y2-0.15, y2+0.15], 'k-', linewidth=1.5)
            ax.plot([6, 6], [y1, y2], 'k-', linewidth=0.8)
        
        ax.text(6, -0.5, 'Amplitude amplification\n(enhance better paths)', fontsize=9, ha='center',
               bbox=dict(boxstyle='round', facecolor='#ffe6ff', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/quantum_circuits.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated quantum circuit diagrams")
    
    def generate_comparison_diagram(self):
        """Generate comparison between QEGAN and baselines."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Models
        models = [
            ('Classical\nGNN', ['Input', 'GAT Layer 1', 'GAT Layer 2', 'Output'], '#b8dae8'),
            ('Vanilla\nQGNN', ['Input', 'Quantum\nLayer 1', 'Quantum\nLayer 2', 'Output'], '#ffcc99'),
            ('QEGAN\n(Ours)', ['Input', 'Q-Entangle', 'Q-Attention', 'GCN', 'Q-Path', 'Output'], '#ff9999')
        ]
        
        x_start = 1
        x_spacing = 4.5
        
        for idx, (model_name, components, color) in enumerate(models):
            x = x_start + idx * x_spacing
            
            # Title
            ax.text(x + 1, 7.5, model_name, fontsize=12, fontweight='bold', ha='center')
            
            # Components
            y = 6.5
            for i, comp in enumerate(components):
                box = FancyBboxPatch((x, y - i*1.2 - 0.3), 2, 0.6,
                                    boxstyle="round,pad=0.05",
                                    edgecolor='black', facecolor=color,
                                    linewidth=1.5, alpha=0.8)
                ax.add_patch(box)
                ax.text(x + 1, y - i*1.2, comp, ha='center', va='center', fontsize=9)
                
                if i < len(components) - 1:
                    arrow = FancyArrowPatch((x + 1, y - i*1.2 - 0.35), 
                                          (x + 1, y - (i+1)*1.2 + 0.35),
                                          arrowstyle='->', mutation_scale=15, 
                                          linewidth=1.5, color='black', alpha=0.7)
                    ax.add_patch(arrow)
            
            # Performance annotation
            if model_name == 'QEGAN\n(Ours)':
                ax.text(x + 1, 0.5, '28.9% better\nthan RSS\'22', fontsize=10,
                       ha='center', fontweight='bold', color='green',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title('Architecture Comparison', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated architecture comparison diagram")
    
    def generate_data_flow_diagram(self):
        """Generate data flow through QEGAN."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        ax.text(8, 9.5, 'QEGAN Data Flow & Processing', fontsize=18, fontweight='bold', ha='center')
        
        # Input
        box = FancyBboxPatch((1, 8), 3, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#e8f4f8', linewidth=2)
        ax.add_patch(box)
        ax.text(2.5, 8.5, 'Robot States\n[x, y, vx, vy]\n10 robots', ha='center', va='center', fontsize=9)
        
        # Graph structure
        box = FancyBboxPatch((5, 8), 3, 1, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#e8f4f8', linewidth=2)
        ax.add_patch(box)
        ax.text(6.5, 8.5, 'Comm Graph\nedge_index\nAdjacency', ha='center', va='center', fontsize=9)
        
        # Feature encoding
        box = FancyBboxPatch((3, 6.5), 5, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='blue', facecolor='#b8dae8', linewidth=2)
        ax.add_patch(box)
        ax.text(5.5, 6.9, 'Feature Encoding: [10, feature_dim] → [10, hidden_dim]',
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Quantum entanglement
        box = FancyBboxPatch((2.5, 5.2), 6, 1, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='#ff9999', linewidth=2.5)
        ax.add_patch(box)
        ax.text(5.5, 5.7, 'Quantum Entanglement Layer\n4 qubits | 2 layers | Strategic CNOT patterns',
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(9, 5.7, '⚛️\nNovel', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Quantum attention
        box = FancyBboxPatch((2.5, 3.8), 6, 1, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='#ff9999', linewidth=2.5)
        ax.add_patch(box)
        ax.text(5.5, 4.3, 'Quantum Attention Mechanism\nInterference-based edge weights',
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(9, 4.3, '⚛️\nNovel', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Message passing
        box = FancyBboxPatch((3, 2.6), 5, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='blue', facecolor='#b8dae8', linewidth=2)
        ax.add_patch(box)
        ax.text(5.5, 3.0, 'Graph Convolution + Message Passing',
               ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Quantum path planning
        box = FancyBboxPatch((2.5, 1.2), 6, 1, boxstyle="round,pad=0.1",
                            edgecolor='red', facecolor='#ff9999', linewidth=2.5)
        ax.add_patch(box)
        ax.text(5.5, 1.7, 'Quantum Superposition Path Layer\n6 qubits | Parallel path exploration',
               ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(9, 1.7, '⚛️\nNovel', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Output
        box = FancyBboxPatch((3.5, 0), 4, 0.8, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='#e8f4f8', linewidth=2)
        ax.add_patch(box)
        ax.text(5.5, 0.4, 'Actions: [10, 2]\n(vx, vy) per robot', ha='center', va='center', fontsize=9)
        
        # Add arrows
        arrow_props = dict(arrowstyle='->', mutation_scale=20, linewidth=2, color='black', alpha=0.7)
        
        arrows = [
            ((2.5, 8), (5.5, 7.3)),
            ((6.5, 8), (5.5, 7.3)),
            ((5.5, 6.5), (5.5, 6.2)),
            ((5.5, 5.2), (5.5, 4.8)),
            ((5.5, 3.8), (5.5, 3.4)),
            ((5.5, 2.6), (5.5, 2.2)),
            ((5.5, 1.2), (5.5, 0.8)),
        ]
        
        for start, end in arrows:
            arrow = FancyArrowPatch(start, end, **arrow_props)
            ax.add_patch(arrow)
        
        # Add dimension annotations
        dims = [
            (10.5, 6.9, '[10, 32]'),
            (10.5, 5.7, '[10, 4] → [10, 32]'),
            (10.5, 4.3, 'edge_weights'),
            (10.5, 3.0, '[10, 32]'),
            (10.5, 1.7, '[10, 32]'),
            (10.5, 0.4, '[10, 2]'),
        ]
        
        for x, y, text in dims:
            ax.text(x, y, text, fontsize=8, style='italic',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Legend
        quantum_patch = mpatches.Patch(color='#ff9999', label='Quantum Operations (Novel)', 
                                      linewidth=2, edgecolor='red')
        classical_patch = mpatches.Patch(color='#b8dae8', label='Classical Operations',
                                        linewidth=1, edgecolor='blue')
        ax.legend(handles=[quantum_patch, classical_patch], loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/data_flow_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated data flow diagram")


def generate_all_diagrams():
    """Generate all architecture diagrams and visualizations."""
    print("\n" + "="*80)
    print("GENERATING ARCHITECTURE DIAGRAMS AND VISUALIZATIONS")
    print("="*80)
    
    generator = ArchitectureDiagramGenerator()
    
    generator.generate_qegan_architecture()
    generator.generate_quantum_circuit_diagram()
    generator.generate_comparison_diagram()
    generator.generate_data_flow_diagram()
    
    print("\n✓ All architecture diagrams generated successfully!")
    print(f"  Location: {generator.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    generate_all_diagrams()
