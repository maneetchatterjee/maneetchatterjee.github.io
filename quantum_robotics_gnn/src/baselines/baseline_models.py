"""
Baseline models for comparison with QEGAN.

1. Classical Graph Neural Network with Attention (Classical GNN)
2. Vanilla Quantum Graph Neural Network (Vanilla QGNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import add_self_loops
import pennylane as qml
import numpy as np


class ClassicalGNN(nn.Module):
    """
    Classical Graph Neural Network with attention mechanism.
    Baseline for comparison with quantum approaches.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        heads: int = 4
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=0.1)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        """Forward pass through classical GNN."""
        x = self.input_proj(x)
        x = self.activation(x)
        
        for gat_layer in self.gat_layers:
            x_new = gat_layer(x, edge_index)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        x = self.output_proj(x)
        return x


class VanillaQuantumLayer(nn.Module):
    """
    Vanilla quantum layer that simply replaces classical layers with quantum circuits.
    No entanglement-based attention or superposition path planning.
    """
    
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.params = nn.Parameter(torch.randn(n_qubits, 3) * 0.1)
        
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')
    
    def _circuit(self, inputs, params):
        """Simple quantum circuit without entanglement structure."""
        # Encode inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Apply rotations (no strategic entanglement)
        for i in range(self.n_qubits):
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)
        
        # Simple CNOT chain (not optimized for robot interactions)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Process through vanilla quantum circuit."""
        batch_size = x.shape[0]
        outputs = []
        
        for i in range(batch_size):
            input_normalized = torch.tanh(x[i]) * np.pi / 2
            quantum_output = self.qnode(input_normalized, self.params)
            outputs.append(torch.stack(quantum_output))
        
        return torch.stack(outputs).float()


class VanillaQGNNLayer(MessagePassing):
    """
    Vanilla Quantum GNN layer - straightforward quantum circuit integration.
    Lacks the sophisticated quantum attention and entanglement structure of QEGAN.
    """
    
    def __init__(self, in_channels: int, out_channels: int, n_qubits: int = 4):
        super().__init__(aggr='add')
        
        self.quantum_layer = VanillaQuantumLayer(n_qubits=n_qubits)
        self.linear = nn.Linear(in_channels, out_channels)
        self.quantum_proj = nn.Linear(n_qubits, out_channels)
        
    def forward(self, x, edge_index):
        """Forward pass with vanilla quantum processing."""
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Classical part
        x_classical = self.linear(x)
        
        # Vanilla quantum processing (no entanglement strategy)
        if x.shape[1] >= self.quantum_layer.n_qubits:
            x_for_quantum = x[:, :self.quantum_layer.n_qubits]
        else:
            padding = torch.zeros(
                x.shape[0],
                self.quantum_layer.n_qubits - x.shape[1],
                device=x.device
            )
            x_for_quantum = torch.cat([x, padding], dim=1)
        
        x_quantum = self.quantum_layer(x_for_quantum)
        x_quantum = self.quantum_proj(x_quantum)
        
        x_combined = x_classical + x_quantum
        
        # Standard message passing (no quantum attention)
        return self.propagate(edge_index, x=x_combined)
    
    def message(self, x_j):
        """Standard message function without quantum attention."""
        return x_j


class VanillaQGNN(nn.Module):
    """
    Vanilla Quantum Graph Neural Network.
    Baseline quantum approach without novel contributions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_qubits: int = 4
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.qgnn_layers = nn.ModuleList([
            VanillaQGNNLayer(hidden_dim, hidden_dim, n_qubits=n_qubits)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        """Forward pass through vanilla QGNN."""
        x = self.input_proj(x)
        x = self.activation(x)
        
        for layer in self.qgnn_layers:
            x_new = layer(x, edge_index)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new
        
        x = self.output_proj(x)
        return x


def create_classical_gnn(input_dim: int, hidden_dim: int, output_dim: int):
    """Create classical GNN baseline."""
    return ClassicalGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=2,
        heads=4
    )


def create_vanilla_qgnn(input_dim: int, hidden_dim: int, output_dim: int):
    """Create vanilla QGNN baseline."""
    return VanillaQGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=2,
        n_qubits=4
    )
