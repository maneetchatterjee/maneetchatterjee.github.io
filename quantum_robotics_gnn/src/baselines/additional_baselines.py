"""
Additional baseline models from recent literature for comprehensive comparison.

Implements state-of-the-art multi-agent coordination methods from top-tier venues:
- CommNet (NIPS 2016)
- Graph Attention Networks (ICLR 2018)  
- Multi-Agent Transformer (NeurIPS 2021)
- DGN - Deep Graph Networks (ICML 2020)
- GNN-based MARL (IJCAI 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops
import numpy as np


class CommNet(nn.Module):
    """
    Communication Neural Network (CommNet) - NIPS 2016
    
    Reference: Sukhbaatar et al., "Learning Multiagent Communication with 
    Backpropagation", NIPS 2016
    
    Baseline for multi-agent communication and coordination.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_rounds: int = 2
    ):
        super().__init__()
        
        self.num_rounds = num_rounds
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Communication modules
        self.comm_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for _ in range(num_rounds)
        ])
        
        # Output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index):
        """Forward pass with communication rounds."""
        # Encode
        h = self.encoder(x)
        
        # Communication rounds
        for comm_layer in self.comm_layers:
            # Average messages from neighbors
            messages = []
            for i in range(x.size(0)):
                # Find neighbors
                neighbors = edge_index[1][edge_index[0] == i]
                if len(neighbors) > 0:
                    neighbor_hidden = h[neighbors]
                    avg_message = neighbor_hidden.mean(dim=0)
                else:
                    avg_message = torch.zeros_like(h[i])
                messages.append(avg_message)
            
            messages = torch.stack(messages)
            
            # Update hidden state
            h = h + comm_layer(messages)
        
        # Decode
        return self.decoder(h)


class MultiAgentTransformer(nn.Module):
    """
    Multi-Agent Transformer - NeurIPS 2021
    
    Reference: Wen et al., "Multi-Agent Reinforcement Learning is a 
    Sequence Modeling Problem", NeurIPS 2021
    
    Uses self-attention for agent-agent interaction modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        """Forward pass through transformer."""
        # Project input
        x = self.input_proj(x)
        
        # Add batch dimension for transformer
        x = x.unsqueeze(0)
        
        # Transform
        x = self.transformer(x)
        
        # Remove batch dimension
        x = x.squeeze(0)
        
        # Output
        return self.output_proj(x)


class DGN(nn.Module):
    """
    Deep Graph Network (DGN) - ICML 2020
    
    Reference: Jiang et al., "Graph Convolutional Reinforcement Learning 
    for Multi-Agent Cooperation", ICML 2020
    
    State-of-the-art GNN for multi-agent cooperation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index):
        """Forward pass through DGN."""
        x = self.input_proj(x)
        x = self.activation(x)
        
        for gcn_layer in self.gcn_layers:
            x_new = gcn_layer(x, edge_index)
            x_new = self.activation(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual
        
        return self.output_proj(x)


class ATOC(nn.Module):
    """
    Actor-Attention-Critic with Communication (ATOC) - AAAI 2019
    
    Reference: Jiang and Lu, "Learning Attentional Communication for 
    Multi-Agent Cooperation", AAAI 2019
    
    Attention-based communication for multi-agent systems.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        comm_dim: int = 32
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Attention for communication
        self.query_net = nn.Linear(hidden_dim, comm_dim)
        self.key_net = nn.Linear(hidden_dim, comm_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
        
        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index):
        """Forward with attention-based communication."""
        # Encode
        h = self.input_proj(x)
        
        # Compute attention
        queries = self.query_net(h)
        keys = self.key_net(h)
        values = self.value_net(h)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(queries.size(-1))
        
        # Mask attention based on edges
        attention_mask = torch.zeros(x.size(0), x.size(0), device=x.device)
        if edge_index.size(1) > 0:
            attention_mask[edge_index[0], edge_index[1]] = 1
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores + attention_mask, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply attention
        comm_features = torch.matmul(attention_weights, values)
        
        # Concatenate with own features
        combined = torch.cat([h, comm_features], dim=-1)
        
        # Output action
        return self.action_net(combined)


class G2ANet(nn.Module):
    """
    Graph to Attention Network (G2ANet) - IJCAI 2020
    
    Reference: Liu et al., "Multi-Agent Game Abstraction via Graph 
    Attention Neural Network", IJCAI 2020
    
    Hard attention mechanism for agent selection.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1)
            for _ in range(2)
        ])
        
        # Hard attention gate
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index):
        """Forward with hard attention gating."""
        x = self.input_proj(x)
        x = self.activation(x)
        
        for gat_layer in self.gat_layers:
            x_new = gat_layer(x, edge_index)
            x_new = self.activation(x_new)
            
            # Hard attention gate
            gate = self.gate_net(x)
            x = x + gate * x_new
        
        return self.output_proj(x)


class TarMAC(nn.Module):
    """
    Targeted Multi-Agent Communication (TarMAC) - ICLR 2019
    
    Reference: Das et al., "TarMAC: Targeted Multi-Agent Communication", 
    ICLR 2019
    
    Targeted communication with signature-based addressing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        signature_dim: int = 16
    ):
        super().__init__()
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Signature networks
        self.signature_net = nn.Linear(hidden_dim, signature_dim)
        
        # Message networks
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Aggregation
        self.aggregation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        """Forward with targeted communication."""
        # Encode
        h = self.encoder(x)
        
        # Compute signatures
        signatures = self.signature_net(h)
        signatures = F.normalize(signatures, dim=-1)
        
        # Compute messages
        messages = self.message_net(h)
        
        # Targeted communication
        comm_features = []
        for i in range(x.size(0)):
            # Find neighbors
            neighbors = edge_index[1][edge_index[0] == i]
            
            if len(neighbors) > 0:
                # Compute targeting weights
                target_scores = torch.matmul(signatures[i:i+1], signatures[neighbors].T)
                target_weights = F.softmax(target_scores, dim=-1)
                
                # Aggregate targeted messages
                targeted_message = torch.matmul(target_weights, messages[neighbors])
                comm_features.append(targeted_message.squeeze(0))
            else:
                comm_features.append(torch.zeros_like(h[i]))
        
        comm_features = torch.stack(comm_features)
        
        # Combine with own features
        combined = torch.cat([h, comm_features], dim=-1)
        combined = self.aggregation_net(combined)
        
        return self.decoder(combined)


# Factory functions
def create_commnet(input_dim: int, hidden_dim: int, output_dim: int):
    """Create CommNet baseline."""
    return CommNet(input_dim, hidden_dim, output_dim, num_rounds=2)


def create_transformer(input_dim: int, hidden_dim: int, output_dim: int):
    """Create Multi-Agent Transformer baseline."""
    return MultiAgentTransformer(input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2)


def create_dgn(input_dim: int, hidden_dim: int, output_dim: int):
    """Create DGN baseline."""
    return DGN(input_dim, hidden_dim, output_dim, num_layers=3)


def create_atoc(input_dim: int, hidden_dim: int, output_dim: int):
    """Create ATOC baseline."""
    return ATOC(input_dim, hidden_dim, output_dim, comm_dim=32)


def create_g2anet(input_dim: int, hidden_dim: int, output_dim: int):
    """Create G2ANet baseline."""
    return G2ANet(input_dim, hidden_dim, output_dim, num_heads=4)


def create_tarmac(input_dim: int, hidden_dim: int, output_dim: int):
    """Create TarMAC baseline."""
    return TarMAC(input_dim, hidden_dim, output_dim, signature_dim=16)
