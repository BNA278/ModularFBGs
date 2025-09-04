import torch
import torch.nn as nn
from torch_geometric.nn import GINConv


class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.conv = GINConv(self.mlp)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, edge_index):
        return self.conv(x, edge_index) + (1 + self.epsilon) * x


class AttentionReadout(nn.Module):
    def __init__(self, input_dim):
        super(AttentionReadout, self).__init__()
        self.Q1 = nn.Linear(input_dim, input_dim)
        self.Q2 = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, F):
        F_mean = torch.mean(F, dim=0, keepdim=True)
        S = torch.sigmoid(self.Q2(torch.tanh(self.Q1(F) + self.Q1(F_mean))))
        S = self.softmax(S)
        G = torch.sum(S * F, dim=0)
        return G