import torch
import torch.nn as nn
import torch.nn.functional as F

from GIN import GINLayer, AttentionReadout
from SBM import StochasticBlockModel
from Transformer import DynamicGraphTransformer


class ModularConstrainedDynamicGNN(nn.Module):
    def __init__(self, node_dims, hidden_dim, num_modules, nhead, num_layers, dropout=0.5):
        super(ModularConstrainedDynamicGNN, self).__init__()

        self.gin_layers_116 = nn.ModuleList([GINLayer(node_dims[0], hidden_dim), GINLayer(hidden_dim, hidden_dim)])
        self.gin_layers_160 = nn.ModuleList([GINLayer(node_dims[1], hidden_dim), GINLayer(hidden_dim, hidden_dim)])
        self.gin_layers_200 = nn.ModuleList([GINLayer(node_dims[2], hidden_dim), GINLayer(hidden_dim, hidden_dim)])

        self.readout = AttentionReadout(hidden_dim)
        self.transformer = DynamicGraphTransformer(hidden_dim * 3, nhead, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim * 3, 2)
        self.dropout = nn.Dropout(dropout)
        self.num_modules = num_modules

        self.sbm_116 = StochasticBlockModel(num_modules)
        self.sbm_160 = StochasticBlockModel(num_modules)
        self.sbm_200 = StochasticBlockModel(num_modules)

    def forward(self, x_list, edge_index_list, A_list):
        T = x_list.shape[0]
        graph_embeddings = []

        for t in range(T):
            graph_embs = []

            x_116 = x_list[t, 0]
            edge_index_116 = edge_index_list[t, 0]
            for gin_layer in self.gin_layers_116:
                x_116 = F.relu(gin_layer(x_116, edge_index_116))
            graph_embs.append(self.readout(x_116))

            x_160 = x_list[t, 1]
            edge_index_160 = edge_index_list[t, 1]
            for gin_layer in self.gin_layers_160:
                x_160 = F.relu(gin_layer(x_160, edge_index_160))
            graph_embs.append(self.readout(x_160))

            x_200 = x_list[t, 2]
            edge_index_200 = edge_index_list[t, 2]
            for gin_layer in self.gin_layers_200:
                x_200 = F.relu(gin_layer(x_200, edge_index_200))
            graph_embs.append(self.readout(x_200))

            concatenated_emb = torch.cat(graph_embs, dim=-1)
            graph_embeddings.append(concatenated_emb)

        graph_embeddings = torch.stack(graph_embeddings)
        dynamic_embeddings = self.transformer(graph_embeddings)
        global_embedding = torch.sum(dynamic_embeddings, dim=0)
        global_embedding = self.dropout(global_embedding)
        logits = self.fc(global_embedding)

        return logits, dynamic_embeddings

    def compute_modularity_constraint(self, dynamic_embeddings, A_list):
        hidden_dim = dynamic_embeddings.shape[1] // 3

        self.sbm_116.fit(A_list[0])
        partition_116 = self.sbm_116.block_assignments
        L_M_116 = self._compute_single_modularity_loss(dynamic_embeddings[:, :hidden_dim], A_list[0], partition_116)

        self.sbm_160.fit(A_list[1])
        partition_160 = self.sbm_160.block_assignments
        L_M_160 = self._compute_single_modularity_loss(dynamic_embeddings[:, hidden_dim:2 * hidden_dim], A_list[1],
                                                       partition_160)

        self.sbm_200.fit(A_list[2])
        partition_200 = self.sbm_200.block_assignments
        L_M_200 = self._compute_single_modularity_loss(dynamic_embeddings[:, 2 * hidden_dim:], A_list[2], partition_200)

        return L_M_116 + L_M_160 + L_M_200

    def _compute_single_modularity_loss(self, features, adj_matrix, partition):
        L_M = 0
        n_modules = torch.max(partition).item() + 1

        for k in range(n_modules):
            nodes_in_k = (partition == k).nonzero(as_tuple=True)[0]
            if len(nodes_in_k) == 0:
                continue

            features_k = features[:, nodes_in_k]
            features_k = features_k.view(-1, features_k.size(-1))

            adj_k = adj_matrix[nodes_in_k][:, nodes_in_k]

            similarity = torch.mm(features_k, features_k.t())
            L_M += torch.sum(similarity * adj_k.repeat(features.size(0), features.size(0))) / len(nodes_in_k)

        return -L_M

    def compute_loss(self, logits, labels, dynamic_embeddings, A_list, lambda1=0.1, lambda2=0.1):
        ce_loss = F.cross_entropy(logits, labels)

        ortho_loss = torch.norm(
            torch.mm(dynamic_embeddings, dynamic_embeddings.t()) -
            torch.eye(dynamic_embeddings.size(0), device=dynamic_embeddings.device),
            p='fro'
        )

        mod_loss = self.compute_modularity_constraint(dynamic_embeddings, A_list)

        total_loss = ce_loss + lambda1 * ortho_loss + lambda2 * mod_loss
        return total_loss, ce_loss, ortho_loss, mod_loss

