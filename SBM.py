import torch
import torch.nn as nn
import numpy as np


class StochasticBlockModel(nn.Module):
    def __init__(self, n_blocks: int, max_iter: int = 100, tol: float = 1e-6):
        super(StochasticBlockModel, self).__init__()
        self.n_blocks = n_blocks
        self.max_iter = max_iter
        self.tol = tol
        self.block_probs = None
        self.block_assignments = None

    def _initialize_parameters(self, n_nodes: int):
        self.block_assignments = torch.randint(0, self.n_blocks, (n_nodes,))
        self.block_probs = torch.ones(self.n_blocks, self.n_blocks) * 0.5

    def _compute_log_likelihood(self, adj_matrix: torch.Tensor) -> float:
        log_likelihood = 0.0
        n_nodes = adj_matrix.shape[0]

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                block_i = self.block_assignments[i]
                block_j = self.block_assignments[j]
                p = self.block_probs[block_i, block_j]

                if adj_matrix[i, j] > 0:
                    log_likelihood += torch.log(p + 1e-10)
                else:
                    log_likelihood += torch.log(1 - p + 1e-10)

        return log_likelihood.item()

    def _e_step(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        n_nodes = adj_matrix.shape[0]
        responsibilities = torch.zeros(n_nodes, self.n_blocks)

        for i in range(n_nodes):
            for k in range(self.n_blocks):
                log_prob = 0.0
                for j in range(n_nodes):
                    if i == j:
                        continue
                    block_j = self.block_assignments[j]
                    p = self.block_probs[k, block_j]
                    if adj_matrix[i, j] > 0:
                        log_prob += torch.log(p + 1e-10)
                    else:
                        log_prob += torch.log(1 - p + 1e-10)
                responsibilities[i, k] = log_prob

        return torch.softmax(responsibilities, dim=1)

    def _m_step(self, adj_matrix: torch.Tensor, responsibilities: torch.Tensor):
        n_nodes = adj_matrix.shape[0]
        self.block_assignments = torch.argmax(responsibilities, dim=1)

        new_block_probs = torch.zeros(self.n_blocks, self.n_blocks)
        block_counts = torch.zeros(self.n_blocks, self.n_blocks)

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                block_i = self.block_assignments[i]
                block_j = self.block_assignments[j]
                new_block_probs[block_i, block_j] += adj_matrix[i, j]
                new_block_probs[block_j, block_i] += adj_matrix[i, j]
                block_counts[block_i, block_j] += 1
                block_counts[block_j, block_i] += 1

        block_counts = torch.where(block_counts == 0, torch.ones_like(block_counts), block_counts)
        self.block_probs = new_block_probs / block_counts

    def fit(self, adj_matrix: torch.Tensor, n_init: int = 5):
        n_nodes = adj_matrix.shape[0]
        best_ll = -np.inf

        for init in range(n_init):
            self._initialize_parameters(n_nodes)
            prev_ll = -np.inf

            for iteration in range(self.max_iter):
                responsibilities = self._e_step(adj_matrix)
                self._m_step(adj_matrix, responsibilities)
                current_ll = self._compute_log_likelihood(adj_matrix)

                if abs(current_ll - prev_ll) < self.tol:
                    break
                prev_ll = current_ll

            if current_ll > best_ll:
                best_ll = current_ll
                best_assignments = self.block_assignments.clone()
                best_probs = self.block_probs.clone()

        self.block_assignments = best_assignments
        self.block_probs = best_probs
        return self