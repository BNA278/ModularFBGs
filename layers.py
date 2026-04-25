import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.mlp = nn.Linear(out_features, 2)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        x, y = e.shape[0], e.shape[1]
        rea_e = rearrange(e, 'x y z -> (x y) z')
        rea_adj = rearrange(adj, 'a b c -> (a b) c')
        attention = torch.mul(rea_e, rea_adj)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = rearrange(attention, '(x y) z -> x y z', x=x, y=y)
        attention = attention.to(self.device)
        h_prime = torch.matmul(attention, Wh)


        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class AttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, device, concat=True):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.mlp = nn.Linear(out_features, 2)
    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        x, y = e.shape[0], e.shape[1]
        rea_e = rearrange(e, 'x y z -> (x y) z')
        rea_adj = rearrange(adj, 'a b c -> (a b) c')
        attention = torch.mul(rea_e, rea_adj)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = rearrange(attention, '(x y) z -> x y z', x=x, y=y)
        attention = attention.to(self.device)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(attention)
        else:
            return attention
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.permute(0, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        edge = adj.nonzero().t()
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'