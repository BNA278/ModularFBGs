from GNNMA.layers import GraphAttentionLayer, SpGraphAttentionLayer, AttentionLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
def restore(sigma, u, v, K):  # 奇异值、左特征向量、右特征向量
    m = len(u)
    n = len(v[0])
    a = torch.zeros((m, n)).to('cpu')
    for k in range(K):
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        uk = torch.from_numpy(uk)
        vk = torch.from_numpy(vk)
        a += sigma[k] * torch.mm(uk, vk)  # 前 k 个奇异值的加和
    return a

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.conv1=nn.Conv1d(in_channels=116,out_channels=116,kernel_size=1,padding=0)
        self.conv2=nn.Conv1d(in_channels=116,out_channels=116,kernel_size=1,padding=0)

        # self.conv1=nn.Conv1d(in_channels=200,out_channels=200,kernel_size=1,padding=0)
        # self.conv2=nn.Conv1d(in_channels=200,out_channels=200,kernel_size=1,padding=0)

        # self.conv1 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=1, padding=0)
        # self.conv2 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=1, padding=0)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, Z, X, perform_svd):
        K=self.conv1(X.permute(0,2,1))# BS,x_c,x_dim
        Q=K.permute(0,2,1)# BS,x_dim,x_c
        V=self.conv2(Z.permute(0,2,1))# Bs,z_c,z_dim
        attention=self.softmax(torch.matmul(Q,K))#BS,x_dim,x_dim
        out=torch.bmm(attention,V).permute(0,2,1)#BS,z_dim,z_c
        if perform_svd:
            list = []
            for x in out:
                k = 6
                x = x.cpu()
                x = x.requires_grad_()
                x = x.detach()
                u, s, v =  svds(x, k=k)

                out = restore(s, u, v, k)
                list.append(out)
            out = torch.stack(list, dim=0)
            out = out.to('cuda:0')
            return out
        else:
            return out

class NEResGCN(nn.Module):
    def __init__(self,layer):
        super(NEResGCN,self).__init__()
        self.layer =layer
        self.relu  =nn.ReLU()
        self.atten =nn.ModuleList([Attention() for i in range(layer)])
        self.norm_n=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.norm_e=nn.ModuleList([nn.BatchNorm1d(116) for i in range(layer)])
        self.node_w=nn.ParameterList([nn.Parameter(torch.randn((116,116),dtype=torch.float32)) for i in range(layer)])
        self.edge_w=nn.ParameterList([nn.Parameter(torch.randn((116,116),dtype=torch.float32)) for i in range(layer)])
        self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(116*116,32*2),nn.ReLU(),nn.BatchNorm1d(32*2)) for i in range(layer+1)])
        self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(116*116,32*8),nn.ReLU(),nn.BatchNorm1d(32*8)) for i in range(layer+1)])


        # self.norm_n=nn.ModuleList([nn.BatchNorm1d(200) for i in range(layer)])
        # self.norm_e=nn.ModuleList([nn.BatchNorm1d(200) for i in range(layer)])
        # self.node_w=nn.ParameterList([nn.Parameter(torch.randn((200,200),dtype=torch.float32)) for i in range(layer)])
        # self.edge_w=nn.ParameterList([nn.Parameter(torch.randn((200,200),dtype=torch.float32)) for i in range(layer)])
        # self.line_n=nn.ModuleList([nn.Sequential(nn.Linear(200*200,32*2),nn.ReLU(),nn.BatchNorm1d(32*2)) for i in range(layer+1)])
        # self.line_e=nn.ModuleList([nn.Sequential(nn.Linear(200*200,32*9),nn.ReLU(),nn.BatchNorm1d(32*9)) for i in range(layer+1)])

        # self.norm_n = nn.ModuleList([nn.BatchNorm1d(160) for i in range(layer)])
        # self.norm_e = nn.ModuleList([nn.BatchNorm1d(160) for i in range(layer)])
        # self.node_w = nn.ParameterList(
        #     [nn.Parameter(torch.randn((160, 160), dtype=torch.float32)) for i in range(layer)])
        # self.edge_w = nn.ParameterList(
        #     [nn.Parameter(torch.randn((160, 160), dtype=torch.float32)) for i in range(layer)])
        # self.line_n = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(160 * 160, 64 * 2), nn.ReLU(), nn.BatchNorm1d(64 * 2)) for i in range(layer + 1)])
        # self.line_e = nn.ModuleList(
        #     [nn.Sequential(nn.Linear(160 * 160, 64 * 8), nn.ReLU(), nn.BatchNorm1d(64 * 8)) for i in range(layer + 1)])

        self.clase =nn.Sequential(nn.Linear(32*10*(self.layer+1),1024),nn.Dropout(0.2),nn.Sigmoid(),
                                   nn.Linear(1024,2))

        # self.clase = nn.Sequential(nn.Linear(128 * 4 * (self.layer + 1), 1024), nn.Dropout(0.2), nn.ReLU(), nn.Linear(1024, 2))
        # self.clase = nn.Sequential(nn.Linear(128 * 4 * (self.layer + 1), 2),nn.Dropout(0.2), nn.ReLU())
        self.ones=nn.Parameter(torch.ones((116),dtype=torch.float32),requires_grad=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def normalized(self, Z):
        n = Z.size()[0]
        A = Z[0, :, :]
        A = A + torch.diag(self.ones)
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -1))
        A = D.mm(A).reshape(1, 116, 116)
        for i in range(1, n):
            A1 = Z[i, :, :] + torch.diag(self.ones)
            d = A1.sum(1)
            D = torch.diag(torch.pow(d, -1))
            A1 = D.mm(A1).reshape(1, 116, 116)
            A = torch.cat((A, A1), 0)
        return A

    def update_A(self, Z):
        n = Z.size()[0]
        A = Z[0, :, :]
        Value, _ = torch.topk(torch.abs(A.view(-1)), int(116 * 116 * 0.2))
        A = (torch.abs(A) >= Value[-1]) + torch.tensor(0, dtype=torch.float32)
        A = A.reshape(1, 116, 116)
        for i in range(1, n):
            A2 = Z[i, :, :]
            Value, _ = torch.topk(torch.abs(A2.view(-1)), int(116 * 116 * 0.2))
            A2 = (torch.abs(A2) >= Value[-1]) + torch.tensor(0, dtype=torch.float32)
            A2 = A2.reshape(1, 116, 116)
            A = torch.cat((A, A2), 0)
        return A

    def forward(self, X, Z):
        n = X.size()[0]

        XX = self.line_n[0](X.view(n, -1))
        ZZ = self.line_e[0](Z.view(n, -1))
        for i in range(self.layer):
            atten = self.atten[i]
            if i == 0:
                A = atten(Z, X, perform_svd=True)
            else:
                A = atten(Z, X, perform_svd=False)
            # A = self.atten[i](Z, X)
            Z1 = torch.matmul(A, Z)
            Z2 = torch.matmul(Z1, self.edge_w[i])
            Z = self.relu(self.norm_e[i](Z2)) + Z
            ZZ = torch.cat((ZZ, self.line_e[i + 1](Z.view(n, -1))), dim=1)
            X1 = torch.matmul(A, X)
            X1 = torch.matmul(X1, self.node_w[i])
            X = self.relu(self.norm_n[i](X1)) + X
            # X.register_hook(grad_X_hook)
            # feat_X_hook(X)
            XX = torch.cat((XX, self.line_n[i + 1](X.view(n, -1))), dim=1)
        XZ = torch.cat((XX, ZZ), 1)
        # print(XX.shape)
        # print(ZZ.shape)
        y = self.clase(XZ)
        # print(self.clase[0].weight)
        return y


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, device):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.device = device

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, device=self.device, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):   # nfeat=116
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, device=self.device, concat=False)
                           # nhid*nheads = hidden*nb_heads=（8*8），nclass=2

        self.adj = AttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, device=self.device, concat=False)

        self.line_e1 = nn.Linear(116 * 116, 1024)
        self.line_e2 = nn.Linear(1024, 2)
        self.line_e3 = nn.Linear(116, 2)
        self.linear = nn.Linear(118,2)
        # self.linear1 = nn.Linear(116, 64)
        # self.linear2 = nn.Linear(464, 64)

        self.line_n1 = nn.Linear(232, 116)
        self.line_n2 = nn.Linear(116,64)
        self.Dropout = nn.Dropout(0.5)
        self.linear3 = nn.Linear(64,2)
        self.relu = nn.ReLU()
    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training) #4*116*175 对特征矩阵随季剔除
        x_adj = self.adj(x, adj)
        x_adj = x_adj.view(x_adj.size(0), -1)
        x_adj = self.line_e1(x_adj)
        x_adj = self.relu(x_adj)
        x_adj = self.line_e2(x_adj)
        x_adj = self.relu(x_adj)
        return F.log_softmax(x_adj, dim=1) # 4*2


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



