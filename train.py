import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from GNNMA.models import NEResGCN

from collections import Counter
import argparse

import random
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=127, help='Random seed.')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class data_load(Dataset):
    def __init__(self, data_tensor,data_labels,data_adj):
        self.data_tensor = data_tensor
        self.data_labels = data_labels
        self.data_adj = data_adj
        self.features_dict={}
        self.labels_dict={}
        self.adj_dict = {}
        self.x_data = scio.loadmat(self.data_tensor)
        self.y_data = scio.loadmat(self.data_labels)
        self.adj_data = scio.loadmat(self.data_adj)
        self.features = torch.from_numpy(self.x_data['feature'][:].transpose(0,2,1)).to(torch.float32)
        self.features_dim = self.features.shape[2]  # shape=116
        self.labels = torch.from_numpy(self.y_data['label'][:].transpose(1,0)).to(torch.float32)
        self.class_num = len(self.labels.unique())
        self.adj=torch.from_numpy(self.adj_data['corr'][:]).to(torch.float32)
        for id in range(self.features.shape[0]):
            self.features_dict[id]=self.features[id,:,:]
            self.labels_dict[id]=self.labels[id,:]
            self.adj_dict[id]=self.adj[id,:,:]
        self.id_list=list(self.features_dict.keys())

    def __len__(self):   #为了求len（dataset）
        return len(self.id_list)
        # return self.len

    def __getitem__(self, index):  #为了支持下标操作，即索引dataset[index]
        id = self.id_list[index]
        bold = self.features_dict[index]
        label = self.labels_dict[index]
        adj = self.adj_dict[index]

        return {'id':id,'bold':bold,'label':label,'adj':adj}
class data_load(Dataset):
    def __init__(self, data_tensor, data_labels, data_adj):
        self.data_tensor = data_tensor
        self.data_labels = data_labels
        self.data_adj = data_adj


        self.x_data = scio.loadmat(self.data_tensor)
        self.y_data = scio.loadmat(self.data_labels)
        self.adj_data = scio.loadmat(self.data_adj)

        self.features = torch.from_numpy(self.x_data['feature'][:].transpose(0, 2, 1)).to(torch.float32)
        self.features_dim = self.features.shape[2]
        self.labels = torch.from_numpy(self.y_data['label'][:].transpose(1, 0)).to(torch.float32)
        self.class_num = len(self.labels.unique())
        self.adj = torch.from_numpy(self.adj_data['corr'][:]).to(torch.float32)
        class_counts = Counter(self.labels.numpy().flatten())
        target_count = min(class_counts.values())
        self.indices = []
        for class_label in class_counts.keys():
            class_indices = torch.where(self.labels.flatten() == class_label)[0]
            selected_indices = class_indices[torch.randperm(len(class_indices))[:target_count]]
            self.indices.append(selected_indices)
        self.indices = torch.cat(self.indices)
        self.features = self.features[self.indices]
        self.labels = self.labels[self.indices]
        self.adj = self.adj[self.indices]
        self.features_dict = {i: self.features[i] for i in range(len(self.features))}
        self.labels_dict = {i: self.labels[i] for i in range(len(self.labels))}
        self.adj_dict = {i: self.adj[i] for i in range(len(self.adj))}
        self.id_list = list(self.features_dict.keys())

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id = self.id_list[index]
        bold = self.features_dict[index]
        label = self.labels_dict[index]
        adj = self.adj_dict[index]
        return {'id': id, 'bold': bold, 'label': label, 'adj': adj}

class data_save(Dataset):
    def __init__(self, features, labels, adj):
        self.features = features
        self.adj = adj
        self.labels = labels

        self.f_dict = {}
        self.l_dict = {}
        self.a_dict = {}
        for id in range(self.features.shape[0]):
            self.f_dict[id]=self.features[id,:,:]
            self.l_dict[id]=self.labels[id,:]
            self.a_dict[id]=self.adj[id,:,:]
        self.list_id=list(self.f_dict.keys())
    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index):
        id = self.list_id[index]
        bold = self.f_dict[index]
        label = self.l_dict[index]
        adj = self.a_dict[index]
        return {'id':id,'bold':bold,'label':label,'adj':adj}


features_path = r'E:\Dataset\ABIDE\NYU48.mat'
labels_path = r'E:\Dataset\ABIDE\NYU48.mat'
adj_path = r'E:\Dataset\ABIDE\NYU48.mat'

dataset =data_load(features_path,labels_path,adj_path)
S_Test_labels = []
S_Test_y_pred = []
Y_score = []
Score = []
k_fold=5
kf = KFold(n_splits=k_fold,shuffle=True,random_state=788)
acc=0.0
fold = 1
for train_idx , test_idx in kf.split(dataset):
    train_features = [dataset[id]['bold'] for id in train_idx]
    train_features = torch.stack(train_features, dim=0)
    test_features = [dataset[id]['bold'] for id in test_idx]
    test_features = torch.stack(test_features, dim=0)
    train_adj = [dataset[id]['adj'] for id in train_idx]
    train_adj = torch.stack(train_adj, dim=0)
    test_adj = [dataset[id]['adj'] for id in test_idx]
    test_adj = torch.stack(test_adj, dim=0)
    train_labels = [dataset[id]['label'] for id in train_idx]
    train_labels = torch.stack(train_labels, dim=0)
    test_labels = [dataset[id]['label'] for id in test_idx]
    test_labels = torch.stack(test_labels, dim=0)

    model = NEResGCN(5)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()
    Labels = []
    Y_pred = []
    epoch = 1
    for epoch in range(args.epochs):
        train_dataset = data_save(train_features, train_labels, train_adj)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=6,
                                  shuffle=True)
        model.train()
        loss_accumulate = 0.0
        for x in train_loader:
            features = x['bold'].to(device)
            adj = x['adj'].to(device)
            labels = x['label'].squeeze().long().to(device)
            if labels.unsqueeze(0).size()==(1,): Labels.append(labels.tolist())
            else: Labels.extend(labels.tolist())
            y_pred = model(features, adj)  # y_pred的格式为batch*2
            if labels.unsqueeze(0).size()==(1,): labels=labels.unsqueeze(0)
            loss_train = loss(y_pred, labels)
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_accumulate += loss_train
            y_pred = y_pred.argmax(1)
            Y_pred.extend(y_pred.tolist())
        true_list = np.array([Labels[i] - Y_pred[i] for i in range(len(Labels))])
        true_list[true_list != 0] = 1
        err = true_list.sum()
        acc1 = (len(Labels) - err) / len(Labels)
        epoch = epoch + 1
        print('fold', fold, 'epoch', epoch, 'loss_train', loss_accumulate.item(), 'acc_train', acc1)

    model.eval()
    with torch.no_grad():
        test_dataset = data_save(test_features, test_labels, test_adj)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=True)


        Test_labels = []
        Test_y_pred = []
        for x in test_loader:
            features = x['bold'].to(device)
            adj = x['adj'].to(device)
            test_labels = x['label'].squeeze().long().to(device)

            if test_labels.unsqueeze(0).size() == (1,):
                S_Test_labels.append(test_labels.tolist())
            else:
                S_Test_labels.extend(test_labels.tolist())

            Test_labels.append(test_labels.item())
            test_y_pred = model(features, adj)
            softmax = nn.Softmax(dim=1)
            y_score = softmax(test_y_pred)
            y_score = y_score.tolist()[0][1]
            Y_score.append(y_score)
            test_y_pred = test_y_pred.argmax(1)
            if test_y_pred.unsqueeze(0).size() == (1,):
                S_Test_y_pred.append(test_y_pred.tolist())
            else:
                S_Test_y_pred.extend(test_y_pred.tolist())
            Test_y_pred.append(test_y_pred.item())
        true_list = np.array([Test_labels[i] - Test_y_pred[i] for i in range(len(Test_labels))])
        true_list[true_list != 0] = 1
        err = true_list.sum()
        acc2 = (len(Test_labels) - err) / len(Test_labels)
        print('test_result', '\n', 'test_acc:', acc2)
        acc += acc2
        fold += 1

acc = acc/k_fold

tn, fp, fn, tp = confusion_matrix(S_Test_labels, S_Test_y_pred).ravel()
Sens = tp / (tp + fn)
Spec = tn / (tn + fp)
print('Sens', '\n', 'Sens:', Sens)
print('Spec', '\n', 'Spec:', Spec)
print('f1_score', '\n', 'f1_score:', f1_score(S_Test_labels, S_Test_y_pred, average='binary'))
print('final acc:',acc)
Acc = (tp+tn)/(tp + fn + tn + fp)
print('final Acc:',Acc)
fpr, tpr, thresholds = roc_curve(S_Test_labels, Y_score)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()