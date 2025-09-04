import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from get_dataset import BrainGraphDataset, collate_fn
from model import ModularConstrainedDynamicGNN
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def train_model(model, dataloader, optimizer, device, lambda1=0.1, lambda2=0.1):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_ortho_loss = 0
    total_mod_loss = 0

    all_preds = []
    all_labels = []

    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        batch_labels = batch_labels.to(device)

        x_list = []
        edge_index_list = []
        A_list = []

        batch_size = len(batch_data['AAL'])
        num_windows = len(batch_data['AAL'][0])

        for t in range(num_windows):
            window_x_list = []
            window_edge_index_list = []
            window_A_list = []

            for atlas_name in ['AAL', 'DOS160', 'CC200']:
                if atlas_name in batch_data:
                    atlas_graphs = []
                    atlas_adj = []

                    for subject_idx in range(batch_size):
                        graph = batch_data[atlas_name][subject_idx][t]

                        node_features = graph.x.to(device)
                        atlas_graphs.append(node_features)

                        adj_matrix = torch.zeros_like(graph.x)
                        adj_matrix[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr
                        atlas_adj.append(adj_matrix.to(device))

                    window_x_list.append(torch.stack(atlas_graphs))
                    window_A_list.append(torch.stack(atlas_adj))

                    edge_index = batch_data[atlas_name][0][t].edge_index.to(device)
                    window_edge_index_list.append(edge_index)

            x_list.append(torch.stack(window_x_list, dim=1))
            A_list.append(torch.stack(window_A_list, dim=1))
            edge_index_list.append(window_edge_index_list)

        x_list = torch.stack(x_list, dim=0)
        A_list = torch.stack(A_list, dim=0)

        x_list = x_list.permute(1, 0, 2, 3, 4)
        A_list = A_list.permute(1, 0, 2, 3, 4)

        optimizer.zero_grad()

        batch_logits = []
        batch_dynamic_embs = []

        for i in range(batch_size):
            sample_x_list = x_list[i]
            sample_A_list = [A_list[i, t] for t in range(num_windows)]

            sample_edge_index_list = edge_index_list

            logits, dynamic_emb = model(sample_x_list, sample_edge_index_list, sample_A_list)
            batch_logits.append(logits)
            batch_dynamic_embs.append(dynamic_emb)

        batch_logits = torch.stack(batch_logits)
        batch_dynamic_embs = torch.stack(batch_dynamic_embs)

        loss, ce_loss, ortho_loss, mod_loss = model.compute_loss(
            batch_logits, batch_labels, batch_dynamic_embs, A_list, lambda1, lambda2
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_ortho_loss += ortho_loss.item()
        total_mod_loss += mod_loss.item()

        preds = torch.argmax(batch_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, '
                  f'Ortho: {ortho_loss.item():.4f}, Mod: {mod_loss.item():.4f}')

    accuracy = accuracy_score(all_labels, all_preds)

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_ortho_loss = total_ortho_loss / num_batches
    avg_mod_loss = total_mod_loss / num_batches

    return avg_loss, accuracy, avg_ce_loss, avg_ortho_loss, avg_mod_loss


def test_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_labels = batch_labels.to(device)

            logits, _ = model(batch_data['x_list'].to(device),
                              [e.to(device) for e in batch_data['edge_index_list']],
                              [a.to(device) for a in batch_data['A_list']])

            loss = F.cross_entropy(logits, batch_labels)
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    avg_loss = total_loss / len(dataloader)

    acc = accuracy_score(all_labels, all_preds)

    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(all_labels, all_preds, average='binary')

    auc = roc_auc_score(all_labels, all_probs[:, 1])

    return {
        'ACC': acc,
        'SEN': sen,
        'SPE': spe,
        'F1': f1,
        'AUC': auc,
        'avg_loss': avg_loss
    }


def training_loop(model, train_loader, test_loader, optimizer, device, num_epochs=40,
                  lambda1=0.1, lambda2=0.1):

    for epoch in range(num_epochs):
        train_loss, train_acc, ce_loss, ortho_loss, mod_loss = train_model(
            model, train_loader, optimizer, device, lambda1, lambda2
        )

        results = test_model(model, test_loader, device)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print("Train Loss: ", train_loss)
        print("Train Acc: ", train_acc)
        print(f'CE Loss: {ce_loss:.4f}, Ortho Loss: {ortho_loss:.4f}, Mod Loss: {mod_loss:.4f}')
        print("Test Loss: ", results['avg_loss'])
        print("Test Acc: ", results['ACC'])
        print("Test Sen: ", results['SEN'])
        print("Test Spe: ", results['SPE'])
        print("Test F1: ", results['F1'])
        print("Test Auc: ", results['AUC'])
        print('-' * 50)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_dims = [116, 160, 200]
    hidden_dim = 64
    num_modules = 10
    nhead = 4
    num_layers = 1
    dropout = 0.5
    lr = 0.001
    num_epochs = 40
    batch_size = 8
    lambda1 = 0.1
    lambda2 = 0.1
    window_size = 40
    stride = 5
    sparsity = 0.3

    data_path = ''

    dataset = BrainGraphDataset(
        data_path=data_path,
        atlas_names=['AAL', 'DOS160', 'CC200'],
        window_size=window_size,
        stride=stride,
        sparsity=sparsity
    )

    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[dataset[i][1].item() for i in range(len(dataset))]
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1
    )

    model = ModularConstrainedDynamicGNN(
        node_dims=node_dims,
        hidden_dim=hidden_dim,
        num_modules=num_modules,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    training_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        lambda1=lambda1,
        lambda2=lambda2
    )


if __name__ == "__main__":
    main()

