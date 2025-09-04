import torch
from torch_geometric.data import Data
import scipy.io


def create_sliding_windows(bold_signal, window_size=40, stride=5):
    num_regions, total_time = bold_signal.shape
    windows = []

    for start_idx in range(0, total_time - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = bold_signal[:, start_idx:end_idx]
        windows.append(window)

    return torch.stack(windows)


def compute_pearson_correlation(x):
    x_centered = x - torch.mean(x, dim=1, keepdim=True)

    covariance = torch.mm(x_centered, x_centered.t())

    std_dev = torch.std(x, dim=1, keepdim=True)
    std_matrix = torch.mm(std_dev, std_dev.t())

    correlation = covariance / (std_matrix + 1e-10)

    correlation = (correlation + correlation.t()) / 2
    correlation = torch.clamp(correlation, -1.0, 1.0)
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)

    return correlation


def threshold_matrix(matrix, sparsity=0.3):
    triu_indices = torch.triu_indices(matrix.size(0), matrix.size(1), offset=1)
    upper_triangle = matrix[triu_indices[0], triu_indices[1]]

    k = int(sparsity * upper_triangle.numel())
    if k > 0:
        threshold = torch.topk(upper_triangle.abs().flatten(), k).values[-1]
    else:
        threshold = 0.0

    sparse_matrix = torch.zeros_like(matrix)
    mask = matrix.abs() >= threshold
    sparse_matrix[mask] = matrix[mask]

    sparse_matrix = (sparse_matrix + sparse_matrix.t()) / 2

    return sparse_matrix


def construct_dynamic_fbgs(bold_signals, atlas_names=['AAL', 'DOS160', 'CC200'],
                           window_size=40, stride=5, sparsity=0.3):
    dynamic_fbgs = {}

    for atlas_name in atlas_names:
        if atlas_name not in bold_signals:
            continue

        bold_data = bold_signals[atlas_name]
        num_subjects = bold_data.shape[0]
        subject_fbgs = []

        for subject_idx in range(num_subjects):
            subject_bold = bold_data[subject_idx]

            windows = create_sliding_windows(subject_bold, window_size, stride)
            num_windows = windows.shape[0]

            window_graphs = []
            for window_idx in range(num_windows):
                window_data = windows[window_idx]

                correlation_matrix = compute_pearson_correlation(window_data)

                thresholded_matrix = threshold_matrix(correlation_matrix, sparsity)

                edge_index = (thresholded_matrix != 0).nonzero(as_tuple=False).t()
                edge_weight = thresholded_matrix[thresholded_matrix != 0]

                graph_data = Data(
                    x=thresholded_matrix,
                    edge_index=edge_index,
                    edge_attr=edge_weight
                )

                window_graphs.append(graph_data)

            subject_fbgs.append(window_graphs)

        dynamic_fbgs[atlas_name] = subject_fbgs

    return dynamic_fbgs


class BrainGraphDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, atlas_names=['AAL', 'DOS160', 'CC200'],
                 window_size=40, stride=5, sparsity=0.3):
        super(BrainGraphDataset, self).__init__()

        data = scipy.io.loadmat(data_path)

        self.bold_signals = {}
        self.labels = {}

        for atlas_name in atlas_names:
            if atlas_name in data:
                self.bold_signals[atlas_name] = torch.tensor(data[atlas_name]).float()
                self.labels[atlas_name] = torch.tensor(data['lab']).long()

        self.atlas_names = atlas_names
        self.window_size = window_size
        self.stride = stride
        self.sparsity = sparsity

        self.dynamic_fbgs = construct_dynamic_fbgs(
            self.bold_signals, atlas_names, window_size, stride, sparsity
        )

        first_atlas = atlas_names[0]
        self.num_subjects = len(self.dynamic_fbgs[first_atlas])

    def __len__(self):
        return self.num_subjects

    def __getitem__(self, idx):
        sample = {}
        label = None

        for atlas_name in self.atlas_names:
            if atlas_name in self.dynamic_fbgs:
                sample[atlas_name] = self.dynamic_fbgs[atlas_name][idx]

                if label is None and atlas_name in self.labels:
                    label = self.labels[atlas_name][idx]

        return sample, label


def collate_fn(batch):
    batch_samples, batch_labels = zip(*batch)

    batched_data = {}
    for atlas_name in batch_samples[0].keys():
        batched_data[atlas_name] = []

    batch_labels = torch.stack(batch_labels)

    for sample in batch_samples:
        for atlas_name, graphs in sample.items():
            batched_data[atlas_name].append(graphs)

    return batched_data, batch_labels

