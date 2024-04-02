import os
import glob
import json
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.datasets import MoleculeNet, TUDataset
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.loader import DataLoader
from mol_dataset import MoleculeDataset
import sklearn.preprocessing as preprocessing
import random


def undirected_graph(data):
    data.edge_index = torch.cat(
        [torch.stack([data.edge_index[1], data.edge_index[0]], dim=0), data.edge_index],
        dim=1,
    )
    return data


def split(data, batch):
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    slices = dict()
    slices["x"] = node_slice
    slices["edge_index"] = edge_slice
    slices["y"] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f"_{name}.txt")
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1 : -4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1 : -5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix + "_node_features.pkl"), "rb") as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, "edge_index")
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, "node_indicator") - 1  # from zero
    y: np.array = read_file(folder, prefix, "graph_labels")
    y: torch.tensor = torch.tensor(y, dtype=torch.long)

    supplement = dict()
    if "split_indices" in names:
        split_indices: np.array = read_file(folder, prefix, "split_indices")
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement["split_indices"] = split_indices
    if "sentence_tokens" in names:
        with open(os.path.join(folder, prefix + "_sentence_tokens.json")) as f:
            sentence_tokens: dict = json.load(f)
        supplement["sentence_tokens"] = sentence_tokens

    data = Data(x=x, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices, supplement


def read_syn_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), "rb") as f:
        (
            adj,
            features,
            y_train,
            y_val,
            y_test,
            train_mask,
            val_mask,
            test_mask,
            edge_label_matrix,
        ) = pickle.load(f)

    x = torch.from_numpy(features).float()
    y = (
        train_mask.reshape(-1, 1) * y_train
        + val_mask.reshape(-1, 1) * y_val
        + test_mask.reshape(-1, 1) * y_test
    )
    y = torch.from_numpy(np.where(y)[1])
    edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
    data = Data(x=x, y=y, edge_index=edge_index)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    return data


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), "rb") as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        data_list.append(
            Data(
                x=torch.from_numpy(node_features[graph_idx]).float(),
                edge_index=dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0],
                y=torch.from_numpy(np.where(graph_labels[graph_idx])[0]),
            )
        )

    return data_list


def get_dataset(dataset_dir, dataset_name, task=None):
    sync_dataset_dict = {
        "BA_2Motifs".lower(): "BA_2Motifs",
        "BA_Shapes".lower(): "BA_shapes",
        "BA_Community".lower(): "BA_Community",
        "Tree_Cycle".lower(): "Tree_Cycle",
        "Tree_Grids".lower(): "Tree_Grids",
    }
    sentigraph_names = ["Graph-SST2", "Graph-Twitter", "Graph-SST5"]
    sentigraph_names = [name.lower() for name in sentigraph_names]
    molecule_net_dataset_names = [name.lower() for name in MoleculeNet.names.keys()]
    # TU_dataset_names = [name.lower() for name in TUDataset.names.keys()]

    if dataset_name.lower() == "MUTAG".lower():
        return load_MUTAG(dataset_dir, "MUTAG")
    elif dataset_name.lower() == "Mutagenicity".lower():
        return load_Mutagenicity(dataset_dir, "Mutagenicity")
    elif dataset_name.lower() in sync_dataset_dict.keys():
        sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
        return load_syn_data(dataset_dir, sync_dataset_filename)
    elif dataset_name.lower() in molecule_net_dataset_names:
        return load_MolecueNet(dataset_dir, dataset_name, task)
    elif dataset_name.lower() in sentigraph_names:
        return load_SeniGraph(dataset_dir, dataset_name)
    elif dataset_name.lower() in ["cox2", "enzymes", "bzr", "nci1"]:
        return TUDataset(dataset_dir, dataset_name)
    else:
        raise NotImplementedError


class MUTAGDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name.upper()
        super(MUTAGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.slices["x"]) - 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self):
        return [
            "MUTAG_A",
            "MUTAG_graph_labels",
            "MUTAG_graph_indicator",
            "MUTAG_node_labels",
        ]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        with open(os.path.join(self.raw_dir, "MUTAG_node_labels.txt"), "r") as f:
            nodes_all_temp = f.read().splitlines()
            nodes_all = [int(i) for i in nodes_all_temp]

        adj_all = np.zeros((len(nodes_all), len(nodes_all)))
        with open(os.path.join(self.raw_dir, "MUTAG_A.txt"), "r") as f:
            adj_list = f.read().splitlines()
        for item in adj_list:
            lr = item.split(", ")
            l = int(lr[0])
            r = int(lr[1])
            adj_all[l - 1, r - 1] = 1

        with open(os.path.join(self.raw_dir, "MUTAG_graph_indicator.txt"), "r") as f:
            graph_indicator_temp = f.read().splitlines()
            graph_indicator = [int(i) for i in graph_indicator_temp]
            graph_indicator = np.array(graph_indicator)

        with open(os.path.join(self.raw_dir, "MUTAG_graph_labels.txt"), "r") as f:
            graph_labels_temp = f.read().splitlines()
            graph_labels = [int(i) for i in graph_labels_temp]

        data_list = []
        for i in range(1, 189):
            idx = np.where(graph_indicator == i)
            graph_len = len(idx[0])
            adj = adj_all[idx[0][0] : idx[0][0] + graph_len, idx[0][0] : idx[0][0] + graph_len]
            label = int(graph_labels[i - 1] == 1)
            feature = nodes_all[idx[0][0] : idx[0][0] + graph_len]
            nb_clss = 7
            targets = np.array(feature).reshape(-1)
            one_hot_feature = np.eye(nb_clss)[targets]
            data_example = Data(
                x=torch.from_numpy(one_hot_feature).float(),
                edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                y=label,
            )
            data_list.append(data_example)

        torch.save(self.collate(data_list), self.processed_paths[0])


class Mutagenicity(InMemoryDataset):

    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Mutagenicity.zip"

    splits = ["training", "evaluation", "testing"]

    def __init__(self, root, mode="testing", transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(Mutagenicity, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index("{}.pt".format(mode))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return [
            "Mutagenicity/" + i
            for i in [
                "Mutagenicity_A.txt",
                "Mutagenicity_edge_labels.txt",
                "Mutagenicity_graph_indicator.txt",
                "Mutagenicity_graph_labels.txt",
                "Mutagenicity_node_labels.txt",
            ]
        ]

    @property
    def processed_file_names(self):
        return ["training.pt", "evaluation.pt", "testing.pt"]

    def download(self):

        if os.path.exists(osp.join(self.raw_dir, "Mutagenicity")):
            print("Using existing data in folder Mutagenicity")
            return

        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):

        edge_index = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[0]), delimiter=",").T
        edge_index = torch.from_numpy(edge_index - 1.0).to(torch.long)  # node idx from 0

        edge_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[1]))
        encoder = preprocessing.OneHotEncoder().fit(np.unique(edge_label).reshape(-1, 1))
        edge_attr = encoder.transform(edge_label.reshape(-1, 1)).toarray()
        edge_attr = torch.Tensor(edge_attr)

        node_label = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[-1]))
        encoder = preprocessing.OneHotEncoder().fit(np.unique(node_label).reshape(-1, 1))
        x = encoder.transform(node_label.reshape(-1, 1)).toarray()
        x = torch.Tensor(x)

        z = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[2]), dtype=int)

        y = np.loadtxt(osp.join(self.raw_dir, self.raw_file_names[3]))
        y = torch.unsqueeze(torch.LongTensor(y), 1).long()
        num_graphs = len(y)
        total_edges = edge_index.size(1)
        begin = 0

        data_list = []
        for i in range(num_graphs):

            perm = np.where(z == i + 1)[0]
            bound = max(perm)
            end = begin
            for end in range(begin, total_edges):
                if int(edge_index[0, end]) > bound:
                    break

            data = Data(
                x=x[perm],
                y=y[i],
                z=node_label[perm],
                edge_index=edge_index[:, begin:end] - int(min(perm)),
                edge_attr=edge_attr[begin:end],
                name="mutag_%d" % i,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            begin = end
            data_list.append(data)

        assert len(data_list) == 4337

        random.shuffle(data_list)
        torch.save(self.collate(data_list[1000:]), self.processed_paths[0])
        torch.save(self.collate(data_list[500:1000]), self.processed_paths[1])
        torch.save(self.collate(data_list[:500]), self.processed_paths[2])


class SentiGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    def set_y(self, y):
        self.data.y = y

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [
            "node_features",
            "node_indicator",
            "sentence_tokens",
            "edge_index",
            "graph_labels",
            "split_indices",
        ]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement = read_sentigraph_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SynGraphDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        data = read_syn_data(self.raw_dir, self.name)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class BA2MotifDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(BA2MotifDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return [f"{self.name}.pkl"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        data_list = read_ba2motif_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])


def load_MUTAG(dataset_dir, dataset_name):
    """188 molecules where label = 1 denotes mutagenic effect"""
    dataset = MUTAGDataset(root=dataset_dir, name=dataset_name)
    return dataset


def load_Mutagenicity(dataset_dir, dataset_name):
    dataset = Mutagenicity(root=dataset_dir)
    return dataset


def load_syn_data(dataset_dir, dataset_name):
    """The synthetic dataset"""
    if dataset_name.lower() == "BA_2Motifs".lower():
        dataset = BA2MotifDataset(root=dataset_dir, name=dataset_name)
    else:
        dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


def load_MolecueNet(dataset_dir, dataset_name, task=None):
    """Attention the multi-task problems not solved yet"""

    dataset = MoleculeNet(root="./datasets", name="BBBP")
    print(dataset.data)
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y[:, 0].long()
    print(dataset.data)

    return dataset


def load_SeniGraph(dataset_dir, dataset_name):
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name)
    return dataset


def get_dataloader(dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=5):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, "supplement"):
        assert "split_indices" in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement["split_indices"]
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        # fix dataloader bug(BA_shapes)
        if len(dataset) <= 1:
            length = len(dataset.data.y)
        else:
            length = len(dataset)
        num_train = int(data_split_ratio[0] * length)
        num_eval = int(data_split_ratio[1] * length)
        num_test = length - num_train - num_eval
        train, eval, test = random_split(
            dataset,
            lengths=[num_train, num_eval, num_test],
            generator=torch.Generator().manual_seed(seed),
        )

    dataloader = dict()
    dataloader["train"] = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader["eval"] = DataLoader(eval, batch_size=batch_size, shuffle=False, drop_last=False)
    dataloader["test"] = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader
