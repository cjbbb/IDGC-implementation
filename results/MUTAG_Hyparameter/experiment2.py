import os
import torch
from typing import List


class ClusterParser:
    def __init__(self):
        super().__init__()
        # cluster part
        self.num_cluster = 2
        self.update_epoch = 1
        self.init_train_random = True
        self.reassign_center = True
        # draw part
        self.draw = True
        self.draw_prot = True
        self.draw_init = True
        self.draw_tsne_anlysis = True
        # pca part
        self.pca_dimension = 4
        self.auto_pca = False
        self.pca_norm = False


class TrainParser:
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.batch_size = 12
        self.weight_decay = 0.0
        self.max_epochs = 100
        self.save_epoch = 10
        self.early_stopping = 50
        self.last_layer_optimizer_lr = 1e-4  # the learning rate of the last layer
        self.joint_optimizer_lrs = {
            "features": 1e-4,
            "add_on_layers": 3e-3,
            "prototype_vectors": 3e-3,
        }  # the learning rates of the joint training optimizer
        self.warm_epochs = 0  # the number of warm epochs
        self.proj_epochs = 500  # the epoch to start mcts
        self.sampling_epochs = 100  # the epoch to start sampling edges
        self.nearest_graphs = 10  # number of graphs in projection


class DataParser:
    def __init__(self):
        super().__init__()
        self.dataset_name = "MUTAG"
        self.dataset_dir = "./datasets"
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [
            0.8,
            0.1,
            0.1,
        ]  # the ratio of training, validation and testing set for random split
        # self.data_split_ratio = None
        self.seed = 1


class ModelParser:
    def __init__(self):
        super().__init__()
        self.enable_prot = True  # whether to enable prototype training
        self.device: int = 0  # the device id
        self.model_name: str = "gin"
        self.checkpoint: str = "./checkpoint"
        self.concate: bool = True  # whether to concate the gnn features before mlp
        self.latent_dim: List[int] = [
            128,
            128,
            128,
        ]  # the hidden units for each gnn layer
        self.readout: "str" = "max"  # the graph pooling method
        self.mlp_hidden: List[int] = []  # the hidden units for mlp classifier
        self.gnn_dropout: float = 0.0  # the dropout after gnn layers
        self.dropout: float = 0.5  # the dropout after mlp layers
        self.adj_normlize: bool = True  # the edge_weight normalization for gcn conv
        self.emb_normlize: bool = False  # the l2 normalization after gnn layer node embedding
        self.num_prototypes_per_class = 5  # the num_prototypes_per_class
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3

    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.device)
        else:
            pass


class GATParser:  # hyper-parameter for gat model
    def __init__(self):
        super().__init__()
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3


class MCTSParser(DataParser, ModelParser):
    rollout: int = 10  # the rollout number
    high2low: bool = False  # expand children with different node degree ranking method
    c_puct: float = 5  # the exploration hyper-parameter
    min_atoms: int = 5  # for the synthetic dataset, change the minimal atoms to 5.
    max_atoms: int = 10
    expand_atoms: int = 10  # # of atoms to expand children

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(
            self.checkpoint, self.dataset_name, f"{self.model_name}_best.pth"
        )


class RewardParser:
    def __init__(self):
        super().__init__()
        self.reward_method: str = (  # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
            "mc_l_shapley"
        )
        self.local_raduis: int = 4  # (n-1) hops neighbors for l_shapley
        self.subgraph_building_method: str = "zero_filling"
        self.sample_num: int = 100  # sample time for monte carlo approximation


data_args = DataParser()
model_args = ModelParser()
mcts_args = MCTSParser()
reward_args = RewardParser()
train_args = TrainParser()
cluster_args = ClusterParser()

import torch
import random
import numpy as np

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


# Assumed Accuracy: 0.7333328444447703
# Real Counts: [50, 100, 0]
# Predicted Counts: [102, 48, 0]
# Number of batches: 13
# Train Epoch:92  |Loss: 0.029 | Ld: 27.288 | Acc: 0.967
# Eval Epoch: 92 | Loss: 1.196 | Acc: 0.833
# 建议降低到:  2
# /home/jianbin/anaconda3/lib/python3.9/site-packages/sklearn/manifold/_t_sne.py:982: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
#   warnings.warn(
# Assumed Accuracy: 0.7399995066669955
# Real Counts: [50, 100, 0]
# Predicted Counts: [109, 41, 0]
# Number of batches: 13
# Train Epoch:93  |Loss: 0.029 | Ld: 27.311 | Acc: 0.967
# Eval Epoch: 93 | Loss: 1.934 | Acc: 0.833
# Assumed Accuracy: 0.8499957500212499
# Real Counts: [6, 14, 0]
# Predicted Counts: [7, 13, 0]
# The best validation accuracy is 0.8333333333333334.
# best epoch is  42
# Test: | Loss: 0.584 | Acc: 0.850
# train_loss.png
