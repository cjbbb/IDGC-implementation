import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from utils.Configures import model_args
import torch.nn.init as init

# input
# GCN layer1
# GCN layer2
# GCN layer3
# Aggregation
# MLP layer1
# (Prototype Vector - > Linear Layer)
# Softmax
# output


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
    }

    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


# GCN
class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GCNNet, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = model_args.latent_dim
        self.mlp_hidden = model_args.mlp_hidden
        self.emb_normlize = model_args.emb_normlize
        self.device = torch.device("cuda:" + str(model_args.device))
        self.num_gnn_layers = len(self.latent_dim)
        self.num_mlp_layers = len(self.mlp_hidden) + 1
        self.dense_dim = self.latent_dim[-1]
        self.readout_layers = get_readout_layers(model_args.readout)
        self.top_layer_is_remove = False
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(
            GCNConv(
                input_dim,
                self.latent_dim[0],
                normalize=model_args.adj_normlize,
                weight_initializer="kaiming_uniform",
                bias_initializer="zeros",
            )
        )
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(
                GCNConv(
                    self.latent_dim[i - 1],
                    self.latent_dim[i],
                    normalize=model_args.adj_normlize,
                    weight_initializer="kaiming_uniform",
                    bias_initializer="zeros",
                )
            )
        self.gnn_non_linear = nn.ELU()

        self.mlps = nn.ModuleList()
        if self.num_mlp_layers > 1:
            self.mlps.append(
                nn.Linear(self.dense_dim * len(self.readout_layers), model_args.mlp_hidden[0])
            )
            for i in range(1, self.num_mlp_layers - 1):
                self.mlps.append(nn.Linear(self.mlp_hidden[i - 1], self.mlp_hidden[1]))
            self.mlps.append(nn.Linear(self.mlp_hidden[-1], output_dim))
        else:
            self.mlps.append(nn.Linear(self.dense_dim * len(self.readout_layers), output_dim))

        for m in self.mlps:
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

        self.dropout = nn.Dropout(model_args.dropout)
        self.Softmax = nn.Softmax(dim=-1)
        self.mlp_non_linear = nn.ReLU()

        # prototype layers
        self.enable_prot = model_args.enable_prot
        self.epsilon = 1e-4  # 防止除0的错误
        self.prototype_shape = (
            output_dim * model_args.num_prototypes_per_class,
            128,
        )  # 原型向量的形状 128是原型向量的维度
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.num_prototypes = self.prototype_shape[0]  # number of prototypes
        self.last_layer = nn.Linear(self.num_prototypes, output_dim, bias=False)  # do not use bias
        assert self.num_prototypes % output_dim == 0
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = (
                1  # 对应类别设置为1 其他类别设置为0
            )
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        """
        the incorrect strength will be actual strength if -0.5 then input -0.5
        """
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = (
            -2 * xp
            + torch.sum(x**2, dim=1, keepdim=True)
            + torch.t(torch.sum(self.prototype_vectors**2, dim=1, keepdim=True))
        )
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def forward(self, data, protgnn_plus=False, similarity=None):
        if protgnn_plus:
            logits = self.last_layer(similarity)
            probs = self.Softmax(logits)
            return logits, probs, None, None, None

        # for name, param in self.named_parameters():
        #     print(name, param.data)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_gnn_layers):
            if not self.gnn_layers[i].normalize:
                edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
                x = self.gnn_layers[i](x, edge_index, edge_weight)
            else:
                x = self.gnn_layers[i](x, edge_index)
            if self.emb_normlize:
                x = F.normalize(x, p=2, dim=-1)
            x = self.gnn_non_linear(x)

        node_emb = x

        pooled = []
        for readout in self.readout_layers:
            pooled.append(readout(x, batch))
        x = torch.cat(pooled, dim=-1)
        graph_emb = x

        if self.enable_prot:
            prototype_activations, min_distances = self.prototype_distances(x)
            logits = self.last_layer(prototype_activations)
            probs = self.Softmax(logits)
            return logits, probs, node_emb, graph_emb, min_distances
        else:
            if self.top_layer_is_remove:
                return None, None, node_emb, graph_emb, []

            for i in range(self.num_mlp_layers - 1):
                x = self.mlps[i](x)
                x = self.mlp_non_linear(x)
                x = self.dropout(x)
            logits = self.mlps[-1](x)
            probs = self.Softmax(logits)
            return logits, probs, node_emb, graph_emb, []


if __name__ == "__main__":
    from code.Configures import model_args

    model = GCNNet(7, 2, model_args)
