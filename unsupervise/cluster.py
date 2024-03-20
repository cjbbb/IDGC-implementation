import numpy as np
# import faiss
import torch
from unsupervise.kmeans import *
from unsupervise.reduce_dim import *
from utils.Configures import cluster_args
from scipy.optimize import linear_sum_assignment
from utils.visualize import draw_centers, draw_init_features


def compute_features(dataloader, model):
    # input: dataloader, model
    # return graph features (list)
    model.eval()
    feature_shape = 128
    features = torch.empty(0, 128).to("cuda:0")
    for batch in dataloader["train"]:
        logits, probs, node_embed, graph_embed, min_distances = model(batch)

        if graph_embed.shape[1] != feature_shape:
            feature_shape = graph_embed.shape[1]
            features = torch.empty(0, feature_shape).to("cuda:0")

        features = torch.cat((features, graph_embed), 0)
    return features


def rearrange_centroids(centroids, centroids_list):
    cost = [[np.linalg.norm(i - j) for j in centroids] for i in centroids_list[-1]]
    _, re_indx = linear_sum_assignment(cost)
    centroids = [centroids[i] for i in re_indx]
    return centroids



def generate_labels(
    features,
    centroids_list,
    prototypes=None,
    reduce_dim_way="pca",
    ngpu=1,
):
    ndarry_features = features.detach().cpu().numpy()
    anlysis_pca_dimension(ndarry_features)

    if reduce_dim_way == "pca":
        process_features = preprocess_features_PCA(
            ndarry_features, cluster_args.pca_dimension
        )

    if reduce_dim_way == "tsne":
        process_features = tsne(ndarry_features)

    # labels, centroids = spkmeans(
    #     process_features, centroids_list, cluster_args.num_cluster
    # )
    labels, centroids = k_means_scipy(
        process_features, centroids_list, cluster_args.num_cluster
    )

    if prototypes is not None:
        if reduce_dim_way == "pca":
            prototype_features = prototypes.detach().cpu().numpy()
            prototype_process_features = prototype_PCA(prototype_features)

        if reduce_dim_way == "tsne":
            prototype_process_features = tsne(prototype_features)

        draw_centers(
            process_features, "prot", centroids, labels, prototype_process_features
        )

    else:
        draw_centers(process_features, "centers", centroids, labels)

    if cluster_args.draw_tsne_anlysis:
        tsne_feature = tsne(process_features)
        draw_init_features(tsne_feature, "tsne_anlysis", labels)

    return labels, centroids


def cluster_prototype(prototype_vectors):
    ndarry_features = prototype_vectors.detach().cpu().numpy()

    process_features = preprocess_features_PCA(
        ndarry_features, cluster_args.pca_dimension
    )
    if np.isnan(process_features).any():
        print(ndarry_features)
        print(process_features)
        print("NaN values detected in processed features")
        return
    return process_features


 