import faiss
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.Configures import *

def tsne(data, num_cluster=2):
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(data)
    return X_tsne


def anlysis_pca_dimension(x):
    # print("x.shape:", x.shape[1])

    pca = PCA(n_components=x.shape[1])
    pca.fit(x)
    # Initialize a variable to store the sum of explained variance
    sum_explained_variance = 0.0

    # Initialize a variable to store the number of components
    num_components = 0

    # For each explained variance ratio
    for ratio in pca.explained_variance_ratio_:
        # Add the ratio to the sum
        sum_explained_variance += ratio
        num_components += 1

        # If the sum exceeds 0.95 (i.e., 95% of the variance)
        if sum_explained_variance >= 0.95:
            # Break the loop
            break

    print("suggest reduce to : ", str(num_components)," dimensions")
    return num_components


def prototype_PCA(prototype_vector):
    pca = PCA(n_components=2, whiten=True)
    pca.fit(prototype_vector)
    prototype_vector = pca.transform(prototype_vector)
    return prototype_vector


def preprocess_features_PCA(x, d=2):
    """
    Calculate PCA + Whitening + L2 normalization for each vector

    Args:
        x (ndarray): N x D, where N is number of vectors, D - dimensionality
        d (int): number of output dimensions (how many principal components to use).
    Returns:
        transformed [N x d] matrix xt .
    """
    orig_d = x.shape[1]
    x = x.astype("float32")

    if d != 128:
        if cluster_args.auto_pca == False:
            pcaw = faiss.PCAMatrix(
                d_in=orig_d, d_out=d, eigen_power=-0.5, random_rotation=False
            )
            pcaw.train(x)
            assert pcaw.is_trained
            x = pcaw.apply_py(x)

        else:
            num_components = anlysis_pca_dimension(x)
            # num_components = num_components if num_components >= d else d
            pca = PCA(n_components=num_components, whiten=True)
            pca.fit(x)
            x = pca.transform(x)
            print(x.shape)

    if cluster_args.pca_norm:
        l2normalization = faiss.NormalizationTransform(x.shape[1], 2.0)
        x = l2normalization.apply_py(x)

    return x


def preprocess_features(x, d=128):
    """
    only L2 normalization for each vector
    """

    l2normalization = faiss.NormalizationTransform(d, 2.0)
    x = l2normalization.apply_py(x)
    return x
