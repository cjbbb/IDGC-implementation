import numpy as np
# import faiss
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from utils.visualize import draw_centers
from utils.Configures import cluster_args
from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine


def spkmeans(data, centroids_list, num_clusters=2, nmb_kmeans_iters=10):
    # print(data)
    torch.set_printoptions(precision=8)

    data = torch.tensor(data).cuda()
    random_idx = torch.randperm(len(data))[:num_clusters]

    centroids = data[random_idx]

    for n_iter in range(nmb_kmeans_iters):
        
        centroids_expanded = centroids.unsqueeze(0)

        tensor_data_expanded = data.unsqueeze(1).expand(-1, centroids.size(0), -1)

        cosine_similarities = F.cosine_similarity(
            tensor_data_expanded, centroids_expanded, dim=2
        )

        _, local_assignments = cosine_similarities.max(dim=1)

        sum_tensor = torch.zeros(num_clusters, data.shape[1]).cuda()
        count_tensor = torch.zeros(num_clusters, dtype=torch.float32).cuda()

        for i in range(data.shape[0]):
            cluster_idx = local_assignments[i].item()
            sum_tensor[cluster_idx] += data[i]
            count_tensor[cluster_idx] += 1

        for i in range(num_clusters):
            if count_tensor[i] > 0:
                sum_tensor[i] /= count_tensor[i]
        # print(sum_tensor)
        centroids = sum_tensor

    # print(centroids)
    centroids = centroids.cpu().numpy()
    local_assignments = local_assignments.cpu().numpy()

    center_labelIdx = {}
    for i in range(len(local_assignments)):
        for j in range(len(centroids)):
            if local_assignments[i] == j:
                center_labelIdx.setdefault(str(centroids[j]), []).append(i)

    if cluster_args.reassign_center and len(centroids_list) > 0:
        cost = []
        for i in centroids_list[-1]:
            cost_i = []
            for j in centroids:
                # cost_i.append(1-cosine(i, j))
                cost_i.append(np.linalg.norm(i - j))
            cost.append(cost_i)
        _, re_indx = linear_sum_assignment(cost)
        centroids = [centroids[i] for i in re_indx]
        for center_idx in range(len(centroids)):
            label_idx = center_labelIdx[str(centroids[center_idx])]
            for lable in label_idx:
                local_assignments[lable] = center_idx
    # print(centroids)

    return local_assignments, centroids


def k_means_scipy(data, centroids_list, num_clusters=2):
    clf = KMeans(num_clusters,n_init=10)
    ydata = clf.fit_predict(data)
    label_clf = clf.labels_

    centers = clf.cluster_centers_
    center_labelIdx = {}
    for i in range(len(label_clf)):
        for j in range(len(centers)):
            if label_clf[i] == j:
                center_labelIdx.setdefault(str(centers[j]), []).append(i)

    # print("Verify Center")
    # print(centers)

    # reassign center
    if cluster_args.reassign_center:
        if len(centroids_list) > 0:
            cost = []
            for i in centroids_list[-1]:
                cost_i = []
                for j in centers:
                    cost_i.append(np.linalg.norm(i - j))
                cost.append(cost_i)
            _, re_indx = linear_sum_assignment(cost)
            centers = [centers[i] for i in re_indx]
            for center_idx in range(len(centers)):
                label_idx = center_labelIdx[str(centers[center_idx])]
                for labl in label_idx:
                    label_clf[labl] = center_idx

    # print("Reassign Center")
    # print(centers)

    return label_clf, centers
