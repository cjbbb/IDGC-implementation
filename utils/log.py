import os
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.Configures import cluster_args
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score

def calculate_precision(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sum_max = np.sum(np.max(cm, axis=0))
    total = np.sum(cm)
    precision = sum_max / total
    return precision

def print_metrics(predictions, real_labels):
    # Initialize confusion matrix
    unique_predictions = set(predictions)
    unique_real_labels = set(real_labels)

    if len(unique_predictions) != len(unique_real_labels):
        nmi = normalized_mutual_info_score(predictions,real_labels)
        ari = adjusted_rand_score(predictions, real_labels)
        print("NMI:", nmi)
        print("ARI:", ari)
        return 0,0,nmi,0,ari,0
    else:

        confusion_matrix = np.zeros((cluster_args.num_cluster+1,cluster_args.num_cluster+1), dtype=int)

        # Populate confusion matrix
        for pred, real in zip(predictions, real_labels):
            confusion_matrix[pred][real] += 1

        # Compute optimal assignment and sum
        rows, cols = linear_sum_assignment(confusion_matrix, maximize=True)
        optimal_sum = confusion_matrix[rows, cols].sum()

        # Compute assumed accuracy
        assumed_acc = optimal_sum / (confusion_matrix.sum() + 1e-4)

        # Compute counts
        real_counts = np.bincount(real_labels, minlength=3)
        pred_counts = np.bincount(predictions, minlength=3)
        assert len(real_labels) == len(predictions)
 
        print("Assumed Accuracy:", assumed_acc)
        print("Real Counts:", real_counts.tolist())
        print("Predicted Counts:", pred_counts.tolist())



        mi = mutual_info_score(predictions, real_labels)
        nmi = normalized_mutual_info_score(predictions,real_labels)
        ri = metrics.rand_score(predictions, real_labels)
        ari = adjusted_rand_score(predictions, real_labels)
        acc = calculate_precision(predictions, real_labels)
        print("MI:", mi)
        print("NMI:", nmi)
        print("RI:", ri)
        print("ARI:", ari)
        print("ACC:", acc)
        return assumed_acc,mi,nmi,ri,ari,acc



def append_record(info):
    f = open("./log/hyper_search", "a")
    if not os.path.exists("./log"):
        os.makedirs("./log")
    f.write(info)
    f.write("\n")
    f.close()


def clean_record():
    if os.path.isfile("./log/hyper_search"):
        with open("./log/hyper_search", "w") as f:
            pass

    directory = "./draw"

    pattern = r"(prot\d+\.png)|(centers\d+\.png)|(init_features_tsne\d+\.png)|(init_features_pca\d+\.png)|(epoch_\d+_example_\d+\.png)|(tsne_anlysis\d+\.png)"

    # Iterate over files in the specified directory
    for filename in os.listdir(directory):
        # If the filename matches the pattern
        if re.match(pattern, filename):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Delete the file
            os.remove(file_path)
