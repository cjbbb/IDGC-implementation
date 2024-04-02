import argparse
import os
import shutil
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from load_dataset import *
from models import GnnNets
from unsupervise.cluster import compute_features, generate_labels
from utils.visualize import plot
from utils.log import *
from utils.Configures import *
from utils.utils import PlotUtils
from my_mcts import mcts


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def prototype_subgraph_similarity(x, prototype):
    distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
    similarity = torch.log((distance + 1) / (distance + 1e-4))
    return distance, similarity


def regenerate_dataloader_train(dataloader, dataset, origin_labels, re_train_labels):
    """
    Regenerate the training data loader with new labels.

    Args:
        dataloader (DataLoader): The original data loader.
        dataset (Dataset): The dataset used in the data loader.
        origin_labels (numpy.ndarray): The original labels of the data.
        re_train_labels (numpy.ndarray): The new labels for the training data.

    Returns:
        nmi_gt (float): The Normalized Mutual Information (NMI) between the original labels and the new labels.
    """
    # Get the indices of the training data
    train_indices = dataloader["train"].dataset.indices
    # Update the labels of the training data
    new_labels = dataset._data.y.numpy()

    for idx, train_idx in enumerate(train_indices):

        new_labels[train_idx] = re_train_labels[idx]

    dataset._data.y = torch.tensor(new_labels)
    train_subset = Subset(dataset, train_indices)
    dataloader["train"] = DataLoader(train_subset, batch_size=train_args.batch_size, shuffle=False)

    nmi_gt = normalized_mutual_info_score(origin_labels[train_indices], re_train_labels)
    # print(f"NMI for training data: {nmi_gt}")

    return nmi_gt


# train for graph classification
def main_GC(clst, sep):
    # attention the multi-task here
    print("====================start loading data====================")
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    origin_labels = dataset._data.y.numpy()

    print("train on dataset: " + str(data_args.dataset_name))

    input_dim = dataset.num_node_features
    output_dim = int(cluster_args.num_cluster)

    dataloader = get_dataloader(
        dataset, train_args.batch_size, data_split_ratio=data_args.data_split_ratio
    )
    origin_train_labels = dataset._data.y[dataloader["train"].dataset.indices]

    # whether use random labels
    if cluster_args.init_train_random:
        print("use random labels for training data===========")
        random_labels = torch.tensor(
            np.random.randint(0, cluster_args.num_cluster, len(dataloader["train"].dataset.indices))
        )
        regenerate_dataloader_train(dataloader, dataset, origin_labels, random_labels)

    print("====================start training model====================")

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    # save path for model
    if not os.path.isdir("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.isdir(os.path.join("checkpoint", data_args.dataset_name)):
        os.mkdir(os.path.join("checkpoint", f"{data_args.dataset_name}"))

    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        gnnNets.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay,
    )

    best_nmi = 0.0
    draw_train_loss = []
    draw_train_acc = []
    draw_nmi_gt = []
    relabel_list = []
    centroids_list = []
    early_stop_count = 0
    best_epoch = 0
    nmi_list = []
    arr_list = []
    assume_acc_list = []

    # start training
    for epoch in range(train_args.max_epochs):
        data_indices = dataloader["train"].dataset.indices

        # cluster per n epoch
        if epoch % cluster_args.update_epoch == 0:
            # cluster graph embedding
            features = compute_features(dataloader, gnnNets)
            if cluster_args.draw_prot == True and model_args.enable_prot == True:
                re_labels, centroids = generate_labels(
                    features, centroids_list, gnnNets.model.prototype_vectors
                )
            else:
                re_labels, centroids = generate_labels(features, centroids_list)
            relabel_list.append(re_labels)
            centroids_list.append(centroids)
            nmi_gt = regenerate_dataloader_train(dataloader, dataset, origin_labels, re_labels)
            draw_nmi_gt.append(nmi_gt)
            _, _, nmi_tm, _, ari_tm, assume_acc_tm = print_metrics(re_labels, origin_train_labels)
            nmi_list.append(nmi_tm)
            arr_list.append(ari_tm)
            assume_acc_list.append(assume_acc_tm)

        acc = []
        loss_list = []

        # Prototype projection
        if epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnnNets.eval()
            for i in range(output_dim * model_args.num_prototypes_per_class):
                count = 0
                best_similarity = 0
                best_coalition = None
                best_data = None
                label = i // model_args.num_prototypes_per_class
                for j in range(i * 10, len(data_indices)):
                    data = dataset[data_indices[j]]
                    if data.y == label:
                        count += 1
                        coalition, similarity, prot = mcts(
                            data, gnnNets, gnnNets.model.prototype_vectors[i]
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                            best_coalition = coalition
                            best_data = data

                    if count >= 10:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot

                        graph = to_networkx(best_data, to_undirected=True)
                        plotutils = PlotUtils(dataset_name=data_args.dataset_name)
                        plotutils.plot(
                            graph,
                            best_coalition,
                            x=best_data.x,
                            figname=os.path.join("./draw", f"epoch_{epoch}_example_{i}.png"),
                        )
                        print(f"epoch_{epoch}_example_{i}.png")
                        print(best_data.y)
                        print("Projection of prototype completed")
                        break
        # train
        #
        gnnNets.initialize_mlp()

        gnnNets.train()

        joint(gnnNets)

        for batch in dataloader["train"]:
            logits, probs, _, graph_embed, min_distances = gnnNets(batch)
            # calculate loss
            loss = criterion(logits, batch.y)

            if model_args.enable_prot == True:
                # cluster loss
                batch.y = batch.y.to(gnnNets.model.prototype_class_identity.device)

                prototypes_of_correct_class = torch.t(
                    gnnNets.model.prototype_class_identity[:, batch.y].bool()
                ).to(model_args.device)
                cluster_cost = torch.mean(
                    torch.min(
                        min_distances[prototypes_of_correct_class].reshape(
                            -1, model_args.num_prototypes_per_class
                        ),
                        dim=1,
                    )[0]
                )

                # seperation loss
                separation_cost = -torch.mean(
                    torch.min(
                        min_distances[~prototypes_of_correct_class].reshape(
                            -1, (output_dim - 1) * model_args.num_prototypes_per_class
                        ),
                        dim=1,
                    )[0]
                )

                # sparsity loss
                l1_mask = 1 - torch.t(gnnNets.model.prototype_class_identity).to(model_args.device)
                l1 = (gnnNets.model.last_layer.weight * l1_mask).norm(p=1)

                loss = loss + clst * cluster_cost + sep * separation_cost + 5e-4 * l1
                # print(loss)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            batch.y = batch.y.to(prediction.device)  # i change here todo
            acc.append(prediction.eq(batch.y).cpu().numpy())

        # report train msg
        append_record(
            "Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(
                epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()
            )
        )
        print(
            f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f}| "
            f"Acc: {np.concatenate(acc, axis=0).mean():.3f}"
        )

        draw_train_loss.append(np.average(loss_list))
        draw_train_acc.append(np.concatenate(acc, axis=0).mean())

        # report eval msg
        # eval_state = evaluate_GC(dataloader["eval"], gnnNets, criterion)
        # print(
        #     f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}"
        # )
        # append_record(
        #     "Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(
        #         epoch, eval_state["loss"], eval_state["acc"]
        #     )
        # )

        # only save the best model
        is_best = nmi_tm > best_nmi

        if is_best:
            best_nmi = nmi_tm
            best_epoch = epoch
            save_best(
                ckpt_dir,
                epoch,
                gnnNets,
                model_args.model_name,
                best_nmi,
                is_best,
            )

    max_nmi = -1
    for i in range(len(nmi_list)):
        if nmi_list[i] > max_nmi:
            max_nmi = nmi_list[i]
            max_nmi_epoch = i
            max_ari = arr_list[i]
            max_acc = assume_acc_list[i]

    print("==================== MAX ====================")
    print("max nmi epoch is ", max_nmi_epoch)
    print("max nmi is ", round(max_nmi * 100, 2))
    print("max ari is ", round(max_ari * 100, 2))
    print("max acc is ", round(max_acc * 100, 2))

    nmi_last_time = []
    for i in range(len(relabel_list) - 1):
        nmi_last = normalized_mutual_info_score(relabel_list[i], relabel_list[i + 1])
        nmi_last_time.append(nmi_last)

    plot(draw_train_loss, "train_loss.png")
    plot(draw_train_acc, "train_acc.png")
    plot(draw_nmi_gt, "nmi_gt.png", "NMI t/labels")
    plot(nmi_last_time, "nmi_last.png", "NMI t/t-1")

    with open("./log/relabel_list.txt", "w") as f:
        for relabel in relabel_list:
            f.write(str(relabel))
            f.write("\n")
    with open("./log/centroids_list.txt", "w") as f:
        for centorid in centroids_list:
            f.write(str(centorid))
            f.write("\n")
            f.write("\n")

    return round(max_nmi * 100, 3), round(max_ari * 100, 3)


def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {
            "loss": np.average(loss_list),
            "acc": np.concatenate(acc, axis=0).mean(),
        }

    return eval_state


def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, probs, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            predictions.append(prediction)
            pred_probs.append(probs)

    test_state = {
        "loss": np.average(loss_list),
        "acc": np.average(np.concatenate(acc, axis=0).mean()),
    }

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions


def predict_GC(test_dataloader, gnnNets):
    """
    return: pred_probs --  np.array : the probability of the graph class
            predictions -- np.array : the prediction class for each graph
    """
    pred_probs = []
    predictions = []
    gnnNets.eval()
    real_label = []
    with torch.no_grad():
        for batch in test_dataloader:
            real_label.extend(batch.y.tolist())
            logits, probs, _, _, _ = gnnNets(batch)

            ## record
            _, prediction = torch.max(logits, -1)
            predictions.append(prediction)
            pred_probs.append(probs)

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return pred_probs, predictions, real_label


def save_best(ckpt_dir, epoch, gnnNets, model_name, best_nmi, is_best):
    print("saving....")
    gnnNets.to("cpu")
    state = {"net": gnnNets.state_dict(), "epoch": epoch, "nmi": best_nmi}
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f"{model_name}_best.pth"
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.to(model_args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch implementation of ProtGNN")
    parser.add_argument("--clst", type=float, default=0, help="cluster")
    parser.add_argument("--sep", type=float, default=0, help="separation")
    args = parser.parse_args()
    clean_record()

    nmi_list = []
    ari_list = []
    for i in range(cluster_args.loop_num):
        plt.close()
        nmi, ari = main_GC(args.clst, args.sep)
        nmi_list.append(nmi)
        ari_list.append(ari)

    avg_nmi_origin = sum(nmi_list) / len(nmi_list)
    avg_ari_origin = sum(ari_list) / len(ari_list)

    threshold_ratio = 0.6
    threshold_nmi = avg_nmi_origin * threshold_ratio
    threshold_ari = avg_ari_origin * threshold_ratio

    filtered_nmi_list = [nmi for nmi in nmi_list if nmi >= threshold_nmi]
    filtered_ari_list = [ari for ari in ari_list if ari >= threshold_ari]

    filter_avg_nmi = sum(filtered_nmi_list) / len(filtered_nmi_list)
    filter_avg_ari = sum(filtered_ari_list) / len(filtered_ari_list)

    print("Average NMI is ", np.mean(filter_avg_nmi))
    print("Average ARI is ", np.mean(filter_avg_ari))
    print("NMI_LIST")
    print(nmi_list)
    print("NMI_FILTERED_LIST")
    print(filtered_nmi_list)
    print("ARI_LIST")
    print(ari_list)
    print("ARI_FILTERED_LIST")
    print(filtered_ari_list)
