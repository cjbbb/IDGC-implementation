import os
import matplotlib.pyplot as plt
import pandas as pd
from utils.Configures import cluster_args

count = -1


def plot(input_list, draw_name, y_label="value"):
    plt.figure(figsize=(10, 5))
    plt.plot(input_list)
    print(draw_name)
    plt.title(draw_name)
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    if not os.path.exists("./draw"):
        os.makedirs("./draw")
    plt.savefig("./draw/" + draw_name)
    plt.show()


def draw_init_features(data, draw_name, label_clf):

    if cluster_args.draw and data.shape[1] == 2 :
        df = pd.DataFrame(data, index=label_clf, columns=["x", "y"])
        df1 = df[df.index == 0]
        df2 = df[df.index == 1]
        df3 = df[df.index == 2]
        plt.figure(figsize=(10, 8), dpi=80)
        axes = plt.subplot()
        type1 = axes.scatter(
            df1.loc[:, ["x"]], df1.loc[:, ["y"]], s=10, c="red", marker="d"
        )
        type2 = axes.scatter(
            df2.loc[:, ["x"]], df2.loc[:, ["y"]], s=10, c="green", marker="*"
        )
        type3 = axes.scatter(
            df3.loc[:, ["x"]], df3.loc[:, ["y"]], s=10, c="black", marker="p"
        )
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        axes.legend((type1, type2, type3), ("0", "1", "2"), loc=1)
        plt.savefig("./draw/" + draw_name + str(count) + ".png")
        plt.close()


def draw_centers(data, draw_name, centers, label_clf, prototype_vectors=None):
    global count
    count += 1
    if cluster_args.pca_dimension == 2:
        if cluster_args.draw and data.shape[1] == 2:
            df_center = pd.DataFrame(centers, columns=["x", "y"])

            df = pd.DataFrame(data, index=label_clf, columns=["x", "y"])
            df1 = df[df.index == 0]
            df2 = df[df.index == 1]
            df3 = df[df.index == 2]

            plt.figure(figsize=(10, 8), dpi=80)
            axes = plt.subplot()
            type1 = axes.scatter(
                df1.loc[:, ["x"]], df1.loc[:, ["y"]], s=10, c="red", marker="d"
            )
            type2 = axes.scatter(
                df2.loc[:, ["x"]], df2.loc[:, ["y"]], s=10, c="green", marker="*"
            )
            type3 = axes.scatter(
                df3.loc[:, ["x"]], df3.loc[:, ["y"]], s=10, c="black", marker="p"
            )

            type_center = axes.scatter(
                df_center.loc[:, "x"], df_center.loc[:, "y"], s=40, c="blue"
            )
            if prototype_vectors is not None:
                df_prototype = pd.DataFrame(prototype_vectors, columns=["x", "y"])
                type_prototype = axes.scatter(
                    df_prototype.loc[:, "x"],
                    df_prototype.loc[:, "y"],
                    s=50,
                    c="purple",
                    marker="x",
                )
            plt.xlabel("x", fontsize=16)
            plt.ylabel("y", fontsize=16)

            if prototype_vectors is not None:
                axes.legend(
                    (type1, type2, type3, type_center, type_prototype),
                    ("0", "1", "2", "center", "prototype"),
                    loc=1,
                )
            else:
                axes.legend(
                    (type1, type2, type3, type_center), ("0", "1", "2", "center"), loc=1
                )

            plt.show()
            plt.savefig("./draw/" + draw_name + str(count) + ".png")
            plt.close()
