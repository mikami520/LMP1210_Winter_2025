#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-14 20:26:03
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-22 01:21:10
FilePath     : /LMP1210_Winter_2025/A3/A3_code.py
Description  : python script for problem 1, 4, 5 and 7 in A3
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import umap as UMAP
from autoencoder import AutoEncoderTrainer, check_device
from typing import Tuple
import time
import warnings
import os
import argparse

# Disable OpenMP warnings
os.environ["KMP_WARNINGS"] = "0"
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


def load_data(
    csv_path: str,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    data = pd.read_csv(csv_path)
    # Onehot Encoding Gender
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    data = data.fillna(0)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=1)
    valid_data, test_data = train_test_split(temp_data, test_size=2 / 3, random_state=1)
    # Check sizes of the splits
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(valid_data)}")
    print(f"Test set size: {len(test_data)}")
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def plot_decision_boundary(clf, X, y, title: str) -> None:
    class_labels = np.unique(y)
    colors = ["cyan", "orange"]
    cell_types = ["B", "CD14 Monocytes"]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    for i, c in enumerate(class_labels):
        plt.scatter(
            X[y == c, 0],
            X[y == c, 1],
            color=colors[i],
            label=f"{cell_types[i]}",
            edgecolors="k",
            alpha=0.7,
        )

    plt.xlabel("Second Largest PC" if "Top" in title else "Lowest PC")
    plt.ylabel("Largest PC" if "Top" in title else "Second Lowest PC")
    plt.title(
        "Scatter plot and Decision Boundary of Largest PC and Second Largest PC"
        if "Top" in title
        else "Scatter plot and Decision Boundary of Smallest PC and Second Smallest PC"
    )
    plt.legend(title="Cell Types")
    fig.savefig(f"Q4Pd{title}.png")


def P1(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    ks = [2, 4, 6, 8]
    print("-----------------------Problem 1-----------------------\n")
    for k in ks:
        start_kmeans = time.time()
        # kmeans model
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=4, random_state=0)
        kmeans.fit(x_train)
        kmeans_pred = kmeans.predict(x_test)
        kmeans_score = adjusted_rand_score(y_test, kmeans_pred)
        end_kmeans = time.time()
        print(
            f"K = {k} - Kmeans ARI score: {kmeans_score}, time: {end_kmeans - start_kmeans}\n"
        )
        # GMM model
        start_gmm = time.time()
        gmm = GaussianMixture(
            n_components=k, init_params="k-means++", n_init=4, random_state=0
        )
        gmm.fit(x_train)
        gmm_pred = gmm.predict(x_test)
        gmm_score = adjusted_rand_score(y_test, gmm_pred)
        end_gmm = time.time()
        print(f"K = {k} - GMM ARI score: {gmm_score}, time: {end_gmm - start_gmm}\n")


def P4(X_data: pd.DataFrame, y_data: pd.DataFrame) -> None:
    print("-----------------------Problem 4a-----------------------\n")
    pca = PCA(n_components=15, random_state=17)
    pca.fit(X_data)
    # Extract explained variance and variance ratio
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    # Print results
    print(f"Explained Variance:\n {explained_variance}\n")
    print(f"Explained Variance Ratio:\n {explained_variance_ratio}\n")

    # Identify the most important principal component
    most_important_pc = explained_variance_ratio.argmax() + 1
    print(
        f"The most important principal component is PC{most_important_pc} with {explained_variance_ratio[most_important_pc - 1]:.2%} variance explained.\n"
    )

    print("-----------------------Problem 4b-----------------------\n")
    pca = PCA(random_state=17)
    pca.fit(X_data)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    print(
        f"Number of components needed to explain 80% variance: {np.argmax(cumulative_variance >= 0.8) + 1}\n"
    )

    print("-----------------------Problem 4c-----------------------\n")
    pca = PCA(n_components=15, random_state=17)
    X_pca = pca.fit_transform(X_data)
    # Create legend with custom colors
    class_labels = np.unique(y_data)
    colors = ["cyan", "orange"]
    cell_types = ["B", "CD14 Monocytes"]

    pc_high1 = X_pca[:, 0]
    pc_high2 = X_pca[:, 1]
    fig = plt.figure(figsize=(8, 8))
    for i, c in enumerate(class_labels):
        plt.scatter(
            pc_high2[y_data == c],
            pc_high1[y_data == c],
            color=colors[i],
            label=f"{cell_types[i]}",
            edgecolors="k",
            alpha=0.7,
        )
    plt.xlabel("Second Largest PC")
    plt.ylabel("Largest PC")
    plt.title("Scatter plot of Largest PC and Second Largest PC")
    plt.legend(title="Cell Types")
    fig.savefig("Q4Pc_largePC.png")

    fig = plt.figure(figsize=(8, 8))
    pc_low1 = X_pca[:, -1]
    pc_low2 = X_pca[:, -2]
    for i, c in enumerate(class_labels):
        plt.scatter(
            pc_low1[y_data == c],
            pc_low2[y_data == c],
            color=colors[i],
            label=f"{cell_types[i]}",
            edgecolors="k",
            alpha=0.7,
        )
    plt.xlabel("Smallest PC")
    plt.ylabel("Second Smallest PC")
    plt.title("Scatter plot of Smallest PC and Second Smallest PC")
    plt.legend(title="Cell Types")
    fig.savefig("Q4Pc_smallPC.png")

    print(
        "Saving scatter plots of largest PC and second largest PC as 'Q4Pc_largePC.png'"
    )
    print(
        "Saving scatter plots of smallest PC and second smallest PC as 'Q4Pc_smallPC.png'\n"
    )

    print("-----------------------Problem 4d-----------------------\n")
    classifier_high = LogisticRegression(random_state=17)
    X_pca_high = np.array([X_pca[:, 1], X_pca[:, 0]]).T
    classifier_high.fit(X_pca_high, y_data)
    score_high = classifier_high.score(X_pca_high, y_data)
    print(f"Accuracy of the classifier using the two largest PCs: {score_high:.2%}\n")

    classifier_low = LogisticRegression(random_state=17)
    X_pca_low = np.array([X_pca[:, -1], X_pca[:, -2]]).T
    classifier_low.fit(X_pca_low, y_data)
    score_low = classifier_low.score(X_pca_low, y_data)

    print(f"Accuracy of the classifier using the two smallest PCs: {score_low:.2%}\n")

    plot_decision_boundary(classifier_high, X_pca_high, y_data, "Top")
    plot_decision_boundary(classifier_low, X_pca_low, y_data, "Bottom")

    print("-----------------------Problem 4e-----------------------\n")
    pca = PCA(random_state=17)
    X_pca = pca.fit_transform(X_data)
    scores = []
    for k in range(1, 11):
        X_pca_k = X_pca[:, :k]
        print(X_pca_k.shape)
        clf = LogisticRegression(random_state=17)
        clf.fit(X_pca_k, y_data)
        score = clf.score(X_pca_k, y_data)
        scores.append([k, score])

    scores = np.array(scores)
    fig = plt.figure(figsize=(8, 8))
    plt.plot(scores[:, 0], scores[:, 1], marker="o")
    plt.xlabel("Number of Top Principal Components k")
    plt.xticks(np.arange(1, 11))
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Top Principal Components")
    fig.savefig("Q4Pe.png")
    print(
        "Saving plot of accuracy vs number of top principal components as 'Q4Pe.png'\n"
    )


def P5(
    rna_seq: pd.DataFrame,
    dna_seq: pd.DataFrame,
    labels: pd.DataFrame,
    pretrain_path: str = None,
    train: bool = False,
) -> None:
    class_labels = np.unique(labels["label"])
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    indexed_labels = np.array([label_to_index[label] for label in labels["label"]])
    cmap = plt.get_cmap("tab10")
    # Create legend with custom colors
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cmap((i - 1) / (len(class_labels) - 1)),
            markersize=10,
            label=f"Subtype {i}",
        )
        for i in class_labels
    ]
    print("-----------------------Problem 5a-----------------------\n")
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    pca = PCA(n_components=2, random_state=17)
    rna_pca = pca.fit_transform(rna_seq)
    ax[0].scatter(
        rna_pca[:, 0],
        rna_pca[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("PCA Scatter plot of RNA-seq data")
    ax[0].legend(handles=handles, title="Cancer Subtypes")

    tsne = TSNE(n_components=2, random_state=17)
    rna_tsne = tsne.fit_transform(rna_seq)
    ax[1].scatter(
        rna_tsne[:, 0],
        rna_tsne[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[1].set_xlabel("t-SNE Embedding 1")
    ax[1].set_ylabel("t-SNE Embedding 2")
    ax[1].set_title("t-SNE Scatter plot of RNA-seq data")
    ax[1].legend(handles=handles, title="Cancer Subtypes")

    umap = UMAP.UMAP(n_components=2, random_state=17)
    rna_umap = umap.fit_transform(rna_seq)
    ax[2].scatter(
        rna_umap[:, 0],
        rna_umap[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[2].set_xlabel("UMAP Embedding 1")
    ax[2].set_ylabel("UMAP Embedding 2")
    ax[2].set_title("UMAP Scatter plot of RNA-seq data")
    ax[2].legend(handles=handles, title="Cancer Subtypes")
    plt.tight_layout()
    fig.savefig("Q5Pa.png")
    print("Saving scatter plots of PCA, t-SNE and UMAP as 'Q5Pa.png'\n")

    kmean = KMeans(n_clusters=len(class_labels), random_state=17)
    kmean.fit(rna_seq)
    kmean_pred = kmean.predict(rna_seq)
    kmean_score = adjusted_rand_score(indexed_labels, kmean_pred)
    print(f"Kmeans ARI score for RNA-seq data: {kmean_score}\n")

    print("-----------------------Problem 5b-----------------------\n")
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    pca = PCA(n_components=2, random_state=17)
    dna_pca = pca.fit_transform(dna_seq)
    ax[0].scatter(
        dna_pca[:, 0],
        dna_pca[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("PCA Scatter plot of Methylation data")
    ax[0].legend(handles=handles, title="Cancer Subtypes")

    tsne = TSNE(n_components=2, random_state=17)
    dna_tsne = tsne.fit_transform(dna_seq)
    ax[1].scatter(
        dna_tsne[:, 0],
        dna_tsne[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[1].set_xlabel("t-SNE Embedding 1")
    ax[1].set_ylabel("t-SNE Embedding 2")
    ax[1].set_title("t-SNE Scatter plot of Methylation data")
    ax[1].legend(handles=handles, title="Cancer Subtypes")

    umap = UMAP.UMAP(n_components=2, random_state=17)
    dna_umap = umap.fit_transform(dna_seq)
    ax[2].scatter(
        dna_umap[:, 0],
        dna_umap[:, 1],
        c=indexed_labels,
        cmap=cmap,
        edgecolors="k",
        alpha=0.7,
    )
    ax[2].set_xlabel("UMAP Embedding 1")
    ax[2].set_ylabel("UMAP Embedding 2")
    ax[2].set_title("UMAP Scatter plot of Methylation data")
    ax[2].legend(handles=handles, title="Cancer Subtypes")
    plt.tight_layout()
    fig.savefig("Q5Pb.png")
    print("Saving scatter plots of PCA, t-SNE and UMAP as 'Q5Pb.png'\n")

    kmean = KMeans(n_clusters=len(class_labels), random_state=17)
    kmean.fit(dna_seq)
    kmean_pred = kmean.predict(dna_seq)
    kmean_score = adjusted_rand_score(indexed_labels, kmean_pred)
    print(f"Kmeans ARI score for Methylation data: {kmean_score}\n")

    print("-----------------------Problem 5c-----------------------\n")

    # Concatenate RNA-seq and Methylation data
    combined_seq = np.concatenate((rna_seq, dna_seq), axis=1)
    num_features = combined_seq.shape[1]

    combined_seq_tensor = torch.tensor(combined_seq, dtype=torch.float32)
    combined_seq_dataloader = torch.utils.data.DataLoader(
        combined_seq_tensor, batch_size=64, shuffle=True
    )

    device = check_device()
    autoencoder = AutoEncoderTrainer(
        input_dim=num_features, num_epochs=3000, device=device
    )
    if train:
        autoencoder.train(combined_seq_dataloader)
    latent = autoencoder.get_latent_representation(combined_seq_tensor, pretrain_path)
    latent = latent.detach().cpu().numpy()

    umap = UMAP.UMAP(n_components=2, random_state=6)
    combined_umap = umap.fit_transform(latent)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(
        combined_umap[:, 0],
        combined_umap[:, 1],
        c=indexed_labels,
        cmap="tab10",
        edgecolors="k",
        alpha=0.7,
    )
    plt.xlabel("UMAP Embedding 1")
    plt.ylabel("UMAP Embedding 2")
    plt.title("UMAP Visualization of Patient Latent Representations")
    plt.legend(handles=handles, title="Cancer Subtypes")
    fig.savefig("Q5Pc.png")

    kmeans = KMeans(n_clusters=len(class_labels), random_state=48, n_init=10)
    kmeans_pred = kmeans.fit_predict(latent)
    kmeans_ami_score = adjusted_mutual_info_score(indexed_labels, kmeans_pred)
    print(f"Kmeans AMI score for latent representation: {kmeans_ami_score}\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--train", action="store_true", help="Train the autoencoder model"
    )
    argparser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Path to the pre-trained autoencoder model",
    )
    args = argparser.parse_args()
    train = args.train
    pretrain_path = args.pretrain

    ##############################################
    # Problem 1 - Revisiting Single-Cell RNA-seq #
    ##############################################

    csv_cell_path = "A3_data/HW2_data.csv"
    data_P1 = pd.read_csv(csv_cell_path).dropna(how="all")
    data_P1["Cell Type"] = data_P1["Cell Type"].map({"B": 0, "CD14 Monocytes": 1})
    data_P1 = data_P1.fillna(0)

    X_data_P1, y_data_P1 = (
        data_P1[[i for i in data_P1.columns if i != "Cell Type"]],
        data_P1["Cell Type"],
    )
    X_train_P1, X_test_P1, y_train_P1, y_test_P1 = train_test_split(
        X_data_P1, y_data_P1, test_size=0.3, random_state=1
    )

    # P1(X_train_P1, y_train_P1, X_test_P1, y_test_P1)

    ############################################
    # Problem 4 - Principal Component Analysis #
    ############################################

    csv_cell_path = "A3_data/HW2_data.csv"
    data_P4 = pd.read_csv(csv_cell_path).dropna(how="all")
    data_P4["Cell Type"] = data_P4["Cell Type"].map({"B": 0, "CD14 Monocytes": 1})

    X_data_P4, y_data_P4 = (
        data_P4[[i for i in data_P4.columns if i != "Cell Type"]],
        data_P4["Cell Type"],
    )

    # P4(X_data_P4, y_data_P4)

    #########################################################
    # Problem 5 - Multi-Omics Analysis for Cancer Subtyping #
    #########################################################

    rna_seq = (
        pd.read_csv("A3_data/A3RNAseq.csv")
        .dropna(how="all")
        .drop(columns=["Unnamed: 0"])
    )
    dna_seq = (
        pd.read_csv("A3_data/A3Methylation.csv")
        .dropna(how="all")
        .drop(columns=["Unnamed: 0"])
    )
    labels = pd.read_csv("A3_data/label.csv").dropna(how="all")

    P5(rna_seq, dna_seq, labels, pretrain_path, train)
