#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-14 20:26:03
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-14 23:47:13
FilePath     : /LMP1210_Winter_2025/A3/A3_code.py
Description  : python script for problem 1, 4, 5 and 7 in A3
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from typing import Tuple


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


def P1(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    ks = [2, 4, 6, 8]
    print("-----------------------Problem 1-----------------------")
    for k in ks:
        # kmeans model
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=1)
        kmeans.fit(x_train, y_train)
        kmeans_pred = kmeans.predict(x_test)
        kmeans_score = adjusted_rand_score(y_test, kmeans_pred)
        
        # GMM model
        gmm = GaussianMixture(n_components=k, init_params="k-means++", random_state=1)
        gmm.fit(x_train, y_train)
        gmm_pred = gmm.predict(x_test)
        gmm_score = adjusted_rand_score(y_test, gmm_pred)
        print(f"K = {k} - Kmeans ARI score: {kmeans_score}, GMM ARI score: {gmm_score}\n")


def P4(X_data: pd.DataFrame, y_data: pd.DataFrame) -> None:
    print("-----------------------Problem 4a-----------------------")
    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(X_data)
    plt.scatter(X_pca[y_data == 0][:, 0], X_pca[y_data == 0][:, 1], label="B")
    plt.scatter(X_pca[y_data == 1][:, 0], X_pca[y_data == 1][:, 1], label="CD14 Monocytes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of CD14 Monocytes and B cells")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    
    ##############################################
    # Problem 1 - Revisiting Single-Cell RNA-seq #
    ##############################################
    
    csv_cell_path = "A3_data/HW2_data.csv"
    data_P1 = pd.read_csv(csv_cell_path)
    data_P1["Cell Type"] = data_P1["Cell Type"].map({"B": 0, "CD14 Monocytes": 1})
    data_P1 = data_P1.fillna(0)

    X_data_P1, y_data_P1 = (
        data_P1[[i for i in data_P1.columns if i != "Cell Type"]],
        data_P1["Cell Type"],
    )
    X_train_P1, X_test_P1, y_train_P1, y_test_P1 = train_test_split(
        X_data_P1, y_data_P1, test_size=0.3, random_state=1
    )
    
    P1(X_train_P1, y_train_P1, X_test_P1, y_test_P1)
    
    
    ############################################
    # Problem 4 - Principal Component Analysis #
    ############################################
    
    csv_cell_path = "A3_data/HW2_data.csv"
    data_P4 = pd.read_csv(csv_cell_path)
    data_P4["Cell Type"] = data_P4["Cell Type"].map({"B": 0, "CD14 Monocytes": 1})
    data_P4 = data_P4.fillna(0)

    X_data_P4, y_data_P4 = (
        data_P4[[i for i in data_P4.columns if i != "Cell Type"]],
        data_P4["Cell Type"],
    )
    
    P4(X_data_P4, y_data_P4)