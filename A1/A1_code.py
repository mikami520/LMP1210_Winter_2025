#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-17 02:43:16
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-29 11:31:49
FilePath     : /LMP1210_Winter_2025/A1/A1_code.py
Description  : python script for problem 4,5,6 in A1
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Source
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


def select_knn_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    metric: str = "minkowski",
) -> None:
    acc_train, acc_valid = [], []
    for k in range(1, 21):
        classifier = KNeighborsClassifier(k, metric=metric)
        classifier.fit(X_train, y_train)
        acc_tr = classifier.score(X_train, y_train)
        acc_val = classifier.score(X_valid, y_valid)
        acc_train.append([k, acc_tr])
        acc_valid.append([classifier, k, acc_val])

    acc_train = np.array(acc_train)
    acc_valid = np.array(acc_valid)
    fig = plt.figure()
    plt.plot(acc_train[:, 0], acc_train[:, -1], label="Training Accuracy")
    plt.plot(acc_valid[:, 1], acc_valid[:, -1], label="Validation Accuracy")
    plt.xticks(range(1, 21))
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.title(f"KNN Accuracy vs k using {metric} distance")
    fig.savefig(f"knn_accuracy_{metric}.png")

    # Find the best k, if k is tied, choose the k with the smallest gap between training and validation accuracy
    best_k_ind = np.argmax(acc_valid[:, -1])
    best_k = acc_valid[best_k_ind][1]
    best_acc = acc_valid[best_k_ind][2]
    print(
        f"-----------------------{metric.capitalize()} Distance-----------------------"
    )
    print(f"Best k: {int(best_k)} / Best KNN validation accuracy: {best_acc}")
    acc_test = acc_valid[best_k_ind][0].score(X_test, y_test)
    print(f"Test accuracy: {acc_test}\n")


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    min_samples_leaf: int = 1,
) -> None:
    classifier = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        min_samples_leaf=min_samples_leaf,
        random_state=1,
    )
    classifier.fit(X_train, y_train)
    acc_train = classifier.score(X_train, y_train)
    acc_valid = classifier.score(X_valid, y_valid)
    acc_test = classifier.score(X_test, y_test)
    print(
        f"------------Decision Tree with Min Sample Leaf {min_samples_leaf}-------------"
    )
    print(f"Training Accuracy: {acc_train}")
    print(f"Validation Accuracy: {acc_valid}")
    print(f"Test Accuracy: {acc_test}\n")

    graph = Source(
        export_graphviz(classifier, out_file=None, feature_names=X_train.columns)
    )
    graph.format = "png"
    graph.render(f"dt_min_sample_leaf_{min_samples_leaf}", view=True)


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    classifier = LogisticRegression(max_iter=1000, random_state=1)
    classifier.fit(X_train, y_train)
    acc_train = classifier.score(X_train, y_train)
    acc_valid = classifier.score(X_valid, y_valid)
    acc_test = classifier.score(X_test, y_test)
    print("---------------------Logistic Regression---------------------")
    print(f"Training Accuracy: {acc_train}")
    print(f"Validation Accuracy: {acc_valid}")
    print(f"Test Accuracy: {acc_test}")


if __name__ == "__main__":
    csv_path = "HW1_data.csv"

    # load and split data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(csv_path)

    ######################################################
    # Problem 4 - Classification with Nearest Neighbours #
    ######################################################
    select_knn_model(X_train, y_train, X_valid, y_valid, X_test, y_test)
    select_knn_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test, metric="cosine"
    )

    #################################################
    # Problem 5 - Classification with Decision Tree #
    #################################################
    train_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_decision_tree(
        X_train, y_train, X_valid, y_valid, X_test, y_test, min_samples_leaf=2
    )
    train_decision_tree(
        X_train, y_train, X_valid, y_valid, X_test, y_test, min_samples_leaf=3
    )

    #######################################################
    # Problem 6 - Classification with Logistic Regression #
    #######################################################
    train_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
