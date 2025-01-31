#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-01-30 21:46:13
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-31 02:47:59
FilePath     : /LMP1210_Winter_2025/A2/A2_code.py
Description  : python script for problem 2, 3, 6 in A2
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._forest import ForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Tuple, List


def score_classifier(
    y_test: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    classes = np.unique(y_test)
    assert len(classes) == 2, "Only binary classification is supported"
    TP = np.sum((y_test == 1) & (y_pred == 1))
    TN = np.sum((y_test == 0) & (y_pred == 0))
    FP = np.sum((y_test == 0) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1


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
    # Find the best k, if k is tied, choose the k with the smallest gap between training and validation accuracy
    best_k_ind = np.argmax(acc_valid[:, -1])
    classifier = acc_valid[best_k_ind][0]
    acc_test = classifier.score(X_test, y_test)
    return acc_test


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
    acc_test = classifier.score(X_test, y_test)
    return acc_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_estimators: List = [10, 20, 50, 100],
) -> None:
    acc_train, acc_valid = [], []
    estimators = n_estimators
    for _, n_est in enumerate(estimators):
        classifier = RandomForestClassifier(n_estimators=n_est, random_state=1)
        classifier.fit(X_train, y_train)
        acc_tr = classifier.score(X_train, y_train)
        acc_val = classifier.score(X_valid, y_valid)
        acc_train.append([n_est, acc_tr])
        acc_valid.append([n_est, acc_val])

    acc_train = np.array(acc_train)
    acc_valid = np.array(acc_valid)
    plt.plot(acc_train[:, 0], acc_train[:, 1], label="Training Accuracy")
    plt.plot(acc_valid[:, 0], acc_valid[:, 1], label="Validation Accuracy")
    plt.xticks(estimators)
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.title("Random Forest Accuracy vs n_estimators")
    plt.savefig("Q2a_random_forest_accuracy.png")
    best_n_ind = np.argmax(acc_valid[:, -1])
    best_n = estimators[best_n_ind]
    print("-----------------------Problem 2-----------------------")
    print(f"Best number of trees: {best_n}")
    best_classifier = RandomForestClassifier(n_estimators=best_n, random_state=1)
    best_classifier.fit(X_train, y_train)
    test_acc = best_classifier.score(X_test, y_test)
    print(f"Test accuracy: {test_acc}\n")

    knn_acc_minkow = select_knn_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )
    knn_acc_cosine = select_knn_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test, metric="cosine"
    )
    dt_acc_minL1 = train_decision_tree(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )
    dt_acc_minL2 = train_decision_tree(
        X_train, y_train, X_valid, y_valid, X_test, y_test, min_samples_leaf=2
    )
    dt_acc_minL3 = train_decision_tree(
        X_train, y_train, X_valid, y_valid, X_test, y_test, min_samples_leaf=3
    )

    classifier = LogisticRegression(max_iter=1000, random_state=1)
    classifier.fit(X_train, y_train)
    logistic_acc = classifier.score(X_test, y_test)

    print(f"KNN Minkowski accuracy: {knn_acc_minkow}")
    print(f"KNN Cosine accuracy: {knn_acc_cosine}")
    print(f"Decision Tree min_samples_leaf=1 accuracy: {dt_acc_minL1}")
    print(f"Decision Tree min_samples_leaf=2 accuracy: {dt_acc_minL2}")
    print(f"Decision Tree min_samples_leaf=3 accuracy: {dt_acc_minL3}")
    print(f"Logistic Regression max_iter=1000 accuracy: {logistic_acc}")
    print(f"Random Forest n_estimator=100 accuracy: {test_acc}\n")
    # Define category colors
    category_colors = {
        "KNN": "blue",
        "Decision Tree": "green",
        "Logistic Regression": "red",
        "Random Forest": "orange",
    }
    plt.figure(figsize=(15, 15))
    plt.bar(
        [
            "metric=Minkowski",
            "metric=Cosine",
            "min_sample_leaf=1",
            "min_sample_leaf=2",
            "min_sample_leaf=3",
            "max_iter=1000",
            "n_estimators=100",
        ],
        [
            knn_acc_minkow,
            knn_acc_cosine,
            dt_acc_minL1,
            dt_acc_minL2,
            dt_acc_minL3,
            logistic_acc,
            test_acc,
        ],
        color=[
            "blue",
            "blue",
            "green",
            "green",
            "green",
            "red",
            "orange",
        ],
    )
    plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=16, fontweight="bold")
    plt.title("Comparison of Different Classifiers", fontsize=20, fontweight="bold")
    # Create custom legend
    legend_patches = [
        mpatches.Patch(color=color, label=category)
        for category, color in category_colors.items()
    ]
    plt.legend(
        handles=legend_patches,
        title="Classifier Type",
        fontsize="x-large",
        title_fontsize="x-large",
    )

    plt.savefig("Q2a_classifier_comparison.png")
    return best_classifier


def plot_feature_importance(
    best_classifier: ForestClassifier, X_train: pd.DataFrame, y_train: pd.DataFrame
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    feat_import1 = best_classifier.feature_importances_
    sorted_idx1 = feat_import1.argsort()

    perm = permutation_importance(
        best_classifier, X_train, y_train, scoring="accuracy", random_state=1
    )
    feat_import2 = perm.importances_mean
    sorted_idx2 = feat_import2.argsort()

    ax[0].barh(
        X_train.columns[sorted_idx1],
        feat_import1[sorted_idx1],
    )
    ax[0].set_title("Feature Importance of Random Forest Using Default Method")
    ax[0].set_xlabel("Feature Importance")
    ax[0].set_ylabel("Feature Names")

    ax[1].barh(
        X_train.columns[sorted_idx2],
        feat_import2[sorted_idx2],
    )
    ax[1].set_title("Feature Importance of Random Forest Using Permutation Importances")
    ax[1].set_xlabel("Feature Importance")
    ax[1].set_ylabel("Feature Names")
    fig.tight_layout()
    plt.savefig("Q2b_feature_importance.png")


def train_binary_classification():
    X_data, y_data = make_classification(
        n_samples=1000, n_features=20, weights=[0.9, 0.1], random_state=1
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=1
    )

    classifier = LogisticRegression(random_state=1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy, precision, recall, f1 = score_classifier(y_test, y_pred)
    print("-----------------------Problem 3-----------------------")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    pre1, rec1, _ = precision_recall_curve(
        y_test,
        y_pred,
    )
    _ = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
        ax=ax1, name="ROC Curve", color="darkorange"
    )
    ax1.set_title("ROC Curve", fontsize=20, fontweight="bold")
    ax1.legend(loc="lower right")
    _ = PrecisionRecallDisplay(precision=pre1, recall=rec1).plot(
        ax=ax2, name="PR Curve", color="darkorange"
    )
    ax2.set_title("Precision-Recall Curve", fontsize=20, fontweight="bold")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("Q3a_binary_classification.png")

    new_classifier = LogisticRegression(class_weight="balanced", random_state=1)
    new_classifier.fit(X_train, y_train)
    y_pred_new = new_classifier.predict(X_test)
    accuracy_new, precision_new, recall_new, f1_new = score_classifier(
        y_test, y_pred_new
    )
    print(f"Accuracy: {accuracy_new}")
    print(f"Precision: {precision_new}")
    print(f"Recall: {recall_new}")
    print(f"F1: {f1_new}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_new)
    pre2, rec2, _ = precision_recall_curve(
        y_test,
        y_pred_new,
    )
    _ = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(
        ax=ax1, name="ROC Curve", color="darkorange"
    )
    ax1.set_title("ROC Curve", fontsize=20, fontweight="bold")
    ax1.legend(loc="lower right")
    _ = PrecisionRecallDisplay(precision=pre2, recall=rec2).plot(
        ax=ax2, name="PR Curve", color="darkorange"
    )
    ax2.set_title("Precision-Recall Curve", fontsize=20, fontweight="bold")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig("Q3b_binary_classification.png")


def train_XGBoost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    classifier = XGBClassifier(random_state=1)
    classifier.fit(X_train, y_train)
    xgb_acc = classifier.score(X_test, y_test)
    print("-----------------------Problem 6-----------------------")
    print(f"XGBoost accuracy: {xgb_acc}")

    xgb_importance = classifier.feature_importances_
    xgb_importance /= xgb_importance.max()
    sorted_idx = xgb_importance.argsort()[::-1]
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(
        np.expand_dims(xgb_importance[sorted_idx][:30], axis=0),
        cmap="rainbow",
        xticklabels=X_test.columns[sorted_idx][:30],
        yticklabels=["Importance"],
        cbar=True,
        linewidth=0.5,
        annot=True,
        fmt=".1f",
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, fontweight="bold")
    plt.title(
        "Top 30 Genes - XGBoost Feature Importance", fontsize=20, fontweight="bold"
    )
    plt.savefig("Q6b_xgboost_feature_importance.png")


def train_MLP(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
):
    classifier = MLPClassifier(random_state=1)
    classifier.fit(X_train, y_train)
    mlp_acc = classifier.score(X_test, y_test)
    print(f"MLP accuracy: {mlp_acc}")

    mlp_perm = permutation_importance(
        classifier, X_test, y_test, scoring="accuracy", random_state=1
    )
    mlp_importance = np.nan_to_num(mlp_perm.importances_mean)
    mlp_importance /= mlp_importance.max()
    sorted_idx = mlp_importance.argsort()[::-1]
    plt.figure(figsize=(20, 15))
    ax = sns.heatmap(
        np.expand_dims(mlp_importance[sorted_idx][:30], axis=0),
        cmap="rainbow",
        xticklabels=X_test.columns[sorted_idx][:30],
        yticklabels=["Importance"],
        cbar=True,
        linewidth=0.5,
        annot=True,
        fmt=".1f",
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, fontweight="bold")
    plt.title("Top 30 Genes - MLP Feature Importance", fontsize=20, fontweight="bold")
    plt.savefig("Q6b_mlp_feature_importance.png")


if __name__ == "__main__":
    csv_path = "HW1_data.csv"

    # load and split data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(csv_path)

    #####################################################################
    # Problem 2 - Revisiting patient classification with Random Forests #
    #####################################################################
    best_classifier = train_random_forest(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        n_estimators=[10, 20, 50, 100],
    )
    plot_feature_importance(best_classifier, X_train, y_train)

    ##########################################################
    # Problem 3 - Binary Classification with Imbalanced Data #
    ##########################################################
    train_binary_classification()

    ############################################################
    # Problem 6 - Cell type assignment for single-cell RNA-seq #
    ############################################################
    csv_cell_path = "HW2_data.csv"
    data = pd.read_csv(csv_cell_path)

    data["Cell Type"] = data["Cell Type"].map({"B": 0, "CD14 Monocytes": 1})
    data = data.fillna(0)

    X_data, y_data = (
        data[[i for i in data.columns if i != "Cell Type"]],
        data["Cell Type"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=1
    )

    train_XGBoost(X_train, y_train, X_test, y_test)
    train_MLP(X_train, y_train, X_test, y_test)
