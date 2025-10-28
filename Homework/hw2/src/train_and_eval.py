from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def plot_roc(y_trues, y_probas, labels, save_path):
    # figure setup
    plt.title("ROC curve of the Logistic Regression model")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    for y_true, y_proba, label in zip(y_trues, y_probas, labels):
        # metrics preparation
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{label} AUC = {auc:.2f}")
    plt.legend()
    plt.savefig(save_path)
    plt.cla()

def evaluate_model(
    y_trues: List[Union[np.array, pd.Series, pd.DataFrame]],
    y_probas: List[Union[np.array, pd.Series, pd.DataFrame]],
    y_preds: List[Union[np.array, pd.Series, pd.DataFrame]],
    labels: List[str],
    plot: bool=True,
    save_path: str=None
) -> Dict:
    ### BEGIN YOUR SOLUTION ###
    res_dic={}
    if plot and save_path is not None:
        plot_roc(y_trues, y_probas, labels, save_path)
    res_lst=[[] for _ in range(4)]
    for y_true, y_pred in zip(y_trues, y_preds):
        res_lst[0].append(accuracy_score(y_true, y_pred))
        res_lst[1].append(precision_score(y_true, y_pred))
        res_lst[2].append(recall_score(y_true, y_pred))
        res_lst[3].append(f1_score(y_true, y_pred))
    res_dic['accuracy']=res_lst[0]
    res_dic['precision']=res_lst[1]
    res_dic['recall']=res_lst[2]
    res_dic['f1']=res_lst[3]
    return res_dic
    ### END YOUR SOLUTION ###


def train_and_eval(
    X_train: List[Union[np.array, pd.DataFrame]],
    X_val: List[Union[np.array, pd.DataFrame]],
    y_true_train: List[Union[np.array, pd.Series, pd.DataFrame]],
    y_true_val: List[Union[np.array, pd.Series, pd.DataFrame]],
    hyperparams: Dict,
    plot: bool=True,
    save_path: str=None
) -> Tuple[object, Dict]:
    ### BEGIN YOUR SOLUTION ###
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_true_train = np.array(y_true_train)
    y_true_val = np.array(y_true_val)
    clf = LogisticRegression(**hyperparams).fit(X_train, y_true_train)
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_proba_train = clf.predict_proba(X_train)[:, 1]
    y_proba_val = clf.predict_proba(X_val)[:, 1]
    metrics = evaluate_model(
        y_trues=[y_true_train, y_true_val],
        y_probas=[y_proba_train, y_proba_val],
        y_preds=[y_pred_train, y_pred_val],
        labels=['train', 'val'],
        plot=plot,
        save_path=save_path
    )
    return clf, metrics
    ### END YOUR SOLUTION ###