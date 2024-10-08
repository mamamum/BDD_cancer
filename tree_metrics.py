import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve
import os


def Calculate_MI(model_name, y_true, y_pred_proba):
    # calculate the conditional entropy
    if model_name == 'might':
        H_YX = np.mean(entropy(y_pred_proba, base=np.exp(1), axis=1))
    else:
        H_YX = np.mean(entropy(y_pred_proba[:, 1:], base=np.exp(1), axis=1))

    # empirical count of each class (n_classes)
    _, counts = np.unique(y_true, return_counts=True)
    # calculate the entropy of labels
    H_Y = entropy(counts, base=np.exp(1))
    return H_Y - H_YX


def Calculate_hd(model_name, y_pred_proba) -> float:
    if model_name == 'might':
        return np.sqrt(
            np.sum((np.sqrt(y_pred_proba[:, 1]) - np.sqrt(y_pred_proba[:, 0])) ** 2)
        ) / np.sqrt(2)
    else:
        return np.sqrt(
            np.sum((np.sqrt(y_pred_proba[:, 2]) - np.sqrt(y_pred_proba[:, 1])) ** 2)
        ) / np.sqrt(2)


def plot_S98(S98, fpr, tpr, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    ax.tick_params(labelsize=15)
    ax.set_xlim([-0.005, 1.005])
    ax.set_ylim([-0.005, 1.005])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)

    ax.plot(fpr, tpr, label="ROC curve")

    spec = int((1 - 0.02) * 100)
    ax.axvline(
        x=0.02,
        ymin=0,
        ymax=S98,
        color="r",
        label="S@" + str(spec) + " = " + str(round(S98, 2)),
        linestyle="--",
    )
    ax.axhline(y=S98, xmin=0, xmax=0.02, color="r", linestyle="--")
    ax.legend(frameon=False, fontsize=15)
    plt.title('S98-' + model_name)
    save_path = os.path.join('figures', model_name + '_S@98.png')
    plt.savefig(save_path)
    return


def Calculate_pAUC(model_name, y_true, y_pred_proba, fpr, tpr, max_fpr=0.1) -> float:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    ax.tick_params(labelsize=15)
    ax.set_xlim([-0.005, 1.005])
    ax.set_ylim([-0.005, 1.005])
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.plot(fpr, tpr, label="ROC curve")

    # Calculate pAUC at the specific threshold
    if model_name == 'might':
        pAUC = roc_auc_score(y_true, y_pred_proba[:, 1], max_fpr=max_fpr)
    else:
        pAUC = roc_auc_score(y_true, y_pred_proba[:, 2], max_fpr=max_fpr)

    pos = np.where(fpr <= max_fpr)[0][-1]
    ax.fill_between(
        fpr[:pos],
        tpr[:pos],
        alpha=0.6,
        color="r",
        label="pAUC@90 = " + str(round(pAUC, 2)),
        linestyle="--",
    )
    ax.legend(frameon=False, fontsize=15)
    plt.title('pAUC-' + model_name)
    save_path = os.path.join('figures', model_name + '_pAUC.png')
    plt.savefig(save_path)
    return pAUC
