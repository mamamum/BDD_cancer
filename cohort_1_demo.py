from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit
from treeple import HonestForestClassifier
from treeple.datasets import (make_trunk_classification,
                              make_trunk_mixture_classification)
from treeple.stats import PermutationHonestForestClassifier, build_oob_forest
from treeple.stats.utils import _mutual_information
from treeple.tree import MultiViewDecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import os
# from random import shuffle

import pandas as pd
import matplotlib.pyplot as plt

import tree_metrics
from print_importance import might_importance

n_estimators = 100000
max_features = 0.3

MODEL_NAMES = {
    "might": {
        "n_estimators": n_estimators,
        "honest_fraction": 0.5,
        "n_jobs": 40,
        "bootstrap": True,
        "stratify": True,
        "max_samples": 1.6,
        "max_features": 0.3,
        "tree_estimator": MultiViewDecisionTreeClassifier(),
    },
    "rf": {
        "n_estimators": int(n_estimators / 5),
        "max_features": 0.3,
    },
    "knn": {
        # XXX: above, we use sqrt of the total number of samples to allow
        # scaling wrt the number of samples
        # "n_neighbors": 5,
    },
    "svm": {
        "probability": True,
    },
    "lr": {
        "max_iter": 1000,
        "penalty": "l1",
        "solver": "liblinear",
    }
}
might_kwargs = MODEL_NAMES["might"]

# filelist = open("filelist.txt", "r").read().split("\n")[:-1]


# get the sample list
sample_list_file = "ManuscriptFeatureMatrices/AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]
sample_list.head()
# get the sample_ids where cohort is Cohort1
cohort1 = sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"]
print(len(cohort1))
cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]
print(len(cohort2))
PON = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]
# print(cohort1)
sample_list["cohort"].unique()


# define a function to get X and y given a file

def get_X_y(f, root="ManuscriptFeatureMatrices/", cohort=cohort1, verbose=False):
    df = pd.read_csv(root + f)
    non_features = ['Run', 'Library', 'Cancer Status', 'Tumor type', 'Stage', 'Library volume (uL)', 'Library Volume',
                    'UIDs Used', 'Experiment', 'P7', 'P7 Primer', 'MAF']
    sample_ids = df["Sample"]
    # print(sample_ids)
    # if sample is contains "Run" column, remove it
    # print(len(sample_ids))

    for i, sample_id in enumerate(sample_ids):
        if "." in sample_id:
            # print(sample_id.split(".")[1])
            if "Wise" in f or 'ichorCNA' in f:
                sample_ids[i] = sample_id
            else:
                sample_ids[i] = sample_id.split(".")[1]
    target = 'Cancer Status'
    y = df[target]
    # convert the labels to 0 and 1
    y = y.replace("Healthy", 0)
    y = y.replace("Cancer", 1)
    # remove the non-feature columns if they exist
    for col in non_features:
        if col in df.columns:
            df = df.drop(col, axis=1)
    nan_cols = df.isnull().all(axis=0).to_numpy()
    # drop the columns with all nan values
    df = df.loc[:, ~nan_cols]
    # if cohort is not None, filter the samples
    if cohort is not None:
        # filter the rows with cohort1 samples
        X = df[sample_ids.isin(cohort)]
        # print(X.shape)
        y = y[sample_ids.isin(cohort)]
    else:
        X = df
    if "Wise" in f:
        # replace nans with zero
        # print('Wise')
        X = X.fillna(0)
    # impute the nan values with the mean of the column
    X.iloc[:, 1] = X.iloc[:, 1].fillna(X.iloc[:, 1].mean(axis=0))
    # print(X.shape)
    # check if there are nan values
    # nan_rows = X.isnull().any(axis=1)
    nan_cols = X.isnull().all(axis=0)
    # remove the columns with all nan values
    X = X.loc[:, ~nan_cols]
    # print(X.shape)
    if verbose:
        if nan_cols.sum() > 0:
            print(f)
            print(f"nan_cols: {nan_cols.sum()}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        else:
            print(f)
            print(f"X shape: {X.shape}, y shape: {y.shape}")
    # X = X.dropna()
    # y = y.drop(nan_rows.index)

    return X, y


def stratified_train_ml(clf, X, y):
    n_samples = X.shape[0]
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    POS = np.zeros((len(y), 3))

    for idx, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        ### Split Training Set into Fitting Set (40%) and Calibarating Set (40%)
        train_idx = np.arange(
            X_train.shape[0]
        )  # use index array to split, so we can use the same index for the permuted array as well
        fit_idx, cal_idx = train_test_split(
            train_idx, test_size=0.5, random_state=idx, stratify=y_train
        )
        X_fit, X_cal, y_fit, y_cal = (
            X_train[fit_idx],
            X_train[cal_idx],
            y_train[fit_idx],
            y_train[cal_idx],
        )

        POS[test_ix, 0] = y_test
        clf.fit(X_fit, y_fit)
        if X_cal.shape[0] <= 1000:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="sigmoid"
            )
        else:
            calibrated_model = CalibratedClassifierCV(
                clf, cv="prefit", method="isotonic"
            )
        calibrated_model.fit(X_cal, y_cal)
        posterior = calibrated_model.predict_proba(X_test)

        POS[test_ix, 1:] = posterior
    return clf, POS


def run_alog(f1, cohort=cohort1, model_name='might'):
    X_1, y_1 = get_X_y('{}.csv'.format(f1), cohort=cohort, verbose=True)
    X = X_1.iloc[:, 1:]

    if model_name == 'might':
        est = HonestForestClassifier(**might_kwargs)

    elif model_name == "rf":
        est = RandomForestClassifier(**MODEL_NAMES[model_name], n_jobs=40)

    elif "knn" in model_name:
        est = KNeighborsClassifier(n_neighbors=int(np.sqrt(X.shape[0]) + 1), )

    elif model_name == "svm":
        est = SVC(**MODEL_NAMES[model_name])

    elif model_name == "lr":
        est = LogisticRegression(**MODEL_NAMES[model_name])

    # X_combine = X_combine.fillna(0)
    X_combine = X.fillna(0)

    if model_name == 'might':
        est, posterior_arr = build_oob_forest(est, X, y_1, verbose=False, )
    else:
        est, posterior_arr = stratified_train_ml(est, np.array(X_combine), np.array(y_1))
    if model_name == 'might':
        POS = np.nanmean(posterior_arr, axis=0)
    else:
        POS = posterior_arr

    fpr, tpr, thresholds = roc_curve(y_1, POS[:, -1], pos_label=1, drop_intermediate=False, )

    # metrics
    S98 = np.max(tpr[fpr <= 0.02])
    tree_metrics.plot_S98(S98, fpr, tpr, model_name)

    MI = tree_metrics.Calculate_MI(model_name, y_1, POS)

    pAUC = tree_metrics.Calculate_pAUC(model_name, y_1, POS, fpr, tpr)

    hd = tree_metrics.Calculate_hd(model_name, POS)

    # importance
    might_importance(model_name, est, X_combine)

    # save the model
    output_fname = (f"{model_name}.npz")
    print(model_name, f1)
    print(model_name, S98, MI, pAUC, hd)
    np.savez_compressed(
        output_fname,
        model_name=model_name,
        y=y_1,
        S98=S98,
        posterior_arr=posterior_arr,
        MI=MI,
        pAUC=pAUC,
        hd=hd
    )
    return S98


for i in range(20):
    Parallel(n_jobs=20)(delayed(run_alog)(f1='WiseCondorX.Wise-1', cohort=cohort1, model_name=modelname)
                        for modelname in ['might', 'rf', 'knn', 'lr', 'svm'])
