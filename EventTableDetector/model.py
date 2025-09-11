"""
Model management: training, loading, and saving for field classifiers and enhanced baseline LR.
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .feature_extraction import extract_column_features_and_labels_from_dir

MODEL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix'))

def train_column_classifier(
    train_dir: str,
    nrows: int = 1000,
    max_word_threshold: int = 6,
    method: str = "logistic",
    params: dict = None,
    verbose: bool = True,
    save_model: bool = True,
    model_folder: str = None,
):
    """
    Train a column classifier for case/activity/irrelevant.

    Args:
        train_dir (str): Directory with training CSVs.
        nrows (int): Max rows per file.
        max_word_threshold (int): Screening param.
        method (str): "logistic" or "tree".
        params (dict): Extra sklearn parameters.
        verbose (bool): Print info.
        save_model (bool): Whether to persist model.
        model_folder (str): Where to store model.

    Returns:
        (model, feature_names)
    """
    X, y = extract_column_features_and_labels_from_dir(
        train_dir=train_dir, nrows=nrows,
        max_word_threshold=max_word_threshold, verbose=verbose
    )
    feature_names = ["n_unique_ratio_norm", "max_freq_norm", "entropy_norm", "sim_norm"]
    params = params or {}
    if method == "logistic":
        clf = LogisticRegression(max_iter=1000, multi_class="auto", **params)
        model_name = "column_classifier_logistic.joblib"
        coef_name = "column_classifier_logistic_coef.npy"
        intercept_name = "column_classifier_logistic_intercept.npy"
    elif method == "tree":
        clf = DecisionTreeClassifier(**params)
        model_name = "column_classifier_tree.joblib"
        featimp_name = "column_classifier_tree_featimp.npy"
    else:
        raise ValueError("method must be 'logistic' or 'tree'")
    clf.fit(X, y)
    if verbose:
        print(f"Trained {method} classifier on {X.shape[0]} samples.")
    if save_model:
        if model_folder is None:
            model_folder = MODEL_FOLDER
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, model_name)
        joblib.dump(clf, model_path)
        if verbose:
            print(f"Saved {method} model to {model_path}")
        if method == "logistic":
            coef_path = os.path.join(model_folder, coef_name)
            intercept_path = os.path.join(model_folder, intercept_name)
            np.save(coef_path, clf.coef_)
            np.save(intercept_path, clf.intercept_)
            if verbose:
                print(f"Saved logistic coefficients to {coef_path}, intercept to {intercept_path}")
        elif method == "tree":
            featimp_path = os.path.join(model_folder, featimp_name)
            np.save(featimp_path, clf.feature_importances_)
            if verbose:
                print(f"Saved tree feature importances to {featimp_path}")
    return clf, feature_names

def load_column_classifier(model_folder: str = None, method: str = "logistic"):
    """
    Load a previously trained column classifier from disk.

    Args:
        model_folder (str): Path to model folder.
        method (str): "logistic" or "tree".

    Returns:
        sklearn estimator
    """
    if model_folder is None:
        model_folder = MODEL_FOLDER
    if method == "logistic":
        model_name = "column_classifier_logistic.joblib"
    elif method == "tree":
        model_name = "column_classifier_tree.joblib"
    else:
        raise ValueError("method must be 'logistic' or 'tree'")
    model_path = os.path.join(model_folder, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def train_baseline_lr(train_dir: str, verbose: bool = True):
    """
    Train and save a logistic regression for enhanced baseline candidate scoring.

    Args:
        train_dir (str): Directory with training CSVs.
        verbose (bool): Print info.

    Returns:
        trained sklearn LogisticRegression
    """
    X_train, y_train = extract_column_features_and_labels_from_dir(train_dir, nrows=None, verbose=verbose)
    mask = y_train.isin(["case", "activity"])
    X_bin = X_train[mask]
    y_bin = y_train[mask].replace({"case": 0, "activity": 1})
    if len(X_bin) == 0 or y_bin.nunique() < 2:
        raise ValueError("Not enough data to train baseline LR.")
    lr = LogisticRegression(solver='lbfgs', max_iter=200, tol=1e-3, n_jobs=-1)
    lr.fit(X_bin, y_bin)
    model_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act.joblib")
    coef_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act_coef.npy")
    intercept_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act_intercept.npy")
    joblib.dump(lr, model_path)
    np.save(coef_path, lr.coef_)
    np.save(intercept_path, lr.intercept_)
    if verbose:
        print(f"Trained and saved baseline LR model to {model_path}")
    return lr

def load_baseline_lr(train_dir: str = None):
    """
    Load a trained baseline LR model for enhanced baseline candidate scoring.

    Args:
        train_dir (str): Only needed for retraining if not found.

    Returns:
        sklearn LogisticRegression
    """
    import joblib
    model_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act.joblib")
    coef_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act_coef.npy")
    intercept_path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act_intercept.npy")
    if os.path.exists(model_path) and os.path.exists(coef_path) and os.path.exists(intercept_path):
        lr = joblib.load(model_path)
        lr.coef_ = np.load(coef_path)
        lr.intercept_ = np.load(intercept_path)
        return lr
    if train_dir:
        return train_baseline_lr(train_dir)
    else:
        raise FileNotFoundError("Baseline LR model and parameters not found.")