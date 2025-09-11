"""
Persistence utilities for saving and loading model parameters, feature means, etc.
"""

import os
import numpy as np
import joblib

MODEL_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Train', 'TrainMatrix'))

def save_feature_means(case_mean, act_mean, path=None):
    """
    Save the mean feature vectors for case and activity columns to disk.

    Args:
        case_mean (np.ndarray): Mean vector for case columns.
        act_mean (np.ndarray): Mean vector for activity columns.
        path (str, optional): Directory to save. Defaults to MODEL_FOLDER.
    """
    if path is None:
        path = MODEL_FOLDER
    np.save(os.path.join(path, "case_feature_mean.npy"), case_mean)
    np.save(os.path.join(path, "act_feature_mean.npy"), act_mean)

def load_feature_means(path=None):
    """
    Load the mean feature vectors for case and activity columns from disk.

    Args:
        path (str, optional): Directory to load from. Defaults to MODEL_FOLDER.

    Returns:
        (case_mean, act_mean): Tuple of np.ndarrays.
    """
    if path is None:
        path = MODEL_FOLDER
    case_mean = np.load(os.path.join(path, "case_feature_mean.npy"))
    act_mean = np.load(os.path.join(path, "act_feature_mean.npy"))
    return case_mean, act_mean

def save_lr_model(model, path=None):
    """
    Save a trained sklearn LogisticRegression model.

    Args:
        model: The model to save.
        path (str, optional): Full path for model file.
    """
    if path is None:
        path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act.joblib")
    joblib.dump(model, path)

def load_lr_model(path=None):
    """
    Load a saved sklearn LogisticRegression model.

    Args:
        path (str, optional): Full path for model file.

    Returns:
        Loaded model.
    """
    if path is None:
        path = os.path.join(MODEL_FOLDER, "baseline_lr_case_act.joblib")
    return joblib.load(path)